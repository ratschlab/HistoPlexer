import os 
import glob
import numpy as np
import json 
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

import ast
import math
import argparse
import h5py
import openslide
import sys 
import time
from types import SimpleNamespace
from src.utils.imc_feats_utils import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.models.generator import unet_translator

parser = argparse.ArgumentParser(description='Arguments to extract features for HE for multimodal task')
parser.add_argument('--device', type=str, required=False, default='cuda:0', help='which cuda gpu to run code on')
parser.add_argument('--metafile', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/tcga_prognosis/tcga-skcm_meta_immune_wsi.csv', help='metadata for tcga sample')
parser.add_argument('--histoplexer_model_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/checkpoint-step_495000.pt', help='Histoplexer trained translator model path')
parser.add_argument('--resnet_chkpt_path', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/tcga_prognosis/resnet18-f37072fd.pth', help='Resnet model path')
parser.add_argument('--he_feats_path', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/tcga_prognosis/resnet18_SSL_features_tcga', help='Path to saved features he features, output_dir from he_feats.py')
parser.add_argument('--save_path', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/tcga_prognosis/tcga_he-imc_feats', help='Path to saved features')
parser.add_argument('--dump_batches', type=str2bool, required=False, default=True, help='If true then features are written for each batch, else all features collected for sample and then dumped')

args = parser.parse_args()
metafile = args.metafile
device = args.device
histoplexer_model_path = args.histoplexer_model_path
resnet_chkpt_path = args.resnet_chkpt_path
he_feats_path = args.he_feats_path
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

# --- loading translator ---
# config file
config_path = os.path.dirname(histoplexer_model_path) + '/config.json'
with open(config_path, "r") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
# initialize model
translator = unet_translator(
    input_nc=config.input_nc,
    output_nc=config.output_nc,
    use_high_res=config.use_high_res,
    use_multiscale=config.use_multiscale,
    ngf=config.ngf,
    depth=config.depth,
    encoder_padding=config.encoder_padding,
    decoder_padding=config.decoder_padding, 
    device="cpu", 
    extra_feature_size=config.fm_feature_size
)

checkpoint = torch.load(histoplexer_model_path)
translator.load_state_dict(checkpoint['trans_ema_state_dict']) # trans_state_dict
translator.to(device)
translator.eval() 

# --- loading resnet model with weights --- 
resnet = torchvision.models.__dict__['resnet18'](pretrained=False)
resnet_state = torch.load(resnet_chkpt_path, map_location=device)
resnet.load_state_dict(resnet_state)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet = resnet.to(device)
resnet.eval()

# -- defining hooks -- 
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
# global avg pooling 
gap = nn.AdaptiveAvgPool2d(1) 
# translator hook in center block, can also do "translator_model.center_block[0].conv1"
translator.center_block[0].register_forward_hook(get_features('feats_translator'))

# --------------------------------
# ----Loading HE feature file ----
# --------------------------------
df_meta = pd.read_csv(metafile)[['case_id', 'wsi_paths', 'wsi_count']]
df_meta['wsi_paths'] = df_meta['wsi_paths'].fillna("")
df_meta['wsi_paths'] = df_meta['wsi_paths'].apply(lambda x: ast.literal_eval(x) if x else [])
df_meta['wsi_count'].value_counts()

batch_size = 128
workers = 8 
    
for index, row in df_meta.iterrows():
    print('\n')
    print("case_id:", row['case_id'])
    print("wsi_count:", row['wsi_count'])
    print(row['wsi_paths'])

    if  row['wsi_count'] >= 1: 
        # h5 file for saving 
        save_path_caseid = os.path.join(save_path, row['case_id'] + '.h5')
        print(os.path.exists(save_path_caseid))
        if os.path.exists(save_path_caseid): 
            continue 
        else: 
            with h5py.File(save_path_caseid, 'a') as hf:
                for wsi_path in row['wsi_paths']: 
                    time_start = time.time()
                    wsi_name = wsi_path.split('/')[-1].split('.svs')[0]
                    wsi_group = hf.create_group(wsi_name)
                    feats_path = os.path.join(he_feats_path, wsi_name + '_features.h5')
                    if os.path.exists(feats_path): 
                        # load h5 file 
                        with h5py.File(feats_path,'r') as hdf5_file:
                            feats_he = hdf5_file['features'][:] # H&E features
                            coords = hdf5_file['coords'][:].astype(int) # feats is a tuple of H&E and IMC features            
                            print(feats_he.shape, coords.shape)
                            wsi = openslide.open_slide(wsi_path)

                            # pass wsi and coords to data loader 
                            feats_translator = []
                            features = {} # placeholder for batch features

                            generator = extract_features(translator, resnet, feats_translator, features, 
                                                         device, wsi, coords, workers=workers, batch_size=batch_size)
                            
                            # looping through all batches
                            x = 0
                            for i, (coord, feats_trans, feats_res_tumor_immune, feats_res_tumor) in enumerate(generator):
                                # print(feats_trans.shape, feats_trans.shape[1])

                                if not args.dump_batches: # collect batches                
                                    if x == 0: 
                                        feats_trans_loop = np.empty((0, feats_trans.shape[1]))
                                        feats_res_tumor_immune_loop = np.empty((0, feats_res_tumor_immune.shape[1]))
                                        feats_res_tumor_loop = np.empty((0, feats_res_tumor.shape[1]))

                                    feats_trans_loop = np.vstack((feats_trans_loop, feats_trans))
                                    feats_res_tumor_immune_loop = np.vstack((feats_res_tumor_immune_loop, feats_res_tumor_immune))
                                    feats_res_tumor_loop = np.vstack((feats_res_tumor_loop, feats_res_tumor))
                                
                                else: # dump batches
                                    write_to_h5(wsi_group, {"features_imc_trans": feats_trans, 
                                                            "features_imc_resnet": feats_res_tumor_immune, 
                                                            "features_imc_resnet_tumor": feats_res_tumor
                                    })

                                x += len(coord)
                                # print('X: ', x, feats_trans_loop.shape)

                            if not args.dump_batches: # dump all collected batches
                                # -- writing he and coords in new h5 file -- 
                                write_to_h5(wsi_group, {"features_imc_trans": feats_trans_loop, 
                                                        "features_imc_resnet": feats_res_tumor_immune_loop, 
                                                        "features_imc_resnet_tumor": feats_res_tumor_loop
                                                        })
                            wsi_group.create_dataset('features_he', data=feats_he)
                            wsi_group.create_dataset('coords', data=coords)
                            print('Time to extract featurs: ', (time.time()- time_start))
