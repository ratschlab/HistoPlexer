import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import os

import torch
import torchvision

from codebase.utils.constants import *
from codebase.utils.inference_utils import *
from codebase.utils.eval_utils import get_last_epoch_number, get_protein_list
from codebase.experiments.cgan3.training_helpers import str2bool
from codebase.experiments.cgan3.network import * #Translator, Discriminator

parser = argparse.ArgumentParser(description='Inference on a selected set.')
parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
parser.add_argument('--set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
parser.add_argument('--submission_id', type=str, required=True, default=None, help='')
parser.add_argument('--pth_name', type=str, required=False, default=None, help='Which epoch to use (pth file name), if "which_model" is old then provide desired model_state pth')
parser.add_argument('--epoch', type=str, required=False, default='last', help='if last then takes the last one found, if best then searching for chkpt_selection info, or say "45K" or step size for which inference already done')
parser.add_argument('--level', type=int, required=False, default=2, help='Which resolution to use {2,4,6}')
parser.add_argument('--which_cluster', type=str, required=True, help='biomed or dgx:gpu')
parser.add_argument('--which_model', type=str, required=False, default='new', help='Old model or new model with fixed checkerboard effect')

args = parser.parse_args()
save_path = Path(args.project_path).joinpath('results')
model_path = Path(args.project_path).joinpath('results', args.submission_id, 'tb_logs')
# update paths based on project folder
CV_SPLIT_ROIS_PATH = Path(args.project_path).joinpath(CV_SPLIT_ROIS_PATH)
pth_name = args.pth_name
epoch = args.epoch
data_set = args.set

# define cv_split (based on args.txt)
job_args = json.load(open(Path(args.project_path).joinpath('results',args.submission_id,'args.txt')))
cv_split = job_args['cv_split']
model_depth = job_args['model_depth']
BINARY_HE_ROI_STORAGE = get_he_roi_storage(Path(args.project_path).joinpath('data/tupro'), job_args['which_HE'])

# setting device 
dev0 = 'cuda:0' if 'dgx' not in args.which_cluster else 'cuda:' + args.which_cluster.split(':')[1]
dev0 = torch.device(dev0 if torch.cuda.is_available() else 'cpu')
print('Device: ', dev0)

if pth_name is None:
    if epoch=='best':
        chkpt_path = Path(str(model_path).replace('tb_logs', 'chkpt_selection'))
        # to allow to use different best checkpoints for performing snaity checks
        best_data_set = epoch.split('_')[-1] if '_' in epoch else 'valid'
        chkpt_file = '-'.join(['best_epoch','level_'+str(args.level),best_data_set])+'.txt'
        if os.path.exists(chkpt_path):
            pth_name = json.load(open(chkpt_path.joinpath(chkpt_file)))['chkpt_file']
            step_name = json.load(open(chkpt_path.joinpath(chkpt_file)))['best_step']
            print('Using best step: '+str(step_name))
        # else:
        #     print('Checkpoint selection data not found - using the last epoch found')
        #     epoch = 'last'
    elif epoch == 'last':
        print('Using the last step/epoch found.')
        step_name = get_last_epoch_number(model_path)
        pth_name = step_name + '_translator.pth'
    else: # eg when epoch='145K'
        step_name = str(epoch)
        pth_name = step_name + '_translator.pth'
else:
    step_name = str(pth_name.split('_')[0]) 

print('Analysing', args.submission_id, pth_name, step_name, data_set)    

if args.which_model=="old": # load from model_state 
    protein_subset = get_protein_list(job_args['protein_set'])  
    channel_list = [protein2index[prot_name] for prot_name in protein_subset]
    
    network = unet_translator(len(channel_list), depth=model_depth, flag_asymmetric=True, flag_multiscale=True, last_activation='relu',which_decoder='convT', encoder_padding=1, decoder_padding=1)    
    checkpoint = torch.load(model_path.joinpath(pth_name), map_location=torch.device(dev0))    
    network.load_state_dict(checkpoint['trans_state_dict'])
else: 
    network = torch.load(model_path.joinpath(pth_name), map_location=torch.device(dev0))
    
network.to(dev0)
network.eval()

# ----- get sample_roi names for a given CV split -----
cv = json.load(open(CV_SPLIT_ROIS_PATH))
sample_rois = cv[cv_split][data_set]


# ----- iterate through all sample_rois for the selected set ----- 
for s_roi in sample_rois:
    print(s_roi)
    he_roi_np_lvl0 = np.load(BINARY_HE_ROI_STORAGE.joinpath(s_roi + ".npy"), mmap_mode='r')
    he_roi_tensor_lvl0 = get_tensor_from_numpy(he_roi_np_lvl0)    
    
    # get imc desired shapes based on input HE image
    imc_desired_shapes = get_target_shapes(model_depth, he_roi_np_lvl0.shape[0])
    # pad image to make compatible with model (eg /2)
    he_roi_tensor_lvl0 = pad_img(he_roi_tensor_lvl0, he_roi_np_lvl0.shape[0]).to(dev0)

    # predict IMC 
    with torch.no_grad():
        pred_imc_roi_tensor = network(he_roi_tensor_lvl0)
    
    # save all IMC levels
    save_path_images = save_path.joinpath(args.submission_id, data_set+"_images", "step_"+step_name)
    for i in range(len(pred_imc_roi_tensor)): 
        
        save_path_level = save_path_images.joinpath('level_' + str((i+1)*2))
        
        if not os.path.exists(save_path_level):
            save_path_level.mkdir(parents=True, exist_ok=False)

        # crop image to desired shape and save
        pred_imc = torchvision.transforms.CenterCrop([imc_desired_shapes[i], imc_desired_shapes[i]])(pred_imc_roi_tensor[-(i+1)])
        pred_imc_roi_np = pred_imc[0].detach().cpu().numpy().transpose((1, 2, 0))        
        np.save(save_path_level.joinpath(s_roi + ".npy"), pred_imc_roi_np)