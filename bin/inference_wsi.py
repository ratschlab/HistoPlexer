import os 
import cv2
import numpy as np 
import pandas as pd
import openslide 
import glob
import argparse
import time 
import tqdm
import tifffile
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nnF
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

import torchstain
from types import SimpleNamespace
import json
from shapely.geometry import Polygon, box
from shapely.affinity import scale
from shapely.ops import unary_union

from src.inference.histoplexer_inference_wsi import HistoplexerInferenceWSI

parser = argparse.ArgumentParser(description='HistoPlexer prediction on whole slide images')
parser.add_argument("--checkpoint_path", type=str, required=False, default=None, help="Path to checkpoint file")
parser.add_argument('--wsi_paths', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/HE_new_wsi', help='path where wsi h&e images reside')
parser.add_argument('--sample', type=str, required=False, default=None, help='sample name for which we need to run wsi prediction')
parser.add_argument("--device", type=str, required=False, default='cuda:0', help="device to use")
parser.add_argument('--data_set', type=str, required=False, default="test", help='Which data_set to use {test, external_test, HE_ultivue}')
parser.add_argument('--ref_img_path', type=str, required=False, default=None, help='Path to reference image for stain normalization')
parser.add_argument('--chunk_size', type=int, required=False, default=1024, help='the tile size used for inference')
parser.add_argument('--batch_size', type=int, required=False, default=128, help='the batch size used for inference')
parser.add_argument('--save_path', type=str, required=False, default=None, help='the path used for saving')

# ----- paths etc -----
args = parser.parse_args()
data_set = args.data_set
wsi_paths = args.wsi_paths

if args.save_path is None:
    save_path = Path(os.path.dirname(args.checkpoint_path))  
else:
    os.makedirs(args.save_path, exist_ok=True)
    save_path = Path(args.save_path)
print('save_path: ', save_path) 

# ----- getting config for experiment -----
config_path = os.path.dirname(args.checkpoint_path) + '/config.json'        
with open(config_path, "r") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
protein_subset = config.markers
print('protein_subset: ', len(protein_subset), protein_subset)

# ----- defining specifics for inference -----
chunk_size = args.chunk_size 
chunk_padding = 0
batch_size = args.batch_size
loader_kwargs = {'num_workers': 8, 'pin_memory': True}

# ----- prepare reference img for stain normalization -----
normalizer = None
if args.ref_img_path:
    print("Initilize MacenkoNormalizer...")
    ref_img = Image.open(args.ref_img_path)
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(tsfm(ref_img))
    
# ----- Initialize HistoplexerInferenceWSI -----
histoplexer_wsi = HistoplexerInferenceWSI(
    checkpoint_path=args.checkpoint_path,
    seg_level=6,  # Default segmentation level
    chunk_size=chunk_size,
    batch_size=batch_size,
    n_proteins=len(protein_subset),
    loader_kwargs=loader_kwargs,
    device=args.device,
    normalizer=normalizer
)

# ----- get sample_roi names for a cv split from experiment config -----
splits = pd.read_csv(config.split)
if ("external_test" in data_set) or ('HE_ultivue' in data_set): 
    samples = np.array(list(set([i.split('.')[0] for i in os.listdir(wsi_paths)])))
elif "all" in data_set: # this includes cases that are tupro samples, even ones that are not in the cv split
    samples = np.array(list(set([i.split('.')[0].split('-')[0].split('_')[0] for i in os.listdir(wsi_paths)]))) 
    print(len(samples))    
else:
    sample_rois = list(set(splits[data_set].dropna()))   
    # getting sample names from sample_rois
    samples = np.array(list(set([i.split('_', 1)[0] for i in sample_rois]))) 

if args.sample is None: 
    print('samples: ', samples)
    save_path_imgs = save_path.joinpath(data_set + "_wsis")
    for sample in samples: 
        print('sample: ', sample)
        try: 
            wsi_path = glob.glob(wsi_paths + '/' + sample + '*')[0]
            wsi = openslide.open_slide(wsi_path)
        except:
            print('no WSI image found for sample ', sample)
            continue
        
        print('save file exists: ', os.path.isfile(os.path.join(save_path_imgs, 'level_2', sample + '.npy')))
        if not os.path.isfile(os.path.join(save_path_imgs, 'level_2', sample + '.npy')): 
            start_time = time.time()
            histoplexer_wsi.get_wsi_inference(sample, wsi, save_path_imgs)

            # timing
            end_time = time.time()
            print('time for wsi: ', end_time-start_time)
            hours, rem = divmod(end_time-start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        else: 
            print('Inference already done for: ', sample)

else: 
    # predict for only chosen sample
    sample = args.sample     
    # find this sample is in which dataset     
    if ('all' not in data_set) and ('HE_ultivue' not in data_set) and  ('external_test' not in data_set): 
        sample_rois_test = np.array(list(set(splits['test'].dropna())))   
        samples_test = np.array(list(set([i.split('_', 1)[0] for i in sample_rois_test]))) 
        samples_test = np.append(samples_test, sample_rois_test)
        
        sample_rois_train = np.array(list(set(splits['train'].dropna()))) 
        samples_train = np.array(list(set([i.split('_', 1)[0] for i in sample_rois_train]))) 
        samples_train = np.append(samples_train, sample_rois_train)
    
        if sample in samples_test:             
            data_set = 'test'
        elif sample in samples_train: 
            data_set = 'train'
    save_path_imgs = save_path.joinpath(data_set + "_wsis")
    
    # check if inference is already done for this sample
    if not os.path.isfile(os.path.join(save_path_imgs, 'level_2', sample + '.npy')):
        if (len(glob.glob(wsi_paths + '/' + sample + '*')) != 0):
            wsi_path = glob.glob(wsi_paths + '/' + sample + '*')[0]
            wsi = openslide.open_slide(wsi_path)

            start_time = time.time()
            histoplexer_wsi.get_wsi_inference(sample, wsi, save_path_imgs)

            # timing
            end_time = time.time()
            print('time for wsi: ', end_time-start_time)
            hours, rem = divmod(end_time-start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        else: 
            print('no WSI image found for sample ', args.sample)
    else: 
        print('Inference already done for: ', sample)
