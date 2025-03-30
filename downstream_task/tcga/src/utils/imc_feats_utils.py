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

# ------------------------------------------------------
# ----Functions for dataloader and inference ----
# ------------------------------------------------------

def collate_features(batch):
    # Item 2 is the boolean value from tile filtering.
    img = torch.cat([item[0] for item in batch])
    # img = np.vstack([item[0] for item in batch])

    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def crop_rect_from_slide(slide, rect, tile_size=1024):
    minx, miny, maxx, maxy = rect
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (tile_size, tile_size))

class BagOfTiles(Dataset):
    def __init__(self, wsi, coords):
        self.wsi = wsi
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = crop_rect_from_slide(self.wsi, coord)

        # Convert from RGBA to RGB
        img = img.convert('RGB')

        # Turn the PIL image into a (C x H x W) torch.FloatTensor (32 bit by default)
        # in the range [0.0, 1.0].
        img = transforms.functional.to_tensor(img).unsqueeze(0)

        return img, coord
    

def extract_features(translator, resnet, feats_translator, features, device, wsi, coords, workers=8, batch_size=128):
    # Use multiple workers if running on the GPU, otherwise we'll need all workers for
    # evaluating the model.
    kwargs = (
        {"num_workers": workers, "pin_memory": True, "pin_memory_device": device} 
    )

    loader = DataLoader(
        dataset=BagOfTiles(wsi, coords),
        batch_size=batch_size,
        collate_fn=collate_features,
        **kwargs,
    )
    gap = nn.AdaptiveAvgPool2d(1) 

    with torch.no_grad():
        for he_batch, coords_batch in loader:

            # fixing the problem of last batch 
            len_coord_original = len(coords_batch)
            if len_coord_original < batch_size: 
                print('SMALLER batch size found of : ', len_coord_original)
                coords_batch = np.tile(coords_batch, (batch_size // coords_batch.shape[0] + 1, 1))[:batch_size]

                num_repeats = batch_size // len_coord_original + 1
                # he_batch = torch.tensor(np.tile(he_batch, (num_repeats, 1, 1, 1))[:batch_size])
                he_batch = torch.tile(he_batch, (num_repeats, 1, 1, 1))[:batch_size]

                # he_batch = np.tile(he_batch, (batch_size // he_batch.shape[0] + 1, 1))[:batch_size]
                print('fixed to batch size : ', len(he_batch), he_batch.shape, coords_batch.shape)


            he_batch = he_batch.to(device, non_blocking=True)
            IMC_generated  = translator(he_batch)
            # print('IMC generated: ', IMC_generated[0].shape, IMC_generated[1].shape, IMC_generated[2].shape)

            translator_features = gap(features['feats_translator']).squeeze().cpu().numpy()

            # -- get IMC input for resnet in range 0 and 1 -- 
            # ["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100"]
            # CD20, CD3, MelanA: 1, 2, 8; "MelanA", "S100", "SOX10": 5,8,9

            # Calculate the min and max 
            min_per_channel = IMC_generated[2].amin((0, 2, 3))
            max_per_channel = IMC_generated[2].amax((0, 2, 3))  

            min_per_channel = min_per_channel.view(1, len(min_per_channel), 1, 1)
            max_per_channel = max_per_channel.view(1, len(max_per_channel), 1, 1)

            # Min-Max normalization
            IMC_generated_norm = (IMC_generated[2] - min_per_channel) / (max_per_channel - min_per_channel)
            
            resnet_input_tumor_immune = torch.index_select(IMC_generated_norm, 1, torch.tensor([1, 2, 8]).to(device))            
            resnet_features_tumor_immune = resnet(resnet_input_tumor_immune).squeeze().cpu().numpy()

            resnet_input_tumor = torch.index_select(IMC_generated_norm, 1, torch.tensor([5, 8, 9]).to(device))            
            resnet_features_tumor = resnet(resnet_input_tumor).squeeze().cpu().numpy()

            yield coords_batch[:len_coord_original], translator_features[:len_coord_original], resnet_features_tumor_immune[:len_coord_original], resnet_features_tumor[:len_coord_original]

def write_to_h5(file, asset_dict):
    for key, val in asset_dict.items():
        if key not in file:
            maxshape = (None,) + val.shape[1:]
            dset = file.create_dataset(
                key, shape=val.shape, maxshape=maxshape, dtype=val.dtype
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + val.shape[0], axis=0)
            dset[-val.shape[0] :] = val
            
            
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')