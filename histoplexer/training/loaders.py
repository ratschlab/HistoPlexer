import numpy as np
import pandas as pd
import os
import torch
import random
import torch.nn as nn

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random

from histoplexer.utils.constants import *
from histoplexer.utils.raw_utils import *
from histoplexer.utils.HEtransform_utils import *


# -----------------------------------
# IMC TRANSFORMS 
# -----------------------------------
def IMC_transforms(standardize_data, minmax_data, cohort_stats_file, channel_list):
    '''
    The function returns the nn.sequential necessary for normalisation of IMC data
    standardize_data: True/False based on if need to standardise IMC data  
    minmax_data: True/False based on if need to apply minmax to IMC data
    cohort_stats_file: path where the stats of the split reside 
    channel_list: the desired markers in the multiplex   
    '''
    if standardize_data or minmax_data: 
        cohort_stats = pd.read_csv(cohort_stats_file, sep='\t', index_col=[0])

    if standardize_data:
        # load cohort stats based on imc preprocessing steps (naming convention)
        mean_mat = cohort_stats['mean_cohort'][channel_list]
        std_mat = cohort_stats['std_cohort'][channel_list]

    if minmax_data:
        min_col = 'min_stand_cohort' if standardize_data else 'min_cohort'
        max_col = 'max_stand_cohort' if minmax_data else 'max_cohort'
        min_mat = cohort_stats[min_col][channel_list]
        max_mat = cohort_stats[max_col][channel_list]

    def default(val, def_val):
        return def_val if val is None else val

    imc_transforms = []
    if standardize_data: 
        imc_transforms.append(
        T.Normalize(mean_mat, std_mat)
        )

    if minmax_data: 
        imc_transforms.append(
        T.Normalize(min_mat, (max_mat-min_mat))
        )
    imc_transform_default = nn.Sequential(*imc_transforms)
    imc_transforms = default(None, imc_transform_default)  
    return imc_transforms

# -----------------------------------
# shared transformations b/w HE and IM 
# -----------------------------------  
def shared_transforms(img1, img2, p=0.5):
    '''
    The function contains the possible transformation that could be applied simultaneously to H&E and IMC data
    eg: random horizontal or vertical flip, random rotation for angles multiple of 90 degress
    img1: H&E ROI (expected)
    img2: IMC ROI (expected)
    p: the probability with which the transformation should be applied 
    '''
    # Random horizontal flipping
    if random.random() < p:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)

    # Random vertical flipping
    if random.random() < p:
        img1 = TF.vflip(img1)
        img2 = TF.vflip(img2)
        
    # Random 90 degree rotation
    if random.random() < p:
        angle = random.choice([90, 180, 270])
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle) 
    return img1, img2

# -----------------------------------
# HE TRANSFORMS 
# -----------------------------------
# sources: https://github.com/gatsby2016/Augmentation-PyTorch-Transforms

def HE_transforms(img, p=[0.0, 0.5, 0.5]):
    '''
    The function contains the possible transformation that could be applied to H&E ROIs
    This includes 
    eg: random horizontal or vertical flip, random rotation for angles multiple of 90 degress
    img1: H&E ROI (expected)
    img2: IMC ROI (expected)
    p: the probability with which the transformation should be applied 
    '''
    # Random color jitter
    if random.random() < p[0]:
        jitter = T.ColorJitter(brightness=.15, hue=.05, saturation=0.15)
        img = jitter(img)

    # Random HED jitter 
    if random.random() < p[1]:
        img = torch.permute(img, (1, 2, 0)) # channel first to last    
        hedjitter = HEDJitter(theta=0.01) # from HEtransform_utils
        img = hedjitter(img) 

    # Random affine transform
    if random.random() < p[2]:
        if not img.shape[2]==3: 
            img = torch.permute(img, (1, 2, 0)) # channel first to last    
        randomaffine = RandomAffineCV2(alpha=0.02) # from HEtransform_utils
        img = randomaffine(img) 
    return img

# -----------------------------------
# Dataloader for training
# ----------------------------------- 
class CGANDataset(Dataset):
    def __init__(self, project_path, align_results: list, name: str, data_path: str, protein_subset=PROTEIN_LIST_MVS, patch_size=400, imc_prep_seq='raw', cv_split='split0', standardize_imc=True, scale01_imc=True, factor_len_dataloader=8.0, which_HE='new', p_flip_jitter_hed_affine=[0.5,0.0,0.5,0.5], use_roi_weights=False):
        super(CGANDataset, self).__init__()

        self.project_path = project_path
        self.align_results = align_results
        self.name = name
        self.data_path = data_path
        self.patch_size = patch_size
        self.channel_list = [protein2index[prot_name] for prot_name in protein_subset]
        self.cv_split = cv_split
        self.use_roi_weights = use_roi_weights
        
        self.HE_ROI_STORAGE = get_he_roi_storage(self.data_path, which_HE)      
        self.IMC_ROI_STORAGE = get_imc_roi_storage(self.data_path, imc_prep_seq, standardize_imc, scale01_imc, cv_split)
         
        # if need to std or minmax IMC data 
        standardize_data = standardize_imc and ('std' not in self.IMC_ROI_STORAGE) 
        minmax_data = scale01_imc and ('minmax' not in self.IMC_ROI_STORAGE) 
        cohort_stats_file = os.path.join(project_path, COHORT_STATS_PATH, cv_split, 'imc_rois_'+imc_prep_seq+'-agg_stats.tsv')

        self.imc_transforms = IMC_transforms(standardize_data, minmax_data, cohort_stats_file, self.channel_list)
        self.imc_transforms_val = IMC_transforms(standardize_data=True, minmax_data=True, cohort_stats_file=cohort_stats_file, channel_list=self.channel_list)
        self.shared_transforms = shared_transforms
        self.he_transforms = HE_transforms

        self.p_shared = p_flip_jitter_hed_affine[0]
        self.p_jitter = p_flip_jitter_hed_affine[1]
        self.p_hed = p_flip_jitter_hed_affine[2]
        self.p_affine = p_flip_jitter_hed_affine[3]

        assert len(self.align_results) > 0, "Dataset received empty list of alignment results !"
        print(self.name + " has ", len(self.align_results), " alignment results !")

        # an estimation of number of training samples 
        self.num_samples = int(len(self.align_results) * ((1000 // (self.patch_size // 2)) ** 2) * factor_len_dataloader)
        print(self.name + " has ", self.num_samples, " training samples !")
        
        # weight for each sample: if use_roi_weights false then all get weight of 1, else obtained from align_results
        sample_weights = [1] * len(self.align_results)
        if self.use_roi_weights==True:
            for i, sample_dict in enumerate(self.align_results): 
                if 'sparsity_weights' in sample_dict.keys(): 
                    weights_dict = sample_dict['sparsity_weights']
                    sparse_markers = list(set(list(weights_dict.keys())) & set(protein_subset))
                    sparse_weights = [weights_dict[x] if x in weights_dict.keys() else 0 for x in protein_subset]
                    if len(sparse_markers)!=0: 
                        sample_weights[i] = round(0.5 + 0.5*(sum(sparse_weights)/len(sparse_markers)), 4)
        self.sample_weights = sample_weights
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):  # CAUTION idx argument is ignored, dataset is purely random !
        
        # load from disk:
        ar = random.choices(self.align_results, self.sample_weights, k=1)[0]
        he_roi = np.load(os.path.join(self.HE_ROI_STORAGE, ar["sample"] + "_" + ar["ROI"] + ".npy"), mmap_mode='r')
        imc_roi = np.load(os.path.join(self.IMC_ROI_STORAGE, ar["sample"] + "_" + ar["ROI"] + ".npy"), mmap_mode='r')
        
        # only keep channels that we need:
        imc_roi = imc_roi[:, :, self.channel_list]
        
        augment_x_offset = random.randint(0, 1000 - self.patch_size)
        augment_y_offset = random.randint(0, 1000 - self.patch_size)

        he_patch = he_roi[4 * augment_y_offset: 4 * augment_y_offset + 4 * self.patch_size,
                          4 * augment_x_offset: 4 * augment_x_offset + 4 * self.patch_size, :]

        imc_patch = imc_roi[augment_y_offset: augment_y_offset + self.patch_size,
                            augment_x_offset: augment_x_offset + self.patch_size, :]   

        he_patch = he_patch.transpose((2, 0, 1)) 
        imc_patch = imc_patch.transpose((2, 0, 1)) 
        he_patch = torch.from_numpy(he_patch.astype(np.float32, copy=False))
        imc_patch = torch.from_numpy(imc_patch.astype(np.float32, copy=False))
        
        he_patch, imc_patch = self.shared_transforms(he_patch, imc_patch, p=self.p_shared)
        imc_patch =  self.imc_transforms(imc_patch)
        he_patch = self.he_transforms(he_patch, p=[self.p_jitter, self.p_hed, self.p_affine])

        if not he_patch.shape[0]==3: 
            he_patch = torch.from_numpy(he_patch.transpose((2, 0, 1)))

        return {'he_patch': he_patch.to(torch.float), 
                'imc_patch': imc_patch.to(torch.float),
                'sample': ar['sample'], 'roi': ar['ROI'], 'x_offset': augment_x_offset, 'y_offset': augment_y_offset
               }    

# -----------------------------------
# loaders at inference time
# ----------------------------------- 
class Eval_dataset(Dataset):
    def __init__(self, imc_pred_paths, imc_roi_storage, channel_list, predicted_channel, data_range=(0, 1), convert_to_rgb=False):

        super(Eval_dataset, self).__init__()
        self.imc_roi_storage = imc_roi_storage
        self.imc_pred_paths = imc_pred_paths
        self.channel_list = channel_list
        self.predicted_channel = predicted_channel
        self.data_range = data_range
        self.num_channels = len(channel_list)
        self.convert_to_rgb = convert_to_rgb
        self.transform = T.Compose([
            T.Lambda(lambda x: self.channel_normalize(x))
        ])
        
    def __len__(self):
        return len(self.imc_pred_paths)
    
    def __getitem__(self, idx):  

        imc_pred_path = self.imc_pred_paths[idx]
        sample_roi = imc_pred_path.split('/')[-1].split('.npy')[0]
        imc_gt_path = os.path.join(self.imc_roi_storage, sample_roi + '.npy')
        imc_pred = np.load(imc_pred_path)[:, :, self.predicted_channel]
        imc_gt = np.load(imc_gt_path)[:, :, self.channel_list]

        # channel first 
        imc_pred = torch.from_numpy(imc_pred).permute(2, 0, 1)
        imc_gt = torch.from_numpy(imc_gt).permute(2, 0, 1)

        # normalise per channel
        imc_pred = self.transform(imc_pred)
        imc_gt = self.transform(imc_gt)

        # channel last 
        if self.convert_to_rgb:
            imc_pred = self.to_rgb(imc_pred)
            imc_gt = self.to_rgb(imc_gt)

        return {'sample_roi': sample_roi, 
                'imc_gt_path': os.path.join(self.imc_roi_storage, sample_roi + '.npy'), 
                'imc_pred_path': imc_pred_path, 
                'imc_pred': imc_pred, 
                'imc_gt': imc_gt}
    
    def channel_normalize(self, x):
        # Normalize each channel individually to the range [0, 1]
        for i in range(self.num_channels):
            min_value = torch.min(x[i])
            max_value = torch.max(x[i])
            denominator = max_value - min_value
            denominator[denominator == 0] = 1
            x[i] = (x[i] - min_value) / denominator
        return x
    
    def to_rgb(self, x):
        x = x.unsqueeze(1) 
        x = x.repeat(1, 3, 1, 1) 
        return x
