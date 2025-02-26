import numpy as np 
import pandas as pd
import glob 
import os
import random 

from src.dataset.dataset import make_dataset, is_image_file

# this script is used to create patches 
# original HE rois is 4000x4000x3 and IMC is 1000x1000x11 -- we extract 220 patches from each roi,
# each patch has HE of shape 1024x1024x3 and IMC of 256x256x11
# these are then fed to cyclegan for training where HE images are downsampled to 256x256x3 in the dataloader

split = '/raid/sonali/project_mvs/meta/tupro/split3_train-test.csv'
src_folder = '/raid/sonali/project_mvs/data/tupro/binary_he_rois'
tgt_folder = '/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x'
save_path = '/raid/sonali/project_mvs/data/tupro/patches'

save_path_he = os.path.join(save_path, 'binary_he_patchs')
save_path_imc = os.path.join(save_path, 'binary_imc_processed_11x')

os.makedirs(save_path_he, exist_ok=True)
os.makedirs(save_path_imc, exist_ok=True)

mode = 'train'
src_paths = sorted(make_dataset(src_folder, mode, split))
tgt_paths = sorted(make_dataset(tgt_folder, mode, split))

print(src_paths[0:2], len(src_paths))
print(tgt_paths[0:2], len(tgt_paths))

assert len(src_paths) == len(tgt_paths), "Source and target data folders should contains the same number of images."
for s, t in zip(src_paths, tgt_paths):
    if os.path.basename(s) != os.path.basename(t):
        raise ValueError(f"File names do not match: {s} and {t}")
print(f'Number of images in the {mode} dataset: {len(src_paths)}')

patch_size = 256
random.seed(0)
n_patches_roi = 220

for idx in range(len(src_paths)):
    sample = os.path.basename(src_paths[idx]).split('.')[0]
    he_roi = np.load(src_paths[idx], mmap_mode='r')
    imc_roi = np.load(tgt_paths[idx], mmap_mode='r')
    
    for j in range(n_patches_roi):    
    
        augment_x_offset = random.randint(0, 1000 - patch_size)
        augment_y_offset = random.randint(0, 1000 - patch_size)

        imc_patch = imc_roi[augment_y_offset: augment_y_offset + patch_size,
                        augment_x_offset: augment_x_offset + patch_size, :]

        factor = int(he_roi.shape[1] / imc_roi.shape[1]) # assume height == width

        he_patch = he_roi[factor * augment_y_offset: factor * augment_y_offset + factor * patch_size,
                            factor * augment_x_offset: factor * augment_x_offset + factor * patch_size, :]
            
        
        save_name = sample + '_' + str(augment_x_offset) + '_' + str(augment_y_offset) + '.npy'
        print(save_name)
        
        np.save(os.path.join(save_path_he, save_name), he_patch)
        np.save(os.path.join(save_path_imc, save_name), imc_patch)                
        
print('Patching done!')

# QC -- checking if the filenames and number of patches correspond in src and tgt folders
src_patches_paths = sorted(glob.glob(save_path_he + '/*npy'))
tgt_patches_paths = sorted(glob.glob(save_path_imc + '/*npy'))

print(src_patches_paths[0:2], len(src_patches_paths))
print(tgt_patches_paths[0:2], len(tgt_patches_paths))

assert len(src_patches_paths) == len(tgt_patches_paths), "Source and target data folders should contains the same number of images."
for s, t in zip(src_patches_paths, tgt_patches_paths):
    if os.path.basename(s) != os.path.basename(t):
        raise ValueError(f"File names do not match: {s} and {t}")
print(f'Number of images in the {mode} dataset: {len(src_patches_paths)}')