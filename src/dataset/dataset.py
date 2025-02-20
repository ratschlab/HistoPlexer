from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy
import torch
import random
import os
import json
import math
import torchvision


from src.utils.data.transforms import HE_transforms, shared_transforms

IMG_EXTENSIONS = [
    '.npy', '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, mode, split_csv, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    
    splits = pd.read_csv(split_csv)
    allowed_filenames = set(splits[mode].dropna())

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and os.path.splitext(fname)[0] in allowed_filenames:
                path = os.path.join(root, fname)
                images.append(path)

    # Return the filtered list of image paths
    return images[:min(max_dataset_size, len(images))]


class BaseDataset(Dataset):
    """
    A base class for datasets used in the project.

    This class initializes common attributes like data paths and provides a basic structure for datasets.

    Attributes:
        data_path (str): Base path for the data.
        split (str): Identifier for the data split.
    """

    def __init__(self,  
                split: str, 
                mode: str, 
                src_folder: str, 
                tgt_folder: str):
        """
        Initialize the BaseDataset class.

        Args:
            split (str): Identifier for the data split.
            mode (str): Identifier for the mode (train/val/test).
            src_folder (str): Path to the source data. 
            tgt_folder (str): Path to the target data.
        """
        super().__init__()
        self.split = split
        assert mode in ['train', 'valid', 'test'], "Mode can only be train, valid or test."
        self.mode = mode
        print(f"Load {self.mode} source data from: {src_folder}...")
        self.src_paths = sorted(make_dataset(src_folder, self.mode, self.split))
        print(f"Load {self.mode} target data from: {tgt_folder}...")
        self.tgt_paths = sorted(make_dataset(tgt_folder, self.mode, self.split))
        
        assert len(self.src_paths) == len(self.tgt_paths), "Source and target data folders should contains the same number of images."
        for s, t in zip(self.src_paths, self.tgt_paths):
            if os.path.basename(s) != os.path.basename(t):
                raise ValueError(f"File names do not match: {s} and {t}")
        print(f'Number of images in the {mode} dataset: {len(self.src_paths)}')

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Subclasses should override this method to return the actual size.

        Returns:
            int: The number of items.
        """
        pass

    def __getitem__(self, idx):
        """
        Retrieve an item by its index.

        Subclasses should override this method to provide the actual data retrieval mechanism.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the data of the item.
        """
        pass
    
    
class TuProDataset(BaseDataset):
    def __init__(self, 
                 split: str, 
                 mode: str, 
                 src_folder: str, 
                 tgt_folder: str,
                 use_high_res: bool = True,
                 p_flip_jitter_hed_affine: list = [0.5, 0.0, 0.5, 0.5],
                 patch_size: int = 256,
                 channels: list=None):
        """
        Initialize the TuProDataset class.

        This class is for creating a dataset for the TuPro project, which includes
        transformations and patch extraction for training a cGAN.

        Args:
            use_high_res (bool): Flag to use high resolution for H&E images. Defaults to True.
            p_flip_jitter_hed_affine (list): Probabilities for flip, jitter, HED, and affine transformations.
            patch_size (int): Size of the patches to be extracted. Defaults to 256.
            channels (list): Selected channels. Default to None (loading all channels). Required for single-plex experiment setting.
        """
        super().__init__(split, mode, src_folder, tgt_folder)
        self.use_high_res = use_high_res
        self.he_transforms = HE_transforms
        self.shared_transforms = shared_transforms
        self.p_shared = p_flip_jitter_hed_affine[0]
        self.p_jitter = p_flip_jitter_hed_affine[1]
        self.p_hed = p_flip_jitter_hed_affine[2]
        self.p_affine = p_flip_jitter_hed_affine[3]
        self.patch_size = patch_size
        self.channels = channels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of image pairs in the dataset.
        """
        return len(self.src_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves an image pair (H&E and IMC) from the dataset by index.

        Args:
            idx (int): Index of the image pair in the dataset.

        Returns:
            dict: A dictionary containing the H&E patch, IMC patch, sample name, and offsets.
        """
        sample = os.path.basename(self.src_paths[idx]).split('.')[0]
        he_roi = np.load(self.src_paths[idx], mmap_mode='r')
        imc_roi = np.load(self.tgt_paths[idx], mmap_mode='r')
        
        if self.channels:
            imc_roi = imc_roi[:, :, self.channels]
        
        augment_x_offset = random.randint(0, 1000 - self.patch_size)
        augment_y_offset = random.randint(0, 1000 - self.patch_size)
        
        imc_patch = imc_roi[augment_y_offset: augment_y_offset + self.patch_size,
                        augment_x_offset: augment_x_offset + self.patch_size, :]
        
        factor = int(he_roi.shape[1] / imc_roi.shape[1]) # assume height == width
        if not self.use_high_res:
            he_roi = scipy.ndimage.zoom(he_roi, (1./factor, 1./factor, 1), order=1) # using bilinear interpolation for faster computation 
            he_patch = he_roi[augment_y_offset: augment_y_offset + self.patch_size,
                            augment_x_offset: augment_x_offset + self.patch_size, :]
        else:
            he_patch = he_roi[factor * augment_y_offset: factor * augment_y_offset + factor * self.patch_size,
                            factor * augment_x_offset: factor * augment_x_offset + factor * self.patch_size, :]
            
        he_patch = he_patch.transpose((2, 0, 1)) # [H, W, C] --> [C, H, W]
        imc_patch = imc_patch.transpose((2, 0, 1)) # [H, W, C] --> [C, H, W]
        
        he_patch = torch.from_numpy(he_patch.astype(np.float32)) # changed, removed copy false
        imc_patch = torch.from_numpy(imc_patch.astype(np.float32))
        
        he_patch, imc_patch = self.shared_transforms(he_patch, imc_patch, p=self.p_shared)
        he_patch = self.he_transforms(he_patch, p=[self.p_jitter, self.p_hed, self.p_affine])
        
        if not he_patch.shape[0]==3: 
            he_patch = torch.from_numpy(he_patch.transpose((2, 0, 1)))
        
        return {'he_patch': he_patch.to(torch.float), 
                'imc_patch': imc_patch.to(torch.float),
                'sample': sample,
                'x_offset': augment_x_offset, 
                'y_offset': augment_y_offset, 
                'he_path': self.src_paths[idx], 
                'imc_path': self.tgt_paths[idx], 
                'idx': idx
               } 
                 

class InferenceDataset(Dataset):
    def __init__(self, input_paths):
        """
        Args:
            file_paths (list): List of paths to input .npy files.
            output_paths (list): List of paths to save the output of the model.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.input_paths = input_paths

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        img_name = self.input_paths[idx].split('/')[-1].split('.')[0]
        input_img = np.load(self.input_paths[idx], mmap_mode='r') # what if diff file format? 
        input_shape = int(input_img.shape[0]) # used for cropping later
        input_img = self.transform_np_to_tensor(input_img)
        input_img = self.pad_img(input_img, input_shape)
        return input_img, img_name, input_shape

    @staticmethod
    def transform_np_to_tensor(np_img):
        ''' Construct torch tensor from a numpy array
        np_img: numpy array of shape [H,W,C]
        returns a torch tensor with shape [C,H,W]
        '''
        torch_img = np_img.transpose((2, 0, 1))
        torch_img = np.ascontiguousarray(torch_img)
        torch_img = torch.from_numpy(torch_img).float()
        return torch_img
    
    @staticmethod
    def pad_img(torch_img, input_shape=4000):
        ''' Pad image to make it compatible wth the model (eg /2)
        torch_img: torch tensor (expects a squared image)
        input_shape: shape ([0]) of the input H&E image
        returns a padded image (torch tensor)
        '''
        padding = (2**(round(math.log(input_shape, 2))) - input_shape)//2
        torch_img = torchvision.transforms.Pad(padding, padding_mode='reflect')(torch_img)
        return torch_img