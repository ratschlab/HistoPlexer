import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import scipy
import scipy.ndimage
import random
import torch
from torchvision import transforms

def shared_transforms(img1, img2, p=0.5):
    """
    Apply simultaneous transformations to H&E and IMC data.

    This function applies random horizontal or vertical flips and random
    rotations at multiples of 90 degrees to both images.

    Args:
        img1: H&E ROI (expected).
        img2: IMC ROI (expected).
        p: Probability of applying each transformation (default is 0.5).

    Returns:
        A tuple of transformed images (img1, img2).
    """
    # Random horizontal flipping
    if random.random() < p:
        img1 = transforms.functional.hflip(img1)
        img2 = transforms.functional.hflip(img2)

    # Random vertical flipping
    if random.random() < p:
        img1 = transforms.functional.vflip(img1)
        img2 = transforms.functional.vflip(img2)
        
    # Random 90 degree rotation
    if random.random() < p:
        angle = random.choice([90, 180, 270])
        img1 = transforms.functional.rotate(img1, angle)
        img2 = transforms.functional.rotate(img2, angle)

    return img1, img2

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_AB = opt.dataroot
        self.dir_A = os.path.join(self.dir_AB, 'binary_he_patchs')
        self.dir_B = os.path.join(self.dir_AB, 'binary_imc_processed_11x')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert len(self.A_paths) == len(self.B_paths), "The number of H&E and IMC images must match."
        self.patch_size = opt.crop_size
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        # self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.input_nc = 3
        self.output_nc = opt.output_nc
        self.channel = opt.channel
        
        self.shared_transforms = shared_transforms
        
    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        sample = os.path.basename(self.A_paths[idx]).split('.')[0]
        
        he_patch = np.load(self.A_paths[idx], mmap_mode='r')
        imc_patch = np.load(self.B_paths[idx], mmap_mode='r')

        if self.channel != None:
            imc_patch = np.expand_dims(imc_patch[:, :, self.channel], axis = 2)
       
        factor = 4
        he_patch = scipy.ndimage.zoom(he_patch, (1. / factor, 1. / factor, 1), order=1)
        
        he_patch = he_patch.transpose((2, 0, 1))
        imc_patch = imc_patch.transpose((2, 0, 1))
        
        he_patch = torch.from_numpy(he_patch.astype(np.float32))
        imc_patch = torch.from_numpy(imc_patch.astype(np.float32))
        
        he_patch, imc_patch = self.shared_transforms(he_patch, imc_patch)
        
        if he_patch.shape[0] != 3:
            he_patch = torch.from_numpy(he_patch.transpose((2, 0, 1)))
        return {'A': he_patch, 'B': imc_patch, 
                'A_paths': self.A_paths[idx], 'B_paths': self.B_paths[idx]}
        
        # # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # A = A_transform(A)
        # B = B_transform(B)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
    
