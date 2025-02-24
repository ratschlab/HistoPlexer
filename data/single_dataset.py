from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import scipy
import scipy.ndimage
import torch
import random
import os
import torchvision


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.patch_size = opt.crop_size
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = 3

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        
        he_patch = np.load(self.A_paths[idx], mmap_mode='r')
        factor = 4
        he_patch = scipy.ndimage.zoom(he_patch, (1. / factor, 1. / factor, 1), order=1)
        he_patch = he_patch.transpose((2, 0, 1))
        he_patch = torch.from_numpy(he_patch.astype(np.float32))
        he_patch = torchvision.transforms.Pad(padding=12, padding_mode='reflect')(he_patch) # padding of total 24
        
        print(he_patch.shape, self.A_paths[idx])
            
        return {'A': he_patch, 'A_paths': self.A_paths[idx]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
