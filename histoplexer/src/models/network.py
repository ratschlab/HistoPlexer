import numpy as np
import random
from typing import Any
import kornia

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init
import torchvision.transforms.functional as ttf
from torch.nn import Conv2d, ConvTranspose2d
from torch import Tensor
from histoplexer.utils.constants import *


def get_optimizer(network, network_type, lr_scheduler_type='fixed', lazy_c=None):
    ''' Get optimizer for network
    network: network object
    network_type: type of the network {discriminator, translator}
    lr_scheduler_type: type of the learning rate scheduler {fixed, plateau, cosine, step}
    lazy_c: lazy R1 regularization: don't compute for every mini-batch
    '''
    if lr_scheduler_type == 'fixed':
        lr_val = 0.0008 if network_type=='discriminator' else 0.004
    elif lr_scheduler_type == 'fixedsmall':
        lr_val = 0.00008 if network_type=='discriminator' else 0.0004
    elif lr_scheduler_type == 'fixedqeual':
        lr_val = 2.5e-4
    else:
        # settings from pix2pix: 0.0002
        lr_val = 0.002
        
    if lazy_c:
        optimizer = optim.Adam(network.parameters(), lr=lr_val*lazy_c, betas=(0.5**lazy_c, 0.999**lazy_c))
    else:
        optimizer = optim.Adam(network.parameters(), lr=lr_val, betas=(0.5, 0.999))

    return optimizer

def get_lr_scheduler(optimizer, lr_scheduler_type='fixed', **kwargs):
    ''' Get learning rate schedulers
    optimizer: optimizer object
    lr_scheduler_type: type of the learning rate scheduler {fixed, fixedsmall, plateau, cosine, step}
    '''
    if lr_scheduler_type.startswith('fixed'):
        network_lr_scheduler = None
    else:
        if lr_scheduler_type == 'plateau':
            network_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, threshold=0.0001, patience=100)
        elif lr_scheduler_type == 'cosine':
            network_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
        elif lr_scheduler_type == 'step':
            network_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return network_lr_scheduler

        
def update_translator_bool(rule='prob', dis_fake_loss=None):
    ''' Get bool whether to update translator
    rule: rule for updating {always, prob, dis_loss}
    dis_fake_loss: fake_score_mean value from discriminator (used only if rule==dis_loss)
    '''
    if rule == 'always':
        if_update = True
    elif rule == 'prob':
        if_update = random.choice([True, True, False])
    elif rule == 'dis_loss':
        assert dis_fake_loss is not None, 'dis_fake_loss not provided!'
        # TODO: check if 0.5 threshold makes sense
        if_update = dis_fake_loss<0.5

    return if_update

# -----------------------------------------------------------------
# Helper classes for models 
# -----------------------------------------------------------------

class Gauss_Pyramid_Conv(nn.Module):
    """
    Code borrowed from: https://github.com/csjliang/LPTN
    """
    def __init__(self, num_high=3, num_blur=4, channels=11):
        super(Gauss_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.num_blur = num_blur
        self.channels = channels
        
    def downsample(self, x):
        return kornia.filters.blur_pool2d(x, kernel_size=3, stride=2)

    def conv_gauss(self, img):
        # Parameters for gaussian_blur2d: (input, kernel_size, sigma)
        return kornia.filters.gaussian_blur2d(img, (3, 3), (1, 1))
    
    def forward(self, img):
        current = img
        pyr = [current]
        for _ in range(self.num_high):
            # Applying gaussian blur 4 times
            for _ in range(self.num_blur):
                current = self.conv_gauss(current)
            # Downsample using blur_pool2d
            down = self.downsample(current)
            current = down
            pyr.append(current)
        return pyr
    
