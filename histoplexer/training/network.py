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

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights. 
    From: https://gitlab.com/eburling/SHIFT/-/blob/master/models/networks.py

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True
        
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
    
class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, n_chan=1):
        super().__init__()
        self.group_size = group_size
        self.n_chan = n_chan

    def forward(self, x):
        N, C, H, W = x.shape
        G = N
        if self.group_size is not None:
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
        F = self.n_chan
        c = C // F

        # split minibatch in n groups of size G, split channels in F groups of size c
        y = x.reshape(G, -1, F, c, H, W)
        # shift center (per group) to zero
        y = y - y.mean(dim=0)
        # variance per group
        y = y.square().mean(dim=0)
        # stddev
        y = (y + 1e-8).sqrt()
        # average over channels and pixels
        y = y.mean(dim=[2, 3, 4])
        # reshape and tile
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        # add to input as 'handcrafted feature channels'
        x = torch.cat([x, y], dim=1)
        return x

class EQConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                groups=1, bias=True, padding_mode="zeros") -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                        groups, bias, padding_mode)
        # initialize weights from Normal
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # eq lr
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias,
                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                            groups=self.groups)
        
class EQConvTranspose2d(ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros") -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                        groups, bias, dilation, padding_mode)

        # init weights from Normal
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # eq lr
        fan_in = self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor, output_size: Any = None) -> Tensor:
        output_padding = self._output_padding(x, output_size, self.stride, self.padding,
                                            self.kernel_size)
        return torch.conv_transpose2d(input=x, weight=self.weight*self.scale, bias=self.bias,
                                    stride=self.stride, padding=self.padding,
                                    output_padding=output_padding, groups=self.groups,
                                    dilation=self.dilation)
        

class BNRelu(nn.Module):
    """Basic block for batch norm and relu 
    """

    def __init__(self, num_features):
        super(BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.rl = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, inputs):
        return self.rl(self.bn(inputs))

# the kernel size, padding etc adapted from pytorch unet model: https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/    
class BNDownBlock(nn.Module):
    """Basic block for conv + (batch norm and relu) -- with downsampling using stride 2
    """
    def __init__(self, in_ch, out_ch, ksize=3, padding=1, batch_norm=True, eq_lr=False):
        super(BNDownBlock, self).__init__()
        self.batch_norm = batch_norm
        if eq_lr:
            self.conv1 = EQConv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=2, padding_mode='reflect')
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=2, padding_mode='reflect')
        
        if self.batch_norm: 
            self.bn1 = BNRelu(out_ch)
    
    def forward(self, x):
        if self.batch_norm == True: 
            return self.bn1(self.conv1(x))
        elif self.batch_norm == False: 
            return self.conv1(x)

class BNUpBlock(nn.Module):
    """Basic block for conv + (batch norm and relu) -- with upsamping
       module used for upsamping in the decoder part of unet  
       which_conv: 
           if "conv" then input x is first upsampled using interpolation mode can be "nearest" or can be changed to "bilinear" 
               and then conv operation is applied with stride and padding 1. This is used to overcome checkerboard effect
           if "convT" then ConvTranspose2d is applied with kernel and stride 2 which upsamples the input 

    """
    def __init__(self, in_ch, out_ch, which_conv='conv', padding=1, batch_norm=True, eq_lr=False):
        super(BNUpBlock, self).__init__()
        self.batch_norm = batch_norm
        self.which_conv = which_conv
                
        # choosing conv layer, w or w/o upsampling  
        if eq_lr:
            convolutions = nn.ModuleDict([
                        ['conv', EQConv2d(in_ch, out_ch, kernel_size=3, padding=padding, stride=1, padding_mode='reflect')],
                        ['convT', EQConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)] # stride 2 for upsampling
            ])
        else:
            convolutions = nn.ModuleDict([
                        ['conv', nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, stride=1, padding_mode='reflect')],
                        ['convT', nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)] # stride 2 for upsampling
            ])
        
        self.conv1 = convolutions[self.which_conv]
        if self.batch_norm:
            self.bn1 = BNRelu(out_ch)
    
    def forward(self, x):
        if self.which_conv == 'conv': 
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.batch_norm == True: 
            return self.bn1(self.conv1(x))
        elif self.batch_norm == False: 
            return self.conv1(x)

class output_block(nn.Module):
    """Basic block to get outputs from model 
    """
    def __init__(self, in_ch, out_ch, which_conv='conv', which_activation='relu', batch_norm=False, padding=1, eq_lr=False):
        
        """
        Parameters:
            in_ch (int) -- number of filters as input 
            out_ch (int) -- number of filters as output 
            which_conv (str) -- option to choose whether to do upsample+conv ("conv") or convTranspose ("convT") 
            which_activation (str) -- option to choose desired activation function for output layer 
        """

        super(output_block, self).__init__()
        self.which_conv = which_conv
        
        # choosing conv layer, w or w/o upsampling  
        if eq_lr:
            convolutions = nn.ModuleDict([
                        ['conv', EQConv2d(in_ch, out_ch, kernel_size=3, padding=padding, stride=1, padding_mode='reflect')],            
                        ['convT', EQConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)] # upsample 
            ])
        else:
            convolutions = nn.ModuleDict([
                        ['conv', nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, stride=1, padding_mode='reflect')],            
                        ['convT', nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)] # upsample 
            ])
        
        # chosing desired activation 
        activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['relu', nn.ReLU()], 
                    ['identity', nn.Identity()], 
                    ['sigmoid', nn.Sigmoid()], # (0,1)
                    ['tanh', nn.Tanh()] # (-1,1)
        ])
        
        self.conv1 = convolutions[which_conv]
        self.act1 = activations[which_activation]
        
    def forward(self, x):
        if self.which_conv == 'conv': 
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.act1(self.conv1(x))
    
class SNDownBlock(nn.Module):
    """Basic block for conv + leaky relu + spectral norm -- with downsampling
    """
        
    def __init__(self, in_ch, out_ch, ksize=3, padding=1, eq_lr=False):
        super(SNDownBlock, self).__init__()
        if eq_lr:
            self.conv1 = nn.utils.spectral_norm(EQConv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=2))
        else:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=2))
        self.rl = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        return self.rl(self.conv1(x))
    
# -----------------------------------------------------------------
# Translator model for Histoplexer
# -----------------------------------------------------------------
class unet_translator(nn.Module):
    def __init__(self, n_output_channels, last_activation='relu', flag_asymmetric=True, flag_multiscale=True, 
                 n_input_channels=3, n_filter=32, depth=6, which_decoder='conv', encoder_padding=1, decoder_padding=1, eq_lr=False):
        """
        Parameters:
            n_output_channels (int) -- desired number of channels in output, here for IMC 
            flag_asymmetric (bool) -- if input higher resolution than output 
            flag_multiscale (bool) -- if multiscale setting, want outputs from intermediate layers  
            n_input_channels (int) -- number of channels in input, here for H&E 
            n_filter (int) -- desired number of filters in the first symmetric block of model 
            depth (int) -- depth of model, including center block, excluding block for asymmetric setting  
            which_decoder (str) -- if fix for checkerboard effect then "conv", if old upsample then "convT" (inverse transpose)
            eq_lr (bool) -- if use equalised learning rate layers, default: False
            
            different settings: 
            no_checkerboard: which_decoder='conv', encoder_padding=1, decoder_padding=1 (apply upsampling+conv2d in decoder part)
            old:  which_decoder='convT', encoder_padding=1, decoder_padding=1 (the model we have been using so far)
            classic_unet: which_decoder='convT', encoder_padding=0, decoder_padding=0 (no padding, leads to smaller o/p, mirror HE by 256 pixels in dataloader)
            classic_unet_no_checkerboard: which_decoder='conv', encoder_padding=0, decoder_padding=1 (no padding in encoder, upsampling+conv2d in decoder part, leads to smaller o/p, mirror HE by 256 pixels in dataloader)

        """
        super(unet_translator, self).__init__()
        self.flag_asymmetric = flag_asymmetric
        self.flag_multiscale = flag_multiscale
        self.last_activation = last_activation
        self.encoder_padding = encoder_padding
        self.eq_lr = eq_lr

        # ----- defining the input output channels for each block in encoder and decoder -----
        encoder_sizes = [n_input_channels]
        for i in range(depth-1): 
            power = min(i, 3)
            encoder_sizes.append(n_filter*2**power)

        decoder_sizes = encoder_sizes[::-1] # reverse 
        decoder_sizes[-1] = n_output_channels  
        
        # if asymmetric model, want to add two layers in front: simply add the number of filters in encoder_sizes
        if self.flag_asymmetric: 
            encoder_sizes[1:1] = [int(encoder_sizes[1]/2**2), int(encoder_sizes[1]/2)]

        # ----- defining different part of model as list and connecting them in forward pass -----
        self.encoder = nn.ModuleList()
        self.center_block = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # ----- encoder -----
        for i in range(len(encoder_sizes)-1):
            
        # if multiscale and i even (except for i=0), then add 3 he channels 
            self.encoder.append(
                BNDownBlock(encoder_sizes[i] + n_input_channels * self.flag_multiscale * ((i%2)==0) * (i!=0), 
                            encoder_sizes[i+1], padding=encoder_padding, eq_lr=self.eq_lr)
            )
        
        # ----- bottleneck/center block -----
        self.center_block.append(BNDownBlock(encoder_sizes[-1], encoder_sizes[-1], padding=encoder_padding, eq_lr=self.eq_lr)) 
        self.center_block.append(BNUpBlock(encoder_sizes[-1], encoder_sizes[-1], which_conv=which_decoder, padding=decoder_padding, eq_lr=self.eq_lr))
        
        # ----- decoder -----
        for i in range(depth-2):
            self.decoder.append(
                BNUpBlock(decoder_sizes[i]*2, decoder_sizes[i+1], which_conv=which_decoder, padding=decoder_padding, eq_lr=self.eq_lr)
            )
            
        # high resolution final output
        self.decoder.append(output_block(decoder_sizes[i+1]*2, decoder_sizes[i+2], which_conv=which_decoder, 
                                         which_activation=self.last_activation, padding=decoder_padding, eq_lr=self.eq_lr))

        # ----- if flag_multiscale intermediate low resolution outputs -----
        self.indices = []

        if self.flag_multiscale: 
            self.to_imc = nn.ModuleList()
            for output_size in decoder_sizes[(len(decoder_sizes))%2::2][0:-1]:
                 self.to_imc.append(
                    output_block(output_size*2, decoder_sizes[-1], which_conv=which_decoder, 
                                 which_activation=self.last_activation, padding=decoder_padding, eq_lr=self.eq_lr)
                ) 
                    
            # indices of layers in decoder for which we need intermediate outputs
            self.indices = [depth%2 + j*2 for j in range((depth-2)//2)]


    def forward(self, x):
        #  ----- passing input in encoder, saving outputs for skip connection with decoder  ----- 
        he_img = x
        encoder_outs = []
        
        for i, module in enumerate(self.encoder):             
            flag_concat_he = self.flag_multiscale * ((i%2)==0) * (i!=0)
            if flag_concat_he:
                he_crop = ttf.center_crop(ttf.resize(he_img, he_img.shape[-1] // (2**i)), [x.shape[2], x.shape[3]])
                x = torch.cat([x, he_crop], dim=1) 
            x = module(x)
            encoder_outs.append(x)
        
        # ----- passing input in center block ----- 
        for i, module in enumerate(self.center_block):
            x = module(x)
            
        #  ----- passing input in decoder block, saving intermediate outputs if multiscale -----  
        j=0
        outputs = []        
        for i, module in enumerate(self.decoder):
            
            encoder_out = ttf.center_crop(encoder_outs[-(i+1)], [x.shape[2], x.shape[3]])
            x = torch.cat([encoder_out, x], dim=1) 
            if i in self.indices: 
                outputs.append(self.to_imc[j](x))
                j+=1            
            x = module(x)
        outputs.append(x) # add output from last layer of decoder
        return outputs
    
    
# -----------------------------------------------------------------
# Discriminator model for Histoplexer
# -----------------------------------------------------------------
class Discriminator(torch.nn.Module):    
    def __init__(self, n_output_channels, flag_asymmetric=True, flag_multiscale=True, n_input_channels=3, n_filter=32, depth=6, mbdis=True, eq_lr=False):
        """
        Parameters:
            n_output_channels (int) -- desired number of channels in output, here for IMC 
            flag_asymmetric (bool) -- if input higher resolution than output 
            flag_multiscale (bool) -- if multiscale setting, want outputs from intermediate layers  
            n_input_channels (int) -- number of channels in input, here for H&E 
            n_filter (int) -- desired number of filters in the first symmetric block of model 
            depth (int) -- depth of model, including center block, excluding block for asymmetric setting  
            mbdis (bool) -- if use mini batch discrimination
            eq_lr (bool) -- if use equalised learning rate layers, default: False
            
        """
        super(Discriminator, self).__init__()
        self.flag_asymmetric = flag_asymmetric
        self.flag_multiscale = flag_multiscale
        self.depth = depth 
        self.mbdis = mbdis 
        self.mbdis_n_chan = 1 if self.mbdis else 0
        self.eq_lr = eq_lr
        
        # ----- defining the input output channels for each block in discriminator -----
        discriminator_sizes = [n_input_channels]
        for i in range(self.depth-1): 
            power = min(i, 3)
            discriminator_sizes.append(n_filter*2**power)
                      
        # ----- adding initial layers if asymmetric setting  -----
        if self.flag_asymmetric:
            self.asymmetric_block = nn.ModuleList()
            asymmetric_sizes = [discriminator_sizes[0], int(discriminator_sizes[1]/4), int(discriminator_sizes[1]/2)]
            
            for i in range(2): 
                self.asymmetric_block.append(
                    SNDownBlock(asymmetric_sizes[i], asymmetric_sizes[i+1], eq_lr=self.eq_lr)
                )
                
            discriminator_sizes[1:1] = [int(discriminator_sizes[1]/2)]
        else:

            discriminator_sizes[1:1] = [0]

        # ----- adding initial layers discriminator  -----
        self.discriminator = nn.ModuleList()

        for i in range(0, ((self.depth//2)*2)-1): 
            flag_concat = (self.flag_multiscale * ((i%2)==0)) or (i==0) # append for sure when i=0 and also when multiscale and i even 
            self.discriminator.append(
                SNDownBlock(discriminator_sizes[i+1] + flag_concat*(n_input_channels + n_output_channels), 
                            discriminator_sizes[i+2],
                            eq_lr=self.eq_lr)
            )
            
        
        # ----- adding layers for output score maps   -----
        score_maps_sizes = discriminator_sizes[2::2]

        if self.flag_multiscale: 
            self.score_maps = nn.ModuleList()
            for i in range((self.depth//2)-1): 
                if self.mbdis:
                    if self.eq_lr:
                        self.score_maps.append(
                            nn.Sequential(
                                MinibatchStdLayer(group_size=32, n_chan=self.mbdis_n_chan),
                                nn.utils.spectral_norm(EQConv2d(score_maps_sizes[i]+self.mbdis_n_chan, 1, kernel_size=3, padding=1, stride=1))
                            )
                        )
                    else:
                        self.score_maps.append(
                            nn.Sequential(
                                MinibatchStdLayer(group_size=32, n_chan=self.mbdis_n_chan),
                                nn.utils.spectral_norm(nn.Conv2d(score_maps_sizes[i]+self.mbdis_n_chan, 1, kernel_size=3, padding=1, stride=1))
                            )
                        )
                else:
                    if self.eq_lr:
                        self.score_maps.append(
                            nn.utils.spectral_norm(EQConv2d(score_maps_sizes[i], 1, kernel_size=3, padding=1, stride=1))
                        )
                    else:
                        self.score_maps.append(
                            nn.utils.spectral_norm(nn.Conv2d(score_maps_sizes[i], 1, kernel_size=3, padding=1, stride=1))
                        )              
                    
        if self.mbdis:
            if self.eq_lr:
                self.discriminator.append(
                    nn.Sequential(
                        MinibatchStdLayer(group_size=32, n_chan=self.mbdis_n_chan),
                        nn.utils.spectral_norm(EQConv2d(score_maps_sizes[int(self.depth//2)-1]+1, 1, kernel_size=3, padding=1, stride=1))
                    )
                )
            else:
                self.discriminator.append(
                    nn.Sequential(
                        MinibatchStdLayer(group_size=32, n_chan=self.mbdis_n_chan),
                        nn.utils.spectral_norm(nn.Conv2d(score_maps_sizes[int(self.depth//2)-1]+1, 1, kernel_size=3, padding=1, stride=1))
                    )
                )  
        else:
            if self.eq_lr:
                self.discriminator.append(
                    nn.utils.spectral_norm(EQConv2d(score_maps_sizes[int(self.depth//2)-1], 1, kernel_size=3, padding=1, stride=1))
                )
            else:
                self.discriminator.append(
                    nn.utils.spectral_norm(nn.Conv2d(score_maps_sizes[int(self.depth//2)-1], 1, kernel_size=3, padding=1, stride=1))
                )              
        
    def forward(self, he_img, imc_preds):
        x = he_img
        outputs = []
        if self.flag_asymmetric:
            for i, module in enumerate(self.asymmetric_block):
                x = module(x)
                
        for i, module in enumerate(self.discriminator):
            imc_index = int((-1)*((i+1)//2)) -1
            
            if (i==0 and self.flag_asymmetric): 
                x = torch.cat([x, ttf.resize(he_img, he_img.shape[-1] // (2**(i+2))), imc_preds[imc_index]], dim=1)
            
            elif (i==0 and not self.flag_asymmetric): 
                x = torch.cat([ttf.resize(he_img, he_img.shape[-1] // (2**(i+2))), imc_preds[imc_index]], dim=1)
    
            elif (i%2==0 and self.flag_multiscale):
                x = torch.cat([x, ttf.resize(he_img, he_img.shape[-1] // (2**(i+2))), imc_preds[imc_index]], dim=1)
                
            x = module(x)
            
            if self.flag_multiscale and (i%2==0) and i<(self.depth//2): 
                outputs.append(self.score_maps[int(i//2)](x))
        outputs.append(x)
        return outputs