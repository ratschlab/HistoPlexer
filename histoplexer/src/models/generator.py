import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
from typing import List

from src.models.layers import *
from src.models.base_model import BaseModel

# -----------------------------------------------------------------
# Translator model for Histoplexer
# -----------------------------------------------------------------

class unet_translator(BaseModel):
    """A U-Net based translator model for image-to-image translation tasks.

    This model is designed with options for high-resolution input, multi-scale outputs, and configurable depths.

    Attributes:
        use_high_res (bool): If True, the input is of higher resolution than the output.
        use_multiscale (bool): If True, outputs from intermediate layers are also provided.
        depth (int): The depth of the generator model.
        encoder (nn.ModuleList): List of modules in the encoder part of the U-Net.
        center_block (nn.ModuleList): List of modules in the bottleneck/center block of the U-Net.
        decoder (nn.ModuleList): List of modules in the decoder part of the U-Net.
        to_imc (nn.ModuleList): List of modules for intermediate outputs (if `use_multiscale` is True).
        indices (list): Indices of layers in the decoder for which intermediate outputs are needed.
    """

    def __init__(self, 
                input_nc=3, 
                output_nc=11, 
                use_high_res=True, 
                use_multiscale=True, 
                ngf=32, 
                depth=6, 
                encoder_padding=1, 
                decoder_padding=1, 
                device=torch.device('cuda:0')):
        """
        Initializes the unet_translator model with specified configurations.

        Args:
            input_nc (int): Number of channels in the input image. Defaults to 3.
            output_nc (int): Number of channels in the output image. Defaults to 10.
            use_high_res (bool): If True, input has a higher resolution than output. Defaults to True.
            use_multiscale (bool): If True, outputs from intermediate layers are desired. Defaults to True.
            ngf (int): Number of filters in the first symmetric block of the model. Defaults to 32.
            depth (int): Depth of the model, including the center block. Defaults to 6.
            encoder_padding (int): Padding used in the encoder layers. Defaults to 1.
            decoder_padding (int): Padding used in the decoder layers. Defaults to 1.
        """
        super(unet_translator, self).__init__(device)
        self.use_high_res = use_high_res
        self.use_multiscale = use_multiscale
        self.depth = depth

        # defining the input output channels for each block in encoder and decoder 
        encoder_sizes = [input_nc] + [ngf * min(2**i, 8) for i in range(depth - 1)]
        decoder_sizes = encoder_sizes[::-1] # reverse 
        decoder_sizes[-1] = output_nc
        
        # if use_high_res input, add two layers in front, simply add the number of filters in encoder_sizes
        if self.use_high_res:
            encoder_sizes[1:1] = [int(encoder_sizes[1]/2**2), int(encoder_sizes[1]/2)]
            
        # initialize encoder
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_sizes) - 1):
            # if multiscale and i even (except for i=0), then add 3 he channels 
            in_channels = encoder_sizes[i] + (input_nc if self.use_multiscale and i % 2 == 0 and i != 0 else 0) # if multiscale and i even (except for i=0), then add 3 he channels
            self.encoder.append(BNDownBlock(in_channels, encoder_sizes[i + 1], padding=encoder_padding))
            
        # initialize center block
        self.center_block = nn.ModuleList()
        self.center_block.append(BNDownBlock(encoder_sizes[-1], encoder_sizes[-1], padding=encoder_padding)) 
        self.center_block.append(BNUpBlock(encoder_sizes[-1], encoder_sizes[-1], padding=decoder_padding))
        
       # initialize decoder
        self.decoder = nn.ModuleList()
        for i in range(depth - 2):
            in_channels = decoder_sizes[i] * 2
            self.decoder.append(BNUpBlock(in_channels, decoder_sizes[i + 1], padding=decoder_padding))
            
        # add high res final output layer        
        self.decoder.append(output_block(decoder_sizes[i + 1] * 2, decoder_sizes[i + 2], padding=decoder_padding))

        # initialize intermediate output layers for multiscale setup
        if self.use_multiscale:
            self.to_imc = nn.ModuleList()
            for output_size in decoder_sizes[(len(decoder_sizes))%2::2][0:-1]:
                self.to_imc.append(output_block(output_size * 2, output_nc, padding=decoder_padding))
            self.indices = [depth%2 + j*2 for j in range((depth-2)//2)]
        else:
            self.indices = []
            
        self.to(self.device)

    def forward(self, x: torch.Tensor, encode_only: bool=False) -> List[torch.Tensor]:
        """
        Forward pass of the unet_translator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: A list of output tensors, including multi-scale outputs if enabled.
        """
        x_mem = x
        encoder_outs = []
        
        if encode_only:
            feats = []
        
        # encoder pass with skip connections
        for i, module in enumerate(self.encoder):
            if self.use_multiscale and ((i % 2) == 0) and (i != 0):
                x_mem_crop = ttf.center_crop(ttf.resize(x_mem, x_mem.shape[-1] // (2**i)), x.shape[2:4])
                x = torch.cat([x, x_mem_crop], dim=1) 
            x = module(x)
            encoder_outs.append(x)
            if encode_only:
                feats.append(x)

        # center block pass
        for i, module in enumerate(self.center_block):
            x = module(x)
            if encode_only:
                feats.append(x)

        if encode_only:
            return feats
        else:
            j=0
            outputs = []
            # decoder pass with skip connections and multiscale outputs
            for i, module in enumerate(self.decoder):
                encoder_out = ttf.center_crop(encoder_outs[-(i + 1)], x.shape[-2:])
                x = torch.cat([encoder_out, x], dim=1)
                if i in self.indices:
                    outputs.append(self.to_imc[j](x))
                    j+=1    
                x = module(x)
                
            # append final output
            outputs.append(x)
            return outputs