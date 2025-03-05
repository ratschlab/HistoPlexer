import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
from typing import List

from src.models.layers import *
from src.models.base_model import BaseModel

class unet_translator(BaseModel):
    """A U-Net based translator model for image-to-image translation tasks with support for feature integration.

    This model is designed with options for high-resolution input, multi-scale outputs, configurable depths,
    and integration of additional feature vectors at the bottleneck.

    Attributes:
        use_high_res (bool): If True, the input is of higher resolution than the output.
        use_multiscale (bool): If True, outputs from intermediate layers are also provided.
        depth (int): The depth of the generator model.
        has_extra_features (bool): If True, model accepts additional feature vectors.
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
                extra_feature_size=0,
                device=torch.device('cuda:0')):
        """
        Initializes the unet_translator model with specified configurations.

        Args:
            input_nc (int): Number of channels in the input image. Defaults to 3.
            output_nc (int): Number of channels in the output image. Defaults to 11.
            use_high_res (bool): If True, input has a higher resolution than output. Defaults to True.
            use_multiscale (bool): If True, outputs from intermediate layers are desired. Defaults to True.
            ngf (int): Number of filters in the first symmetric block of the model. Defaults to 32.
            depth (int): Depth of the model, including the center block. Defaults to 6.
            encoder_padding (int): Padding used in the encoder layers. Defaults to 1.
            decoder_padding (int): Padding used in the decoder layers. Defaults to 1.
            extra_feature_size (int): Size of extra features to integrate at bottleneck. Defaults to 1024.
        """
        super(unet_translator, self).__init__(device)
        self.use_high_res = use_high_res
        self.use_multiscale = use_multiscale
        self.depth = depth
        self.has_extra_features = extra_feature_size > 0
        self.extra_feature_size = extra_feature_size

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
            in_channels = encoder_sizes[i] + (input_nc if self.use_multiscale and i % 2 == 0 and i != 0 else 0)
            self.encoder.append(BNDownBlock(in_channels, encoder_sizes[i + 1], padding=encoder_padding))
        
        # Feature integration components for bottleneck
        if self.has_extra_features:
            bottleneck_channels = encoder_sizes[-1]
            
            # Feature projection: from bottleneck+extra_features to bottleneck channels
            self.feature_integration = nn.Conv2d(
                bottleneck_channels + self.extra_feature_size,
                bottleneck_channels,
                kernel_size=1
            )
            
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
    
    def forward(self, x: torch.Tensor, extra_features: torch.Tensor = None, encode_only: bool=False) -> List[torch.Tensor]:
        """
        Forward pass of the unet_translator with feature integration at bottleneck.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
            extra_features (torch.Tensor, optional): Additional feature tensor of shape (batch, feature_size).
            encode_only (bool): Whether to return encoder features only.

        Returns:
            List[torch.Tensor]: A list of output tensors, including multi-scale outputs if enabled.
        """
        x_mem = x
        encoder_outs = []
        
        if encode_only:
            feats = []
        
        # Encoder pass with skip connections
        for i, module in enumerate(self.encoder):
            if self.use_multiscale and ((i % 2) == 0) and (i != 0):
                x_mem_crop = ttf.center_crop(ttf.resize(x_mem, x_mem.shape[-1] // (2**i)), x.shape[2:4])
                x = torch.cat([x, x_mem_crop], dim=1) 
            x = module(x)
            encoder_outs.append(x)
            if encode_only:
                feats.append(x)
                
        # Handle feature integration at bottleneck
        if extra_features is not None and self.has_extra_features:
            batch_size = x.shape[0]
            spatial_h, spatial_w = x.shape[2:4]  # Usually 4x4 at bottleneck with depth=6
            
            # Check if extra_features has the correct size
            if extra_features.shape[1] != self.extra_feature_size:
                raise ValueError(f"Expected extra_features with {self.extra_feature_size} features, got {extra_features.shape[1]}")
            
            # Reshape and expand features to match spatial dimensions
            if len(extra_features.shape) == 2:  # (batch, features)
                expanded_features = extra_features.view(batch_size, self.extra_feature_size, 1, 1)
                expanded_features = expanded_features.expand(-1, -1, spatial_h, spatial_w)
            else:  # Already has spatial dimensions
                expanded_features = ttf.resize(extra_features, (spatial_h, spatial_w))
            
            # Concatenate with bottleneck features
            x = torch.cat([x, expanded_features], dim=1)
            
            # Project back to original channel count
            x = self.feature_integration(x)

        # Pass through center block
        for i, module in enumerate(self.center_block):
            x = module(x)
            if encode_only:
                feats.append(x)

        if encode_only:
            return feats
        else:
            j = 0
            outputs = []
            # Decoder pass with skip connections and multiscale outputs
            for i, module in enumerate(self.decoder):
                encoder_out = ttf.center_crop(encoder_outs[-(i + 1)], x.shape[-2:])
                x = torch.cat([encoder_out, x], dim=1)
                if i in self.indices:
                    outputs.append(self.to_imc[j](x))
                    j += 1    
                x = module(x)
                
            # Append final output
            outputs.append(x)
            return outputs