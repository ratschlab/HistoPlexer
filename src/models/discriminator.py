import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
from typing import Tuple, List

from src.models.layers import *
from src.models.base_model import BaseModel

# -----------------------------------------------------------------
# Discriminator model for Histoplexer
# -----------------------------------------------------------------

class Discriminator(BaseModel):
    """Discriminator model with options for high-resolution input and multi-scale outputs.

    Attributes:
        use_high_res (bool): If True, the input is of higher resolution than the output.
        use_multiscale (bool): If True, outputs from intermediate layers are also provided.
        depth (int): The depth of the discriminator model.
        asymmetric_block (nn.ModuleList): Downsampling blocks for high-resolution input.
        dis (nn.ModuleList): Discriminator layers.
        score_maps (nn.ModuleList): Layers for generating score maps in multi-scale output.
    """

    def __init__(self, input_nc=3, output_nc=11, use_high_res=True, 
                 use_multiscale=True, ngf=32, depth=6, device=torch.device('cuda:0')):
        
        
        """Initializes the Discriminator model.

        Args:
            input_nc (int): Number of input channels.
            output_nc (int): Number of output channels.
            use_high_res (bool): Flag to use high-resolution input.
            use_multiscale (bool): Flag to use multi-scale outputs.
            ngf (int): Number of generator filters in the first conv layer.
            depth (int): Depth of the discriminator model.
        """
        super(Discriminator, self).__init__(device)
        self.use_high_res = use_high_res
        self.use_multiscale = use_multiscale
        self.depth = depth

        # initialize discriminator sizes
        dis_sizes = [input_nc] + [ngf * min(2**i, 8) for i in range(depth - 1)]
    
        # downsampling for high-resolution input
        if self.use_high_res:
            self.asymmetric_block = nn.ModuleList()
            asymmetric_sizes = [dis_sizes[0], int(dis_sizes[1]/4), int(dis_sizes[1]/2)]
            for i in range(2):
                self.asymmetric_block.append(SNDownBlock(asymmetric_sizes[i], asymmetric_sizes[i+1]))
            dis_sizes[1:1] = [int(dis_sizes[1]/2)]
        else: 
            dis_sizes[1:1] = [0]

        # discriminator layers
        self.discriminator = nn.ModuleList()
        for i in range(0, ((self.depth//2)*2)-1):
            flag_concat = (self.use_multiscale and ((i%2)==0)) or (i==0) # append for sure when i=0 and also when multiscale and i even 
            self.discriminator.append(SNDownBlock(in_ch=dis_sizes[i+1] + flag_concat*(input_nc + output_nc), out_ch=dis_sizes[i+2]))

        # score maps for multi-scale output
        score_maps_sizes = dis_sizes[2::2]
        if self.use_multiscale:
            self.score_maps = nn.ModuleList()
            for i in range((self.depth//2)-1):
                self.score_maps.append(nn.Sequential(
                        MinibatchStdLayer(group_size=32, n_chan=1),
                        nn.utils.spectral_norm(nn.Conv2d(score_maps_sizes[i]+1, 1, kernel_size=3, padding=1, stride=1)))
                        )            
                    
        self.discriminator.append(nn.Sequential(
                MinibatchStdLayer(group_size=32, n_chan=1),
                nn.utils.spectral_norm(nn.Conv2d(score_maps_sizes[int(self.depth//2)-1]+1, 1, kernel_size=3, padding=1, stride=1)))
                ) 
        
        self.to(self.device)

    def forward(self, x: Tuple[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
        """Forward pass of the Discriminator.

        Args:
            x (Tuple[torch.Tensor, List[torch.Tensor]]): Tuple containing the input tensor and a list of conditional tensors.

        Returns:
            List[torch.Tensor]: A list of output tensors, including multi-scale outputs if enabled.
        """
        x, imc = x[0], x[1]
        x_mem = x
        outputs = []

        # downsampling for high-resolution input
        if self.use_high_res:
            for i, module in enumerate(self.asymmetric_block):
                x = module(x)

        # processing through discriminator layers
        for i, module in enumerate(self.discriminator):
            imc_index = int((-1)*((i+1)//2)) -1
            if (i==0 and self.use_high_res): 
                x = torch.cat([x, ttf.resize(x_mem, x_mem.shape[-1] // (2**(i+2))), imc[imc_index]], dim=1)
            elif (i==0 and not self.use_high_res):
                x = torch.cat([x_mem, imc[imc_index]], dim=1)
            elif (i%2==0 and not self.use_high_res and self.use_multiscale):
                x = torch.cat([x, ttf.resize(x_mem, x_mem.shape[-1] // (2**i)), imc[imc_index]], dim=1)
            elif (i%2==0 and self.use_multiscale):
                x = torch.cat([x, ttf.resize(x_mem, x_mem.shape[-1] // (2**(i+2))), imc[imc_index]], dim=1)

            x = module(x)
            # generating multi-scale score maps
            if self.use_multiscale and (i%2==0) and i<(self.depth//2):
                outputs.append(self.score_maps[int(i//2)](x))

        outputs.append(x)
        return outputs
