import torch
import torch.nn as nn
import numpy as np

from src.models.base_model import BaseModel

    
class Normalize(nn.Module):
    """
    A module for normalizing tensors.

    Attributes:
        power (float): The exponent to use in normalization.
    """

    def __init__(self, power=2):
        """
        Initialize the Normalize module.

        Args:
            power (float): The exponent to use in normalization.
        """
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Forward pass for normalizing a tensor.

        Args:
            x (torch.Tensor): The input tensor to normalize.
            dim (int): The dimension along which to compute the norm.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        norm = (x + 1e-7).pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class PatchSampleF(BaseModel):
    def __init__(self, 
                 use_mlp=True, 
                 nc=256, device=torch.device('cuda:0')):
        """
        Initialize the PatchSampleF module.

        Args:
            use_mlp (bool): Whether to apply an MLP to each patch.
            nc (int): Number of channels for the MLP output.
        """
        super(PatchSampleF, self).__init__(device)  # Calls BaseModel's __init__
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        
        self.to(self.device)

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.to(self.device)
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # [B, H, W, C] --> [B, H*W, C]
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape.flatten(0, 1) # [B, num_patches, C] --> [B*num_patches, C]
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.reshape([B, H, W, x_sample.shape[-1]]).permute(0, 3, 1, 2)
            return_feats.append(x_sample)
        return return_feats, return_ids
    
    def init_model(self, dummy_input):
        """
        Initializes the network with a dummy pass. Necessary for creating certain layers that depend on input size.

        Args:
            dummy_input (torch.Tensor): Dummy data for network initialization.
        """
        self.forward(dummy_input)