import os
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
from codebase.utils.constants import *


#################################################################################
#------------------------ r1 regularisation loss utils -------------------------#
#################################################################################

def get_r1(discriminator, he, imc_list, gamma_0, lazy_c):
    total_r1_loss = 0
    
    # Make sure requires_grad is True for all tensors
    he.detach().requires_grad_(True)
    imc_list = [imc.detach().requires_grad_(True) for imc in imc_list]
    
    score_maps = discriminator(he, imc_list)

    for i, score_map in enumerate(score_maps):
        imc = imc_list[len(imc_list) - i -1]
        bs = imc.shape[0]
        img_size = imc.shape[-1]
        r1_gamma = gamma_0 * (img_size ** 2 / bs)
        
        # compute R1 regularization
        r1_grad = torch.autograd.grad(outputs=[score_map.sum()],
                                      inputs=[imc],
                                      create_graph=True,
                                      only_inputs=True)
        r1_penalty = 0
        for grad in r1_grad:
            r1_penalty += grad.square().sum(dim=[1, 2, 3])
            r1_loss = r1_penalty.mean() * (r1_gamma / 2) * lazy_c

        total_r1_loss += r1_loss
    
    return total_r1_loss


#################################################################################
#--------------------------- contrastive loss utils ----------------------------#
#################################################################################

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

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
            if debug:
                print(classname)
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
    

def init_net(net, init_type='normal', init_gain=0.02, debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x, dim=1):
        # norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        # out = x.div(norm + 1e-7)
        # FDL: To avoid sqrting 0s, which causes nans in grad
        norm = (x + 1e-7).pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    

class PatchSampleF(nn.Module):
    def __init__(self, 
                use_mlp=True, 
                init_type='normal', 
                init_gain=0.02, 
                nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.to(self.device)
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain)
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
    

class PatchNCELoss(nn.Module):
    def __init__(self, 
                batch_size,
                total_step,
                n_step_decay,
                scheduler='lambda',
                lookup='linear',
                nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.batch_size = batch_size
        self.total_step = total_step
        self.n_step_decay = n_step_decay
        self.scheduler = scheduler
        self.lookup = lookup
        self.nce_T = 0.07
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch

    def forward(self, feat_q, feat_k, current_step=-1):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))  # [B*num_patches, 1, C] @ [B*num_patches, C, 1] --> [B*num_patches, 1]
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        # include the negatives from the entire minibatch.
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # [B, num_patches, C] @ [B, C, num_patches] --> [B, num_patches, num_patches]

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        # weight loss based on current step and positive pairs' similarity (https://arxiv.org/abs/2303.06193)
        if current_step != -1:
            # Compute scheduling
            t = (current_step - 1) / self.total_step
            if self.scheduler == 'sigmoid':
                p = 1 / (1 + np.exp((t - 0.5) * 10))
            elif self.scheduler == 'linear':
                p = 1 - t
            elif self.scheduler == 'lambda':
                k = 1 - self.n_step_decay / self.total_step
                m = 1 / (1 - k)
                p = m - m * t if t >= k else 1.0
            elif self.scheduler == 'zero':
                p = 1.0
            else:
                raise ValueError(f"Unrecognized scheduler: {self.scheduler}")
            # Weight lookups
            w0 = 1.0
            x = l_pos.squeeze().detach()
            if self.lookup == 'top':
                x = torch.where(x > 0.0, x, torch.zeros_like(x))
                w1 = torch.sqrt(1 - (x - 1) ** 2)
            elif self.lookup == 'linear':
                w1 = torch.relu(x)
            elif self.lookup == 'bell':
                sigma, mu, sc = 1, 0, 4
                w1 = 1 / (sigma * np.sqrt(2 * torch.pi)) * torch.exp(-((x - 0.5) * sc - mu) ** 2 / (2 * sigma ** 2))
            elif self.lookup == 'uniform':
                w1 = torch.ones_like(x)
            else:
                raise ValueError(f"Unrecognized lookup: {self.lookup}")
            # Apply weights with schedule
            w = p * w0 + (1 - p) * w1
            # Normalize
            w = w / w.sum() * len(w)
            loss *= w
        return loss
    

#################################################################################
#------------------------------- VGG19 utils -----------------------------------#
#################################################################################


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class VGG19(torch.nn.Module):
    def __init__(self, MODELDIR, requires_grad=False):
        super().__init__()
        vgg19 = torchvision.models.vgg19(pretrained=False) 
        vgg19.load_state_dict(torch.load(os.path.join(MODELDIR, 'vgg19_model.pth')))
        vgg_pretrained_features = vgg19.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
        self.norm = Normalization()

    def forward(self, X):
        # Convert single channel binary image to 3-channel image using torch.cat
        if X.shape[1] == 1:  # Check if it's single-channel
            X = torch.cat([X, X, X], dim=1)  # Concatenate along the channel dimension
        
        X = self.norm(X) # normalisation using ImageNet statistics
        
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out