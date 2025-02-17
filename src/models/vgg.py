import os
import torch
import torchvision
import torch.nn as nn

from src.models.base_model import BaseModel

class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
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

class VGG19(BaseModel):
    def __init__(self, vgg_path=None, device=torch.device("cuda:0")):
        super(VGG19, self).__init__(device)
        try:
            vgg19 = torchvision.models.vgg19(pretrained=True) 
        except IOError as e:
            print("Internet connection required for downloading the pre-trained model is not available.")
            vgg19 = torchvision.models.vgg19(pretrained=False)
            assert vgg_path is not None, 'Pretrained model checkpoint is required.'
            vgg19.load_state_dict(torch.load(vgg_path))
            print(f"Pretrained vgg model loaded from {vgg_path}")

        vgg_pretrained_features = vgg19.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for i in range(2):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(2, 7):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(7, 12):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(12, 21):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(21, 30):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
                
        self.norm = Normalization(self.device)

        self.to(self.device)
        
    def forward(self, x):
        if x.shape[1] == 1:  # Check if it's single-channel
            x = torch.cat([x, x, x], dim=1)  # Concatenate along the channel dimension        

        x = self.norm(x) # normalisation using ImageNet statistics
        
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out