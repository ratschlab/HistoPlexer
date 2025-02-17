import numpy as np
import math
import torch
import torchvision


def get_tensor_from_numpy(np_img):
    ''' Construct torch tensor from a numpy array
    np_img: numpy array of shape [H,W,C]
    returns a torch tensor with shape [C,H,W]
    '''
    torch_img = np_img.transpose((2, 0, 1))
    torch_img = np.ascontiguousarray(torch_img)
    torch_img = torch.from_numpy(torch_img).float().unsqueeze(0)
    return torch_img


def get_target_shapes(model_depth=6, input_shape=4000):
    ''' Find the desired shapes of predicted images
    model_depth: depth of the model (eg 6)
    input_shape: shape ([0]) of the input H&E image
    returns a list of desired shapes of the predictions in a decreasing order
    '''
    desired_shapes = []
    for j in range(model_depth//2):
        desired_shapes.append(input_shape//(2**(2*(j+1))))
    return desired_shapes

def pad_img(torch_img, input_shape=4000):
    ''' Pad image to make it compatible wth the model (eg /2)
    torch_img: torch tensor (expects a squared image)
    input_shape: shape ([0]) of the input H&E image
    returns a padded image (torch tensor)
    '''
    padding = (2**(round(math.log(input_shape, 2))) - input_shape)//2
    torch_img = torchvision.transforms.Pad(padding, padding_mode='reflect')(torch_img)
    return torch_img
