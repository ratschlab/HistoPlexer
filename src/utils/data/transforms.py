import random
import torch
from torchvision import transforms

from src.utils.data.HE_transforms import HEDJitter, RandomAffineCV2


def shared_transforms(img1, img2, p=0.5):
    """
    Apply simultaneous transformations to H&E and IMC data.

    This function applies random horizontal or vertical flips and random
    rotations at multiples of 90 degrees to both images.

    Args:
        img1: H&E ROI (expected).
        img2: IMC ROI (expected).
        p: Probability of applying each transformation (default is 0.5).

    Returns:
        A tuple of transformed images (img1, img2).
    """
    # Random horizontal flipping
    if random.random() < p:
        img1 = transforms.functional.hflip(img1)
        img2 = transforms.functional.hflip(img2)

    # Random vertical flipping
    if random.random() < p:
        img1 = transforms.functional.vflip(img1)
        img2 = transforms.functional.vflip(img2)
        
    # Random 90 degree rotation
    if random.random() < p:
        angle = random.choice([90, 180, 270])
        img1 = transforms.functional.rotate(img1, angle)
        img2 = transforms.functional.rotate(img2, angle)

    return img1, img2


def HE_transforms(img, p=[0.0, 0.5, 0.5]):
    """
    Apply transformations specific to H&E ROIs.

    This function applies color jitter, HED jitter, and random affine transforms
    with specified probabilities.

    Args:
        img: H&E ROI (expected).
        p: List of probabilities for each transformation, in the order of 
           [color jitter, HED jitter, affine transform].

    Returns:
        Transformed image.
    """
    # Random color jitter
    if random.random() < p[0]:
        jitter = transforms.ColorJitter(brightness=.15, hue=.05, saturation=0.15)
        img = jitter(img)

    # Random HED jitter
    if random.random() < p[1]:
        img = torch.permute(img, (1, 2, 0))  # channel first to last    
        hedjitter = HEDJitter(theta=0.01)  # from HE_transforms
        img = hedjitter(img)

    # Random affine transform
    if random.random() < p[2]:
        if img.shape[2] != 3: 
            img = torch.permute(img, (1, 2, 0))  # channel first to last    
        randomaffine = RandomAffineCV2(alpha=0.02)  # from HE_transforms
        img = randomaffine(img)

    return img