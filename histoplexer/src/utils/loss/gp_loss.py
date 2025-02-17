import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussPyramidConv(nn.Module): # checked
    """
    Code borrowed from: https://github.com/csjliang/LPTN
    A module that applies Gaussian pyramid convolution to an input image. It creates a series of 
    downsampled images, each progressively lower in resolution and smoothed using Gaussian blur.

    Attributes:
        num_high (int): The number of high-resolution levels in the Gaussian pyramid.
        num_blur (int): The number of times Gaussian blur is applied at each level.
        channels (int): The number of channels in the input images.
    """

    def __init__(self, 
                 num_high=3, 
                 num_blur=4, 
                 channels=11):
        """
        Initialize the GaussPyramidConv module.

        Args:
            num_high (int): The number of high-resolution levels in the Gaussian pyramid.
            num_blur (int): The number of times Gaussian blur is applied at each level.
            channels (int): The number of channels in the input images.
        """
        super(GaussPyramidConv, self).__init__()
        self.num_high = num_high
        self.num_blur = num_blur
        self.channels = channels

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a blur and downsample operation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be downsampled.

        Returns:
            torch.Tensor: The downsampled tensor.
        """
        return kornia.filters.blur_pool2d(x, kernel_size=3, stride=2)

    def conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to the input image.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The blurred image tensor.
        """
        return kornia.filters.gaussian_blur2d(img, (3, 3), (1, 1))
    
    def forward(self, img: torch.Tensor) -> [torch.Tensor]:
        """
        Apply Gaussian pyramid convolution to the input image. This method generates a list of 
        images, each representing a level of the Gaussian pyramid.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            list[torch.Tensor]: A list of tensors representing the Gaussian pyramid levels.
        """
        current = img
        pyr = [current]
        for _ in range(self.num_high):
            for _ in range(self.num_blur):
                current = self.conv_gauss(current)
            down = self.downsample(current)
            current = down
            pyr.append(current)
        return pyr
    
    
class GaussPyramidLoss(nn.Module):
    """
    A module to compute the Gaussian Pyramid Loss between two tensors. This loss applies 
    L1 loss at each level of a Gaussian pyramid representation of the tensors.

    Attributes:
        pyr_conv (GaussPyramidConv): Instance of GaussPyramidConv for generating Gaussian pyramid.
        gp_weights (List[float]): Weights for each level of the Gaussian pyramid.
    """

    def __init__(self, 
                 num_high=3, 
                 num_blur=4, 
                 channels=11, 
                 gp_weights=[0.0625, 0.125, 0.25, 1.0]):
        """
        Initialize the GaussianPyramidLoss module.

        Args:
            num_high (int): The number of high-resolution levels in the Gaussian pyramid.
            num_blur (int): The number of times Gaussian blur is applied at each level.
            channels (int): The number of channels in the input images.
            gp_weights (List[float]): Weights for each level of the Gaussian pyramid. 
                                      Defaults to [0.0625, 0.125, 0.25, 1.0].

        Raises:
            AssertionError: If the length of gp_weights does not match num_high + 1.
        """
        super(GaussPyramidLoss, self).__init__()
        self.pyr_conv = GaussPyramidConv(num_high=num_high, num_blur=num_blur, channels=channels)
        
        assert len(gp_weights) == num_high + 1, \
            f"Length of gp_weights must be equal to num_high + 1. Expected length {num_high + 1}, got {len(gp_weights)}."
        self.gp_weights = gp_weights

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Gaussian Pyramid Loss between fake and real images.

        Args:
            fake (torch.Tensor): The fake images tensor.
            real (torch.Tensor): The real images tensor.

        Returns:
            torch.Tensor: The computed Gaussian Pyramid Loss.
        """
        fake_pyr = self.pyr_conv(fake)
        real_pyr = self.pyr_conv(real)
        loss_pyramid = [F.l1_loss(pf, pr, reduction='none').mean(dim=(2, 3)) for pf, pr in zip(fake_pyr, real_pyr)]
        loss_pyramid = [l * w for l, w in zip(loss_pyramid, self.gp_weights)]
        loss = torch.mean(torch.stack(loss_pyramid))
        return loss