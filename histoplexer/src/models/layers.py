import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------
# Layers used in models for Histoplexer
# -----------------------------------------------------------------

class BNRelu(nn.Module):
    """Basic block for batch normalization followed by a leaky ReLU activation.
    Attributes:
        bn (nn.BatchNorm2d): Batch normalization layer.
        rl (nn.LeakyReLU): Leaky ReLU activation layer.
    """
    def __init__(self, num_features: int):
        """Initializes the BNRelu block with batch normalization and leaky ReLU.
        Args:
            num_features (int): Number of features (channels) in the input tensor.
        """
        super(BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.rl = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of the block.
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying batch normalization and leaky ReLU.
        """
        return self.rl(self.bn(inputs))
    

class BNDownBlock(nn.Module):
    """Basic block for convolution followed by max pooling and an optional batch norm and leaky ReLU.
    Includes downsampling using a stride of 2.
    Attributes:
        conv1 (nn.Conv2d): Convolutional layer.
        maxpool (nn.MaxPool2d): Max pooling layer for downsampling.
        bn1 (BNRelu): Optional batch norm and leaky ReLU layer.
    """
    def __init__(self, 
                in_ch: int, 
                out_ch: int, 
                ksize: int = 3, 
                padding: int = 1, 
                batch_norm: bool = True):
        """Initializes the BNDownBlock with convolution, max pooling, and optional batch norm and leaky ReLU.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            ksize (int): Kernel size for the convolutional layer. Defaults to 3.
            padding (int): Padding for the convolutional layer. Defaults to 1.
            batch_norm (bool): Whether to include batch norm and leaky ReLU. Defaults to True.
        """
        super(BNDownBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=2, padding_mode='reflect')
        if self.batch_norm: 
            self.bn1 = BNRelu(out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of the block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying convolution, max pooling, and optional batch norm and leaky ReLU.
        """
        x = self.conv1(x)
        if self.batch_norm: 
            x = self.bn1(x)
        return x
    

class SNDownBlock(nn.Module):
    """Basic block for convolution with spectral norm and leaky ReLU.
    Attributes:
        conv1 (nn.Module): Convolutional layer with spectral norm and stride 2 for downsampling.
        rl (nn.LeakyReLU): Leaky ReLU activation layer.
    """
    def __init__(self, 
                in_ch: int, 
                out_ch: int, 
                ksize: int = 3, 
                padding: int = 1):
        """Initializes the SNDownBlock with convolution, spectral norm, leaky ReLU, and max pooling.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            ksize (int): Kernel size for the convolutional layer. Defaults to 3.
            padding (int): Padding for the convolutional layer. Defaults to 1.
        """
        super(SNDownBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=2))
        self.rl = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of the block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying convolution with spectral norm, leaky ReLU, and max pooling.
        """
        x = self.rl(self.conv1(x))
        return x


class BNUpBlock(nn.Module):
    """Basic block for upsampling followed by convolution with an optional batch norm and leaky ReLU (for u-net decoder)

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer.
        bn1 (BNRelu): Optional batch norm and leaky ReLU layer.
    """

    def __init__(self, 
                in_ch: int, 
                out_ch: int, 
                ksize: int = 3, 
                padding: int = 1, 
                batch_norm: bool = True):
        """Initializes the BNUpBlock with upsampling and convolution with an optional batch norm and leaky ReLU.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            ksize (int): Kernel size for the convolutional layer. Defaults to 3.
            padding (int): Padding for the convolutional layer. Defaults to 1.
            batch_norm (bool): Whether to include batch norm and leaky ReLU. Defaults to True.
        """
        super(BNUpBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=1, padding_mode='reflect')
        if self.batch_norm:
            self.bn1 = BNRelu(out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of the block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying upsampling, convolution, and optional batch norm and leaky ReLU.
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.batch_norm: 
            return self.bn1(self.conv1(x))
        else: 
            return self.conv1(x)


class output_block(nn.Module):
    """Basic block for output layer with upsampling followed by convolution and activation.

    Attributes:
        upsample (nn.Upsample): Upsampling layer.
        conv1 (nn.Conv2d): Convolutional layer.
        act1 (nn.Module): Activation function layer.
    """
    def __init__(self, 
                in_ch: int, 
                out_ch: int, 
                ksize: int = 3, 
                padding: int = 1):
        """Initializes the output_block with specified configurations.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            ksize (int): Kernel size for the convolutional layer. Defaults to 3.
            padding (int): Padding for the convolutional layer. Defaults to 1.
            act (str): Type of activation function ('relu', 'lrelu', 'identity', 'sigmoid', 'tanh').
        """
        super(output_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=1, padding_mode='reflect')
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying upsampling using interpolation, convolution, and activation.
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.act1(self.conv1(x))
    

class MinibatchStdLayer(nn.Module):
    """
    Implements Minibatch Standard Deviation Layer.
    
    This layer helps improve GAN training by adding a statistical feature to the 
    discriminator, encouraging diversity in generated samples. It computes the 
    standard deviation across groups of images in a minibatch and appends it as 
    additional feature channels.
    
    Attributes:
        group_size (int): Number of samples in each group for computing statistics.
        n_chan (int): Number of channels for the computed statistics (default: 1).
    """

    def __init__(self, group_size, n_chan=1):
        """
        Initializes the Minibatch Standard Deviation Layer.

        Args:
            group_size (int): The number of samples per group when computing statistics.
            n_chan (int): Number of feature channels to append (default is 1).
        """
        super().__init__()
        self.group_size = group_size
        self.n_chan = n_chan

    def forward(self, x):
        """
        Forward pass for the Minibatch Standard Deviation Layer.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W),
                              where N is batch size, C is channels, H and W are spatial dimensions.
        Returns:
            torch.Tensor: Output tensor with additional channels containing computed statistics.
        """

        N, C, H, W = x.shape # Get batch size, channels, height, and width
        G = N # Default group size is the batch size

        # Ensure the group size does not exceed batch size
        if self.group_size is not None:
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
        F = self.n_chan # Number of additional channels for statistics
        c = C // F # Split channels into F groups

        # Reshape tensor: divide batch into G groups and split channels into F groups of size c
        y = x.reshape(G, -1, F, c, H, W)

        # Centering: subtract the mean within each group to normalize
        y = y - y.mean(dim=0)

        # Compute variance across the group
        y = y.square().mean(dim=0)

        # Compute standard deviation with numerical stability (adding epsilon to prevent division by zero)
        y = (y + 1e-8).sqrt()

        # Average across channels, height, and width to get a single summary per group
        y = y.mean(dim=[2, 3, 4])

        # Reshape to match the correct dimensions and repeat across spatial locations
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)

        # Concatenate the computed statistical features as additional channels to the input
        x = torch.cat([x, y], dim=1)

        return x
