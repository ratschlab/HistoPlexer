import torch
import torch.nn as nn
from torch.nn import init
# from torchsummary import summary

class BaseModel(nn.Module):
    """Base class for the generator."""

    def __init__(self, device):
        """Initializes the BaseModel class."""
        super(BaseModel, self).__init__()
        self.device = device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_weights(self, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        
        Args:
            init_type (str): The type of initialization; 'normal', 'xavier', 'kaiming', or 'orthogonal'.
            init_gain (float): Scaling factor for normal, xavier, and orthogonal.
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def save_model(self, path):
        """Save the model's state dictionary.
        
        Args:
            checkpoint_path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model parameters from a saved state.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint.
        """
        self.load_state_dict(torch.load(path, map_location=str(self.device)))

    # def summary(self, input_size):
    #     """Print a summary of the model.
        
    #     Args:
    #         input_size (tuple): The size of the input tensor (C, H, W).
    #     """
    #     summary(self, input_size)

    def requires_grad(self, value: bool):
        """Sets the 'requires_grad' value for all parameters in the model.
        
        Args:
            value (bool): Whether the gradients should be calculated for the parameters.
        """
        for param in self.parameters():
            param.requires_grad = value