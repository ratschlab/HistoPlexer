from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from torch import nn


class BaseTrainer(ABC):
    def __init__(self, args):
        """
        Initializes the base trainer class. Sets up the device and configs.
        """
        self.device = torch.device(args.device)
        self.config = args

    @abstractmethod
    def _G_step(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Performs a single training step for the generator.

        Must be implemented by subclasses.

        Args:
            src (torch.Tensor): The source real images.
            tgt (torch.Tensor): The target real images.
        """
        pass
    
    @abstractmethod
    def _D_step(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Performs a single training step for the discriminator.

        Must be implemented by subclasses.

        Args:
            src (torch.Tensor): The source real images.
            tgt (torch.Tensor): The target real images.
        """
        pass
    
    @abstractmethod
    def train(self):
        """
        Runs the training process.

        Must be implemented by subclasses.
        """
        pass