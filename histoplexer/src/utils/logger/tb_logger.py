import importlib
import time
import torch
import numpy as np


class TBLogger:
    """A utility class for logging data to TensorBoard.

    Attributes:
        log_dir (str): The directory where the TensorBoard logs will be stored.
        tb_logger (SummaryWriter): An instance of the TensorBoard SummaryWriter
            class, which provides methods for writing data to TensorBoard.
    """
    
    def __init__(self, log_dir=None):
        """Initializes the TBLogger class.

        Args:
            log_dir (str, optional): The directory where the TensorBoard logs 
            will be stored. If None, a timestamped subdirectory will be created 
            in the current working directory. Defaults to None.
        """
        self.log_dir = log_dir
        
        # Import the TensorBoard module from PyTorch and create a SummaryWriter instance
        tb_module = importlib.import_module("torch.utils.tensorboard")
        self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)

    def flush(self):
        """Flushes the SummaryWriter instance."""
        self.tb_logger.flush()
        
    def close(self):
        """Closes the SummaryWriter instance, effectively ending the logging session."""
        self.tb_logger.close()

    def run(self, func_name, *args, **kwargs):
        """Runs the specified function of the SummaryWriter instance with the provided arguments.

        Args:
            func_name (str): The name of the function to run.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The return value of the function that was run, or None if the function does not exist.
        """
        # Check if the function name is "log_scalars"
        if func_name == "log_scalars":
            return self.tb_log_scalars(*args, **kwargs)
        else:
            # Attempt to get the function from the SummaryWriter instance and run it
            tb_log_func = getattr(self.tb_logger, func_name)
            return tb_log_func(*args, **kwargs)

        return None

    def tb_log_scalars(self, metric_dict, step):
        """Logs multiple scalar values to TensorBoard.

        Args:
            metric_dict (dict): A dictionary where the keys are the names of the metrics
                and the values are the corresponding scalar values to be logged.
            step (int): The current step or iteration in the process being logged.
        """
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)