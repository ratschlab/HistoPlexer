import os
import numpy as np
import torch

class Monitor:
    """
    Monitors val metrics (validation loss or C-index) and provides a flag indicating
    whether training should stop early based on the metric's performance.
    """
    def __init__(self, save_path, metric='loss', warmup=5, patience=15, stop_epoch=20, verbose=True):
        """
        Args:
            save_path (str): Path to save the model.
            metric (str, optional): Save metric. Defaults to 'loss'.
            warmup (int): Number of initial epochs to allow training without interruption. Default: 5.
            patience (int): How long to wait after the last improvement before triggering early stop. Default: 15.
            stop_epoch (int): The earliest epoch at which stopping can be considered. Default: 20.
            verbose (bool): If True, prints messages about metric improvements. Default: True.
        """
        self.metric = metric
        self.save_path = save_path 
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.metric_min = np.Inf  # For 'loss', lower is better
        self.metric_max = -np.Inf  # For 'c_index', higher is better

    def __call__(self, epoch, metric_dict, model):
        if self.metric == 'loss':
            self.monitor_metric(epoch, metric_dict, model, is_improvement=lambda x, y: x < y)
        elif self.metric == 'c_index':
            self.monitor_metric(epoch, metric_dict, model, is_improvement=lambda x, y: x > y)
        elif self.metric == 'accuracy':
            self.monitor_metric(epoch, metric_dict, model, is_improvement=lambda x, y: x > y)
        elif self.metric == 'weighted_f1':
            self.monitor_metric(epoch, metric_dict, model, is_improvement=lambda x, y: x > y)
        elif self.metric == 'kappa':
            self.monitor_metric(epoch, metric_dict, model, is_improvement=lambda x, y: x > y)
        elif self.metric == 'auc':
            self.monitor_metric(epoch, metric_dict, model, is_improvement=lambda x, y: x > y)
        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

    def monitor_metric(self, epoch, metric_dict, model, is_improvement):
        score = metric_dict[f"val_{self.metric}"]
        if epoch < self.warmup:
            return 
        if self.best_score is None or is_improvement(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
            self.stop = False
        else:
            self.counter += 1
            if self.verbose:
                print(f'Monitoring counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.stop = True

    def save_checkpoint(self, metric, model):
        '''Saves model when the monitored metric improves.'''
        if self.verbose:
            if self.metric == 'loss':
                print(f'Validation loss decreased ({self.metric_min:.6f} --> {metric:.6f}). Saving model ...')
            else:
                print(f'Validation {self.metric} increased ({self.metric_max:.6f} --> {metric:.6f}). Saving model ...')
        if self.metric == 'loss':
            self.metric_min = metric
        else:
            self.metric_max = metric
        torch.save(model.state_dict(), os.path.join(self.save_path, f"model_best_{self.metric}.pt"))