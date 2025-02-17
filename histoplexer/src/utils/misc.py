import os
import random

import numpy as np
import torch


def seed_everything(seed=7, device="cpu"):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_n_trainable_params(model, last_stage=None):
    if last_stage is not None:
        named_params = filter(
            lambda p: p[1].requires_grad and "stage" in p[0] and int(p[0].split("stage")[1][0]) <= last_stage,
            model.named_parameters(),
        )
        return sum([np.prod(p.size()) for n, p in named_params])
    else:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])