import os
from datetime import datetime
import argparse
import json
import torch
import torchvision.transforms.functional as ttf

from histoplexer.utils.dataset_utils import clean_train_val_test_sep_for_manual_aligns_ordered_by_quality
from histoplexer.utils.constants import *


def get_aligns(project_path, cv_split=None, protein_set='full', aligns_set='train'):
    ''' Return a list of aligns for train and valid (dictionary per sample_ROI)
    cv_split: name of the cv split {split1, ..., split5}; if None then report set is used
    good_areas_only: whether to suset to good areas only
    protein_set: name of the protein set (special tratment of 'reduced')
    aligns_set: set of aligns to be returned {train, valid, test}
    Note: needs refactoring; requires constants.py to be sources
    '''
    aligns_order = dict({'train':0, 'valid':1, 'test':2})
    if cv_split is None:
        aligns = clean_train_val_test_sep_for_manual_aligns_ordered_by_quality()
        aligns = aligns[aligns_order[aligns_set]]
     
    else:
        kfold_splits = json.load(open(os.path.join(project_path, CV_SPLIT_DICT_PATH)))
        assert cv_split in [x for x in kfold_splits.keys()], 'Selected cv split not in the kfold_splits'
        aligns = kfold_splits[cv_split][aligns_set]
        if protein_set=='reduced':
            aligns = [ar for ar in aligns if not ar["celltype_relevant_proteins_missing"]]

    return aligns

def resize_tensor(tensor, output_size):
    ''' Resize tensor to given output_size(s)
    tensor: tensor object
    output_size: scalar or list of output_sizes (resizing sequentially)
    returns resized tensors
    '''
    if isinstance(output_size, list) == False:
        output_size = [output_size]
    tensors_resized = []
    for osz in output_size:
        tensor = ttf.resize(tensor, osz)
        tensors_resized.append(tensor)

    return [x for x in tensors_resized]

# from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def get_datetime():
    datetime_full_iso = datetime.today().isoformat().replace(":", "")
    datetime_full_iso = datetime_full_iso[:datetime_full_iso.rfind(".")]
    return datetime_full_iso

def str2bool(v):
    ''' Function alloowing for more flexible use of boolean arguments; 
    from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')