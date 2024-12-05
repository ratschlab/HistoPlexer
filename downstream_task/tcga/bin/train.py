from __future__ import print_function

import os
import time
import json
import random
import argparse
import pandas as pd
import numpy as np
import torch

from src.config.config import Config
from src.trainer import ClassificationTrainer, SurvTrainer #, MultiTaskTrainer
from src.dataset.dataset import PatchSurvDataset, PatchClsDataset #, PatchMultiTaskDataset

def datestr():
    now = time.gmtime()
    return '{:02}{:02}-{:02}{:02}{:02}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

def main(args):
    seed_torch(args.seed)

    # create results directory if necessary
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    results = []
    
    assert args.num_folds > 1 or (args.num_folds <= 1 and args.fold is not None), "If num_folds <= 1, then fold must be specified."

    if args.num_folds > 1:
        fold_range = range(args.num_folds)
    else:
        fold_range = [args.fold]
    
    for fold in fold_range: 
        all_splits = pd.read_csv(os.path.join(args.split_path, f'split_{fold}.csv'))
        
        train_split = all_splits['train'].dropna().reset_index(drop=True)
        val_split = all_splits['val'].dropna().reset_index(drop=True)
        test_split = all_splits['test'].dropna().reset_index(drop=True)
        
        if args.task == "survival":
            train_dataset = PatchSurvDataset(csv_path=args.csv_path, split=train_split, data_path=args.data_path, mode='train', imc_feature_type=args.imc_feature_type)
            val_dataset = PatchSurvDataset(csv_path=args.csv_path, split=val_split, data_path=args.data_path, mode='val', imc_feature_type=args.imc_feature_type)
            test_dataset = PatchSurvDataset(csv_path=args.csv_path, split=test_split, data_path=args.data_path, mode='test', imc_feature_type=args.imc_feature_type)
            
            datasets = (train_dataset, val_dataset, test_dataset)
            trainer = SurvTrainer(args, datasets, fold)
        elif args.task == "classification":
            train_dataset = PatchClsDataset(csv_path=args.csv_path, label_dict=args.label_dict, label_col=args.label_col, split=train_split, data_path=args.data_path, mode='train', imc_feature_type=args.imc_feature_type)
            val_dataset = PatchClsDataset(csv_path=args.csv_path, label_dict=args.label_dict, label_col=args.label_col, split=val_split, data_path=args.data_path, mode='val', imc_feature_type=args.imc_feature_type)  
            test_dataset = PatchClsDataset(csv_path=args.csv_path, label_dict=args.label_dict, label_col=args.label_col, split=test_split, data_path=args.data_path, mode='test', imc_feature_type=args.imc_feature_type)
            
            datasets = (train_dataset, val_dataset, test_dataset)
            trainer = ClassificationTrainer(args, datasets, fold)
        else:
            raise ValueError("Task should be one of 'survival' or 'classification'")
        
        if not args.pretrained_path:
            print(f'Training fold {fold}...')
            trainer.train()
            
        print(f'Validating fold {fold}...')
        test_metrics = trainer.test()
        results.append({"fold": fold, **test_metrics})
        print(f'Fold {fold} done!')
        
    results_df = pd.DataFrame(results)
    results_pivot_df = results_df.melt(id_vars=["fold"], var_name="metric", value_name="value").pivot(index="metric", columns="fold", values="value")
    results_pivot_df['mean'] = results_pivot_df.mean(axis=1)
    results_pivot_df['std'] = results_pivot_df.std(axis=1)
    results_pivot_df.to_csv(os.path.join(args.save_path, 'results.csv'))
        
    print("All folds done!")

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--config_path', type=str, default='./src/config/sample_config.json',
                        help='path to configuration file (default: ./src/config/sample_config.json)')
    args = parser.parse_args()

    # load config file
    with open(args.config_path, 'r') as ifile:
        config = Config(json.load(ifile))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = main(config)
    print("Done!")