import os
import h5py
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ----------- classification dataset ----------- #

class BaseClsDataset(Dataset):
    def __init__(self, csv_path, split, label_dict, label_col='label', ignore=[]):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            split (pd.DataFrame): Train/val/test split. 
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int. 
            label_col (str, optional): Label column. Defaults to 'type'.
            ignore (list, optional): Ignored labels. Defaults to [].
        """        
        slide_data = pd.read_csv(csv_path)
        slide_data = self._df_prep(slide_data, label_dict, ignore, label_col)
        assert len(split) > 0, "Split should not be empty!"
        mask = slide_data['case_id'].isin(split.tolist())
        self.slide_data = slide_data[mask].reset_index(drop=True)
        self.n_cls = len(set(label_dict.values()))
        self.slide_cls_ids = self._cls_ids_prep()
        self._print_info()

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        return None

    def _print_info(self):
        print("Number of classes: {}".format(self.n_cls))
        print("Slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))

    def _cls_ids_prep(self):
        slide_cls_ids = [[] for i in range(self.n_cls)]
        for i in range(self.n_cls):
            slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        return slide_cls_ids

    def get_label(self, ids):
        return self.slide_data['label'][ids]

    @staticmethod
    def _df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data
    

class PatchClsDataset(BaseClsDataset):
    def __init__(self, data_path, mode='train', imc_feature_type='resnet_tumor', **kwargs):
        """
        Args:
            data_path (str): Path to the data.
            mode (str): Operating mode, 'train' or 'test'.
            imc_feature_type (str): Type of IMC features to read.
        """        
        super(PatchClsDataset, self).__init__(**kwargs)
        self.data_path = data_path
        self.mode = mode
        self.imc_feature_type = 'features_imc_' + imc_feature_type
        assert self.imc_feature_type in ['features_imc_trans', 'features_imc_dis', 'features_imc_resnet', 'features_imc_resnet_tumor'], "Invalid IMC feature type!"
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        slide_id = self.slide_data['case_id'][idx]
        label = self.slide_data['label'][idx]
        
        with h5py.File(os.path.join(self.data_path, '{}.h5'.format(slide_id)), 'r') as hdf5_file:
            if self.mode == 'train':
                # Select a random group for training
                group_name = random.choice(list(hdf5_file.keys()))
                group = hdf5_file[group_name]
            else:
                # Concatenate all groups for testing
                features_list_he = []
                features_list_imc = []
                for group_name in hdf5_file.keys():
                    group = hdf5_file[group_name]
                    features_list_he.append(group['features_he'][:])
                    features_list_imc.append(group[self.imc_feature_type][:])
                feats_he = np.concatenate(features_list_he, axis=0)
                feats_imc = np.concatenate(features_list_imc, axis=0)
                feats_he = torch.from_numpy(feats_he)
                feats_imc = torch.from_numpy(feats_imc)
                feats = (feats_he, feats_imc)
                coords = np.concatenate([group['coords'][:] for group_name in hdf5_file.keys()], axis=0)
                return (feats, label, coords, slide_id)
            
            feats_he = torch.from_numpy(group['features_he'][:])
            feats_imc = torch.from_numpy(group[self.imc_feature_type][:])
            feats = (feats_he, feats_imc)
            coords = group['coords'][:]  # Assuming coords are same across groups
        
        return (feats, label, coords, slide_id)

# ----------- survival dataset ----------- #

class BaseSurvDataset(Dataset):
    def __init__(self, csv_path, split, ignore=[]):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            split (pd.DataFrame): Train/val/test split. 
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int. 
            label_col (str, optional): Label column. Defaults to 'DSS.time'.
            cen_col (str, optional): Cencorship column. Defaults to 'DSS'.
            ignore (list, optional): Ignored labels. Defaults to [].
        """        
        slide_data = pd.read_csv(csv_path)
        assert len(split) > 0, "Split should not be empty!"
        mask = slide_data['slide_id'].isin(split.tolist())
        self.slide_data = slide_data[mask].reset_index(drop=True)
        self.n_cls = self.slide_data['pat_label'].nunique()
        self.slide_cls_ids = self._cls_ids_prep()
        # self._print_info()

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        return None

    def _print_info(self):
        print("Number of classes: {}".format(self.n_cls))
        print("Slide-level counts: ", '\n', self.slide_data['pat_label'].value_counts(sort=False))

    def _cls_ids_prep(self):
        slide_cls_ids = [[] for i in range(self.n_cls)]
        for i in range(self.n_cls):
            slide_cls_ids[i] = np.where(self.slide_data['pat_label'] == i)[0] # using patient level label
        return slide_cls_ids

    def get_label(self, ids):
        return self.slide_data['pat_label'][ids]
    
class PatchSurvDataset(BaseSurvDataset):
    def __init__(self, data_path, mode='train', imc_feature_type='resnet_tumor', **kwargs):
        """
        Args:
            data_path (str): Path to the data.
            mode (str): Operating mode, 'train' or 'test'.
            imc_feature_type (str): Type of IMC features to read.
        """
        super(PatchSurvDataset, self).__init__(**kwargs)
        self.data_path = data_path
        self.mode = mode
        self.imc_feature_type = 'features_imc_' + imc_feature_type
        assert self.imc_feature_type in ['features_imc_trans', 'features_imc_dis', 'features_imc_resnet', 'features_imc_resnet_tumor'], "Invalid IMC feature type!"
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data['survival'][idx]
        c = self.slide_data['censorship'][idx]
        
        with h5py.File(os.path.join(self.data_path, '{}.h5'.format(slide_id)), 'r') as hdf5_file:
            if self.mode == 'train':
                # Select a random group for training
                group_name = random.choice(list(hdf5_file.keys()))
                group = hdf5_file[group_name]
            else:
                # Concatenate all groups for testing
                features_list_he = []
                features_list_imc = []
                for group_name in hdf5_file.keys():
                    group = hdf5_file[group_name]
                    features_list_he.append(group['features_he'][:])
                    features_list_imc.append(group[self.imc_feature_type][:])
                feats_he = np.concatenate(features_list_he, axis=0)
                feats_imc = np.concatenate(features_list_imc, axis=0)
                feats_he = torch.from_numpy(feats_he)
                feats_imc = torch.from_numpy(feats_imc)
                feats = (feats_he, feats_imc)
                coords = np.concatenate([group['coords'][:] for group_name in hdf5_file.keys()], axis=0)
                return (feats, torch.tensor(label, dtype=torch.long), event_time, torch.tensor(c, dtype=torch.float), coords, slide_id)
            
            feats_he = torch.from_numpy(group['features_he'][:])
            feats_imc = torch.from_numpy(group[self.imc_feature_type][:])
            feats = (feats_he, feats_imc)
            coords = group['coords'][:]  # Assuming coords are same across groups
            
        return (feats, torch.tensor(label, dtype=torch.long), event_time, torch.tensor(c, dtype=torch.float), coords, slide_id)