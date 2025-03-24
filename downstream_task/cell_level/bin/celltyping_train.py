import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import joblib
import json
import argparse

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix as cfm

from src.celltyping_utils import get_manual_aggregation, plot_feature_imp, plot_cfm

# python -m bin.celltyping_train
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RF clasifier using train set.')
    parser.add_argument('--save_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/final_results/cell_typing', help='Path to save random forest model and results')
    parser.add_argument('--gt_scdata_merged', type=str, required=False, default='/raid/sonali/project_mvs/data/tupro/imc_updated/agg_masked_data-raw_clip99_arc_otsu3_std_minmax_split3-r5.tsv', 
                        help='Path to ground truth avg expression per cell merged data over samples ')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Random seed')
    parser.add_argument('--max_depth', type=int, required=False, default=30, help='Max tree depth (if None, then tree grown until purity, see sklearn docs')
    parser.add_argument('--n_estimators', type=int, required=False, default=100, help='Number of trees to use')
    parser.add_argument('--markers_list', type=list, default=["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"], help='List of markers to use')
    parser.add_argument('--gt_celltypes', type=str, required=False, default='/raid/sonali/project_mvs/data/tupro/imc_updated/coldata.tsv', help='metadata per cell segmented from IMC using CellProfiler (includes coordinates X,Y and cell-type)')    
    parser.add_argument('--cell_types', type=str, required=False, default='all', help='helps in merging cell types')
    parser.add_argument('--split_csv', type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/split3.csv', help='Selected CV split, if None then the splitting used for the report is used')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    markers_list = args.markers_list
    save_fname = '-'.join(['rf', 'cell_type',
                           args.gt_scdata_merged.split('/')[-1].split('.')[0].replace('agg_masked_data-',''),
                           'ntrees'+str(args.n_estimators), 'maxdepth'+str(args.max_depth), 'ct_merged']) # TODO remove ct_merged if not needed 
    
    # load train rois from split csv -- used for training RF
    splits = pd.read_csv(args.split_csv)
    train_aligns = list(set(splits['train'].dropna()))
    valid_aligns = list(set(splits['valid'].dropna()))
    sets = [train_aligns, valid_aligns]
    set_names = ['train', 'valid']
    
    # Get protein expression data
    df_all = pd.read_csv(args.gt_scdata_merged, sep='\t', index_col=[0])
    # subset to ROIs of interest
    df_all = df_all.loc[df_all.sample_roi.isin(train_aligns+valid_aligns),:]
    # Make sure there are no Nan's
    df_all = df_all.loc[~df_all.iloc[:,0].isna(),:]
    df_all['set'] = ['train' if x in train_aligns else 'valid' for x in df_all.sample_roi.to_list()]
    print(df_all.head())
    
    # --- Get cell-type labels and merge with protein data --- 
    # all cell types: 'myeloid', 'Tcells.CD8', 'Bcells', 'tumor', 'vessels', 'Tcells.CD4', 'other'
    cts = pd.read_csv(args.gt_celltypes, sep='\t', index_col=[0])
    print(cts['cell_type'].unique())

    # merge some cell-types to simplify
    if args.cell_types == 'tumor_CD8':
        rf_cell_types = ['tumor', 'Tcells.CD8']
    elif args.cell_types == 'tumor_CD8_CD4':
        rf_cell_types = ['tumor', 'Tcells.CD8', 'Tcells.CD4']
    elif args.cell_types == 'tumor_CD8_CD4_CD20':
        rf_cell_types = ['tumor', 'Tcells.CD8', 'Tcells.CD4', 'Bcells']
    elif args.cell_types == 'all':
        rf_cell_types = ['myeloid', 'Tcells.CD8', 'Bcells', 'tumor', 'vessels', 'Tcells.CD4']
    cts['cell_type'] = cts['cell_type'].apply(lambda x: x if x in rf_cell_types else 'other')

    df_all = df_all.merge(cts.loc[:,['cell_type']], left_index=True, right_index=True, how='inner')
    df_all = df_all.sort_values(by=['set','cell_type'])
    CELL_TYPES = sorted(df_all['cell_type'].unique())
    print('Loaded data')
    
    # One-hot encode cell-type labels
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # subset to protein set of interest and split into x and y
    train_x = df_all.loc[df_all.set=='train', markers_list]
    train_y = ohe.fit_transform(df_all.loc[df_all.set=='train', 'cell_type'].to_numpy().reshape(-1, 1))
    train_y = pd.DataFrame(train_y, columns=CELL_TYPES).astype(int)

    valid_x = df_all.loc[df_all.set=='valid', markers_list]
    valid_y = ohe.transform(df_all.loc[df_all.set=='valid', 'cell_type'].to_numpy().reshape(-1, 1))
    valid_y = pd.DataFrame(valid_y, columns=CELL_TYPES).astype(int)
    
    # Make sure we have a 'one' in every y output vector
    assert np.sum(train_y.sum()) == train_y.shape[0] and np.sum(valid_y.sum()) == valid_y.shape[0]
    print('One-hot encoded cell-type labels')
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=4, class_weight='balanced', max_depth=args.max_depth, oob_score=True)
    rf.fit(train_x, train_y)
    print('RF trained')
    pd.DataFrame({'oob':rf.oob_score_}, index=[0]).to_csv(os.path.join(save_path, save_fname+'-oob.txt'), sep='\t', index=False)
    
    # Feature importance plot
    plot_feature_imp(markers_list, rf.feature_importances_)
    plt.savefig(os.path.join(save_path, save_fname+'-fi.png'))
    
    df_all['RF_preds'] = 'other'
    for set_name in set_names:
        preds = get_manual_aggregation(rf, df_all.loc[df_all.set==set_name, markers_list], CELL_TYPES)
        gt = ohe.fit_transform(df_all.loc[df_all.set==set_name, 'cell_type'].to_numpy().reshape(-1, 1))
        f1 = f1_score(gt, preds, average="weighted")
        print(set_name.upper()+" F1: ", f1)
        
        df_all.loc[df_all.set==set_name,'RF_preds'] = ohe.inverse_transform(preds)
        class_report = classification_report(df_all.loc[df_all.set==set_name,'cell_type'],df_all.loc[df_all.set==set_name,'RF_preds'],output_dict=True)
        pd.DataFrame(class_report).transpose().to_csv(os.path.join(save_path, save_fname+'-report_'+set_name+'.tsv'), sep='\t')

        print(class_report)
        # cfm returns a matrix with GT as index and pred as columns
        confusion_matrix = cfm(df_all.loc[df_all.set==set_name,'cell_type'],df_all.loc[df_all.set==set_name,'RF_preds'])
        confusion_matrix = pd.DataFrame(confusion_matrix, index=CELL_TYPES, columns=CELL_TYPES)
        confusion_matrix.to_csv(os.path.join(save_path, save_fname+'-cfm_'+set_name+'.tsv'), sep='\t')
        print(confusion_matrix)
        plot_cfm(df_all.loc[df_all.set==set_name,'cell_type'], df_all.loc[df_all.set==set_name,'RF_preds'], CELL_TYPES)
        plt.savefig(os.path.join(save_path, save_fname+'-cfm_'+set_name+'.png'))
        
    joblib.dump(rf, os.path.join(save_path, save_fname+'.joblib'))     