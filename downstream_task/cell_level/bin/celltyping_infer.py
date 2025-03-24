import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import joblib
import json
import argparse
import sklearn
import warnings
warnings.simplefilter("ignore")
from sklearn.ensemble import RandomForestClassifier

from src.celltyping_utils import get_manual_aggregation, plot_feature_imp, plot_cfm

# python -m bin.celltyping_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply RF on tabular predicted pseudo-single-cell data.')
    parser.add_argument('--rf_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/final_results/cell_typing/rf-cell_type-raw_clip99_arc_otsu3_std_minmax_split3-r5-ntrees100-maxdepth30-ct_merged.joblib', help='Path fo the trained RF joblib file')
    parser.add_argument('--pred_scdata', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/test_scdata', 
                        help='Path to predicted avg expression per pseudo-cell')
    parser.add_argument('--save_path', type=str, required=False, default=None, help='Path to save predictions')
    parser.add_argument('--data_set', type=str, required=False, default="test", help='Which set from split to use for inference {test, valid, train}')
    parser.add_argument('--split_csv', type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/split3_train-test.csv', help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--cell_types', type=str, required=False, default='all', help='helps in merging cell types')
    parser.add_argument('--markers_list', type=list, default=["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"], help='List of markers to use')

    args = parser.parse_args()
    
    if args.save_path is None:
        save_path = os.path.join(os.path.dirname(args.pred_scdata), args.data_set + '_cell_types')
    else:
        save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    # get data_set rois 
    splits = pd.read_csv(args.split_csv)
    sample_rois = list(set(splits[args.data_set].dropna()))
       
    # Load trained RF
    rf = joblib.load(args.rf_path)
    sel_cols = args.markers_list
    sel_cols.extend(['sample_roi','X','Y', 'radius'])

    # cell types
    if args.cell_types == 'tumor_CD8':
        rf_cell_types = ['tumor', 'Tcells.CD8', 'other']
    elif args.cell_types == 'tumor_CD8_CD4':
        rf_cell_types = ['tumor', 'Tcells.CD8', 'Tcells.CD4', 'other']
    elif args.cell_types == 'tumor_CD8_CD4_CD20':
        rf_cell_types = ['tumor', 'Tcells.CD8', 'Tcells.CD4', 'Bcells', 'other']
    elif args.cell_types == 'all':
        rf_cell_types = ['myeloid', 'Tcells.CD8', 'Bcells', 'tumor', 'vessels', 'Tcells.CD4', 'other']
    rf_cell_types = sorted(rf_cell_types)
    
    for s_roi in sample_rois:
        # Get protein expression data
        df_roi = pd.read_csv(os.path.join(args.pred_scdata, s_roi+'.tsv'), sep='\t', index_col=[0])
        # Remove any rows with NaN and get proteins used for training RF
        df_roi = df_roi.loc[~(df_roi.isna().sum(axis=1)>0),sel_cols]
        # apply RF
        preds = get_manual_aggregation(rf, df_roi.loc[:,~df_roi.columns.isin(['sample_roi','X','Y','radius'])], rf_cell_types)
        preds = pd.DataFrame(preds, index=df_roi.index, columns=rf_cell_types)
        # extract cell-type label per cell
        preds = preds.apply(lambda x: [rf_cell_types[i] for i,y in enumerate(x) if np.isclose(y,1)][0], axis=1).to_frame('pred_cell_type')
        # merge with coords and save
        df_roi = df_roi.loc[:,['X','Y']].merge(preds, left_index=True, right_index=True, how='left')
        df_roi.to_csv(os.path.join(save_path, s_roi+'.tsv'), sep='\t')