import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

from src.coexpression_tsne_utils import *

def main(args):
    save_path = args.save_path
    scale_to01 = args.scale_to01
    scale_to01_forplot = args.scale_to01_forplot
    n_sample = args.n_sample 

    os.makedirs(args.save_path, exist_ok=True)
    np.random.seed(args.seed)
    
    # path to tsne dataframe 
    tsne_df_path = os.path.join(save_path, 'tsne_results_df.csv')
    if os.path.exists(tsne_df_path):
        tsne_results_df = pd.read_csv(tsne_df_path)
    else:    
        pred_all, gt_all = load_data(args.gt_scdata_path, args.pred_scdata_path)
        
        # if scaling desired dbefore embedding (not recommended)
        if scale_to01:
            pred_scaled = scale_df(pred_all.loc[:, ~pred_all.columns.isin(['sample_roi'])])
            pred_all = pd.concat([pred_scaled, pred_all.loc[:, ['sample_roi']]], axis=1)
            gt_scaled = scale_df(gt_all.loc[:, ~gt_all.columns.isin(['sample_roi'])])
            gt_all = pd.concat([gt_scaled, gt_all.loc[:, ['sample_roi']]], axis=1)

        # subsample n_sample cells per ROI for tSNE and merge GT and pred in one dataframe
        merged = pd.concat([pred_all.groupby('sample_roi').sample(n_sample), gt_all.groupby('sample_roi').sample(n_sample)])
        merged['type'] = ['Pred'] * (n_sample * pred_all.sample_roi.nunique()) + ['GT'] * (n_sample * gt_all.sample_roi.nunique())
        merged = merged.reset_index(drop=True)

        # tsne embedding of the data (jointly between GT and pred) https://distill.pub/2016/misread-tsne/
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
        tsne_results = tsne.fit_transform(merged.loc[:, ~merged.columns.isin(['sample_roi', 'type'])])

        # -- plotting tsne across proteins 
        # Scale data to ensure [0,1] range per protein (for plotting) 
        # The typical scaling to 0.99% and the top 1% set to 1 (e.g., https://www.sciencedirect.com/science/article/pii/S0092867419302673)
        tsne_results_df = pd.DataFrame({'tSNE1': tsne_results[:, 0], 'tSNE2': tsne_results[:, 1]})
        if scale_to01_forplot:
            pred_scaled = scale_df(merged.loc[merged['type'] == 'Pred', ~merged.columns.isin(['sample_roi', 'type'])].reset_index(drop=True))
            gt_scaled = scale_df(merged.loc[merged['type'] == 'GT', ~merged.columns.isin(['sample_roi', 'type'])].reset_index(drop=True))
            merged = pd.concat([pd.concat([pred_scaled, gt_scaled], axis=0).reset_index(drop=True), merged.loc[:, ['sample_roi', 'type']].reset_index(drop=True)], axis=1).reset_index(drop=True)

        tsne_results_df = pd.concat([tsne_results_df, merged], axis=1)
        tsne_results_df['sample_id'] = [x.split('_')[0] for x in tsne_results_df['sample_roi']]
        tsne_results_df.to_csv(tsne_df_path, index=False)
        
    # plotting tsne 
    protein_sets = {
        'immune': ['CD16', 'CD20', 'CD3', 'CD8a'],
        'tumor': ['gp100', 'MelanA', 'S100', 'SOX10'],
        'other': ['CD31', 'HLA-ABC', 'HLA-DR'],
    } 
    
    plot_tsne_results(tsne_results_df, save_path, protein_sets)

    
# python -m bin.coexpression_tsne --scale_to01_forplot
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE plots for co-expression/localisation patterns of markers.")
    parser.add_argument('--gt_scdata_path', type=str, default='/raid/sonali/project_mvs/data/tupro/imc_updated/agg_masked_data-raw_clip99_arc_otsu3_std_minmax_split3-r5', 
                        help='Path to ground truth avg expression per cell')
    parser.add_argument('--pred_scdata_path', type=str, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/test_scdata', 
                        help='Path to predicted avg expression per cell')
    parser.add_argument('--data_set', type=str, default='test', help='Dataset to use')
    parser.add_argument('--save_path', type=str, default='/raid/sonali/project_mvs/nmi_results/final_results/co-expression_patterns-tsne', help='Path to save the t-sne results')
    parser.add_argument('--seed', type=int, default=456, help='Random seed for reproducibility.')
    parser.add_argument('--scale_to01', action='store_true', help='Scale to [0,1] before tSNE embedding.') 
    parser.add_argument('--scale_to01_forplot', action='store_true', help='Scale to [0,1] for plotting only.')
    parser.add_argument('--n_sample', type=int, default=1000, help='Number of cells to sample from each ROI for t-sne plotting.')

    args = parser.parse_args()
    main(args)