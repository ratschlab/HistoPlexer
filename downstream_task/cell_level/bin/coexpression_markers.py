import os
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
import json
from sklearn.metrics import mean_squared_error

import argparse
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import sys 
import statistics

from src.coexpression_utils import prep_pointplot_df, calculate_scores_and_best_experiments, plot_coexpression_patterns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Co-expression patterns analysis')
    parser.add_argument('--base_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results', help='Path to the experiments directory')
    parser.add_argument('--data_set', type=str, default='test', help='Dataset to use')
    parser.add_argument('--level', type=int, default=2, help='Resolution level')
    parser.add_argument('--co_exp_setting', type=str, default='pruned', help='pruned or all marker pairs')
    parser.add_argument('--gt_scdata_path', type=str, default='/raid/sonali/project_mvs/data/tupro/imc_updated/agg_masked_data-raw_clip99_arc_otsu3_std_minmax_split3-r5', 
                        help='Path to ground truth avg expression per cell')
    parser.add_argument('--which_pairs', type=str, default='all', help='which protein pairs to plot: all, or pairs with positive_corr or negative_corr')
    parser.add_argument('--save_path', type=str, default='/raid/sonali/project_mvs/nmi_results/final_results/coexpression_analysis', help='Path to save the results')
    parser.add_argument('--plot', action='store_true', help='Plot the results when passed')
        
    args = parser.parse_args()
    base_path = args.base_path
    co_exp_setting = args.co_exp_setting
    gt_scdata_path = args.gt_scdata_path
    data_set = args.data_set
    os.makedirs(args.save_path, exist_ok=True)

    # keywords to find all experiments for different baselines
    keywords = {
                'ours-FM-MP': 'tupro-patches_ours-FM_channels-all_seed-',
                'ours-FM-SP': 'tupro-patches_ours-FM_channels-all-pseudoplex_seed',
                'ours-FM-uni2-MP': 'tupro-patches_ours-FM-uni2_channels-all_seed-', 
                'ours-FM-virchow2-MP': 'tupro-patches_ours-FM-virchow2_channels-all_seed-',
                'Ours-MP': 'tupro_ours_channels-all_seed-',
                'Ours-SP': 'tupro_ours_channels-all-pseudoplex_seed',
                'PyramidP2P-MP': 'tupro_pyramidp2p_channels-all_seed-',
                'PyramidP2P-SP': 'tupro_pyramidp2p_channels-all-pseudoplex_seed',
                'Pix2pix-MP': 'tupro_pix2pix_channels-all_seed-',
                'Pix2pix-SP': 'tupro_pix2pix_channels-all-pseudoplex_seed',
                'cyclegan-MP': 'tupro_cyclegan_channels-all_seed', 
                'cyclegan-SP': 'tupro_cyclegan_channels-all-pseudoplex_seed'
                }
    
    experiments_dict = {}

    for experiment_type in keywords.keys():     
        exps = glob.glob(os.path.join(base_path + '/*/*' + keywords[experiment_type] + '*'))
        experiments_dict[experiment_type] = exps
        
    print('experiments_dict')
    print(experiments_dict)
        
    # get mse scores
    scores, best_experiments, df = calculate_scores_and_best_experiments(co_exp_setting, experiments_dict, data_set, gt_scdata_path, save_path=args.save_path)
    
    print('best_experiments: ')
    print(best_experiments)
    
    if args.plot:
        # pass colors for each experiment as needed
        adjusted_method_colors = {'Ours-MP': '#1f77b4', 'Ours-SP': '#6baed6',
                                    'Pix2pix-MP': '#bc4b51',  'Pix2pix-SP': '#f1948a', 
                                    'PyramidP2P-MP': '#2ca02c', 'PyramidP2P-SP': '#98df8a', 
                                    'GT': '#7f7f7f'}

        method_markers = {'MP': 'o', 'SP': '^'}
        line_styles = {'MP': '-', 'SP': '--'}
        plot_coexpression_patterns(co_exp_setting, best_experiments, data_set, gt_scdata_path, adjusted_method_colors, method_markers, which_pairs='all', save_path=args.save_path)