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

def prep_pointplot_df(df_corr, keep_lower=True):
    if keep_lower:
        plot_df = df_corr.where(np.tril(np.ones(df_corr.shape)).astype(bool))
    else:
        plot_df = df_corr.where(np.triu(np.ones(df_corr.shape)).astype(bool))
    plot_df.index.name = 'protein1'
    plot_df = plot_df.reset_index(drop=False).melt(id_vars='protein1', var_name='protein2', value_name='corr_value')
    plot_df = plot_df.loc[~plot_df.corr_value.isna(),:]
    plot_df = plot_df.loc[plot_df['protein1']!=plot_df['protein2']]
    plot_df['protein_pair'] = [' : '.join(sorted([x,y])) for x,y in zip(plot_df['protein1'],plot_df['protein2'])]
    plot_df = plot_df.set_index('protein_pair')
    return plot_df

def calculate_scores_and_best_experiments(co_exp_setting, experiments_dict, data_set, gt_scdata_path, which_pairs='all', save_path=None):
    scores = {experiment_type: [] for experiment_type in experiments_dict}

    for experiment_type in experiments_dict.keys(): 
        print(experiment_type)
        for experiment_path in experiments_dict[experiment_type]: 
            print(experiment_path)

            # --- getting correlation matrix for each roi and aggregating ---
            df_corr_agg = dict()
            
            pred_paths = glob.glob(os.path.join(experiment_path, data_set+'_scdata') + '/*.tsv')

            for pred_path in pred_paths:
                roi_name = pred_path.split('/')[-1].split('.')[0]
                pred = pd.read_csv(pred_path, sep='\t', index_col=[0])
                gt = pd.read_csv(os.path.join(gt_scdata_path, roi_name + '.tsv'), sep='\t', index_col=[0])

                pred = pred.loc[:,~pred.columns.isin(['sample_roi', 'X', 'Y', 'radius'])]
                gt = gt.loc[:,pred.columns]
                
                # gt internal correlation
                gt_corr = gt.corr('spearman')
                gt_corr = gt_corr.where(np.tril(np.ones(gt_corr.shape)).astype(bool)).fillna(0)
                # pred internal correlation
                pred_corr = pred.corr('spearman')
                pred_corr = pred_corr.where(np.triu(np.ones(pred_corr.shape)).astype(bool)).fillna(0)
                df_corr = gt_corr+pred_corr
                # make sure the diagonal is one
                for i in range(df_corr.shape[0]):
                    df_corr.iloc[i,i] = 1
                df_corr_agg[roi_name] = df_corr

            # --- getting mean and std across rois ---
            df_corr_mean = pd.DataFrame(np.zeros(df_corr.shape), index=df_corr.index, columns=df_corr.columns)
            for k in df_corr_agg.keys():
                df_corr_mean = df_corr_mean+df_corr_agg[k]
            df_corr_mean = df_corr_mean/len(df_corr_agg.keys())

            df_corr_std = pd.DataFrame(np.zeros(df_corr.shape), index=df_corr.index, columns=df_corr.columns)
            for k in df_corr_agg.keys():
                df_corr_std = df_corr_std+(df_corr_agg[k]-df_corr_mean)**2
            df_corr_std = df_corr_std/len(df_corr_agg.keys()) 

            # --- Merging data for plotting ---
            # Merge mean values
            plot_df_gt = prep_pointplot_df(df_corr_mean)
            plot_df_gt['data_type'] = 'GT'
            plot_df_gt = plot_df_gt.sort_values(by='corr_value', ascending=True)
            plot_df_pred = prep_pointplot_df(df_corr_mean, keep_lower=False)
            plot_df_pred['data_type'] = 'Prediction'
            plot_df_pred = plot_df_pred.loc[plot_df_gt.index,:]
            merged = pd.concat([plot_df_gt, plot_df_pred])
            merged = merged.reset_index(drop=False)

            # Merge std values
            plot_df_gt = prep_pointplot_df(df_corr_std)
            plot_df_gt['data_type'] = 'GT'
            plot_df_gt = plot_df_gt.sort_values(by='corr_value', ascending=True)
            plot_df_pred = prep_pointplot_df(df_corr_std, keep_lower=False)
            plot_df_pred['data_type'] = 'Prediction'
            plot_df_pred = plot_df_pred.loc[plot_df_gt.index,:]
            merged_std = pd.concat([plot_df_gt, plot_df_pred])
            merged_std = merged_std.reset_index(drop=False)
            merged_std.columns = [x.replace('corr_value', 'corr_std') for x in merged_std.columns]

            merged = merged.merge(merged_std, on=['protein_pair', 'data_type'], how='left')
            merged_gt = merged.loc[merged['data_type']=='GT',:]
            merged_pred = merged.loc[merged['data_type']=='Prediction',:]

            if co_exp_setting == 'pruned':
                corr_threshold = 0.15
                merged_gt = merged_gt[(merged_gt['corr_value'] > corr_threshold) | (merged_gt['corr_value'] < -corr_threshold)].reset_index(drop=True)
                merged_pred = merged_pred[merged_pred['protein_pair'].isin(merged_gt['protein_pair'])].reset_index(drop=True)
            else: 
                corr_threshold = 0.0

            if which_pairs == 'positive_corr':
                merged_gt = merged_gt[merged_gt['corr_value'] > corr_threshold].reset_index(drop=True)
                merged_pred = merged_pred[merged_pred['protein_pair'].isin(merged_gt['protein_pair'])].reset_index(drop=True)
            elif which_pairs == 'negative_corr':
                merged_gt = merged_gt[merged_gt['corr_value'] < -corr_threshold].reset_index(drop=True)
                merged_pred = merged_pred[merged_pred['protein_pair'].isin(merged_gt['protein_pair'])].reset_index(drop=True)

            # calculating mse and nsme 
            merged_pred_ = merged_pred.rename(columns={'corr_value': 'corr_value_pred', 'corr_std': 'corr_std_pred'})
            merged_df = merged_gt.merge(merged_pred_, on='protein_pair', how='inner')[['protein_pair', 'corr_value', 'corr_value_pred', 'corr_std', 'corr_std_pred']]
            mse = ((merged_df['corr_value'] - merged_df['corr_value_pred']) ** 2).mean()
            variance = merged_df['corr_value'].var() # variance from reference 

            print("Mean Square Error (MSE):", mse)
            scores[experiment_type].append(mse)
            print('\n')

    print('scores')
    print(scores)
    # get avg and std score for each experiment type
    try: 
        avg_scores = {experiment_type: round(statistics.mean(scores[experiment_type]), 3) for experiment_type in scores}
        std_scores = {experiment_type: round(statistics.stdev(scores[experiment_type]), 3) for experiment_type in scores}
        print('avg_scores')
        print(avg_scores)
        print('std_scores')
        print(std_scores)
    except:
        pass
    
    # finding exp with best score
    best_experiments = {}
    list_csv = []
    for experiment_type in scores: 
        scores_experiment_type = scores[experiment_type]
        # remove nan from list
        scores_experiment_type = [x for x in scores_experiment_type if str(x) != 'nan']
        scores_experiment_type = sorted(scores_experiment_type)[:3]
        # if len 1 then best_Experiment[]
        if len(scores_experiment_type) == 1:
            best_experiments[experiment_type] = experiments_dict[experiment_type][0]
            list_csv.append([experiment_type, scores_experiment_type[0], 0])

        else:         
            mean = round(statistics.mean(scores_experiment_type), 4)
            st_dev = round(statistics.stdev(scores_experiment_type), 4)
            print(experiment_type, mean, st_dev)
            list_csv.append([experiment_type, mean, st_dev])

            smallest_index = scores[experiment_type].index(min(scores[experiment_type]))
            best_exp = experiments_dict[experiment_type][smallest_index]
            best_experiments[experiment_type] = best_exp
        
    print(best_experiments)
    
    columns = ['exp_name', 'MSE_coexp_mean', 'MSE_coexp_std', 'best_exp']
    
    # add best exp in df
    for i in range(len(list_csv)):
        list_csv[i].append(best_experiments[list_csv[i][0]].split('/')[-1])
    
    # Convert list to DataFrame
    df = pd.DataFrame(list_csv, columns=columns)
    df.to_csv(os.path.join(save_path, 'MSE_coexp_'+ which_pairs +'.csv'), index=False)
    print(df)
    
    return scores, best_experiments, df

def plot_coexpression_patterns(co_exp_setting, best_experiments, data_set, gt_scdata_path, adjusted_method_colors, method_markers, which_pairs='all', save_path=None):
    if which_pairs != 'all':
        fig, ax1 = plt.subplots(figsize=(10, 8)) # need smaller size fig if plotting only negative or positive corr pairs
    else:
        fig, ax1 = plt.subplots(figsize=(20, 8))

    for j, method_setting in enumerate(best_experiments.keys()): 
        setting  = method_setting.split('-')[-1]
        print('method_setting: ', method_setting)
        print('setting: ', setting)
        
        experiment_path = best_experiments[method_setting]
        print('experiment_path: '+experiment_path)

        # --- getting correlation matrix for each roi and aggregating ---
        df_corr_agg = dict()            
        pred_paths = glob.glob(os.path.join(experiment_path, data_set+'_scdata') + '/*.tsv')

        for pred_path in pred_paths:
            roi_name = pred_path.split('/')[-1].split('.')[0]
            
            pred = pd.read_csv(pred_path, sep='\t', index_col=[0])
            pred = pred.loc[:,~pred.columns.isin(['sample_roi', 'X', 'Y', 'radius'])]
            gt = pd.read_csv(os.path.join(gt_scdata_path, roi_name + '.tsv'), sep='\t', index_col=[0])
            gt = gt.loc[:,pred.columns]
            
            # gt internal correlation
            gt_corr = gt.corr('spearman')
            gt_corr = gt_corr.where(np.tril(np.ones(gt_corr.shape)).astype(bool)).fillna(0)
            # pred internal correlation
            pred_corr = pred.corr('spearman')
            pred_corr = pred_corr.where(np.triu(np.ones(pred_corr.shape)).astype(bool)).fillna(0)
            df_corr = gt_corr+pred_corr
            # make sure the diagonal is one
            for i in range(df_corr.shape[0]):
                df_corr.iloc[i,i] = 1 
            df_corr_agg[roi_name] = df_corr

        # --- getting mean and std across rois ---
        df_corr_mean = pd.DataFrame(np.zeros(df_corr.shape), index=df_corr.index, columns=df_corr.columns)
        for k in df_corr_agg.keys():
            df_corr_mean = df_corr_mean+df_corr_agg[k]
        df_corr_mean = df_corr_mean/len(df_corr_agg.keys())

        df_corr_std = pd.DataFrame(np.zeros(df_corr.shape), index=df_corr.index, columns=df_corr.columns)
        for k in df_corr_agg.keys():
            df_corr_std = df_corr_std+(df_corr_agg[k]-df_corr_mean)**2
        df_corr_std = df_corr_std/len(df_corr_agg.keys()) 
            
        # --- Merging data for plotting ---
        # Merge mean values
        plot_df_gt = prep_pointplot_df(df_corr_mean)
        plot_df_gt['data_type'] = 'GT'
        plot_df_gt = plot_df_gt.sort_values(by='corr_value', ascending=True)
        plot_df_pred = prep_pointplot_df(df_corr_mean, keep_lower=False)
        plot_df_pred['data_type'] = 'Prediction'
        plot_df_pred = plot_df_pred.loc[plot_df_gt.index,:]
        merged = pd.concat([plot_df_gt, plot_df_pred])
        merged = merged.reset_index(drop=False)

        # Merge std values
        plot_df_gt = prep_pointplot_df(df_corr_std)
        plot_df_gt['data_type'] = 'GT'
        plot_df_gt = plot_df_gt.sort_values(by='corr_value', ascending=True)
        plot_df_pred = prep_pointplot_df(df_corr_std, keep_lower=False)
        plot_df_pred['data_type'] = 'Prediction'
        plot_df_pred = plot_df_pred.loc[plot_df_gt.index,:]
        merged_std = pd.concat([plot_df_gt, plot_df_pred])
        merged_std = merged_std.reset_index(drop=False)
        merged_std.columns = [x.replace('corr_value', 'corr_std') for x in merged_std.columns]

        merged = merged.merge(merged_std, on=['protein_pair', 'data_type'], how='left')
        merged_gt = merged.loc[merged['data_type']=='GT',:]
        merged_pred = merged.loc[merged['data_type']=='Prediction',:]

        # pruning to choose those protein pair where correlation is strong 
        if co_exp_setting == 'pruned':
            corr_threshold = 0.15
            merged_gt = merged_gt[(merged_gt['corr_value'] > corr_threshold) | (merged_gt['corr_value'] < -corr_threshold)].reset_index(drop=True)
            merged_pred = merged_pred[merged_pred['protein_pair'].isin(merged_gt['protein_pair'])].reset_index(drop=True)
        else: 
            corr_threshold = 0.0

        if which_pairs == 'positive_corr':
            merged_gt = merged_gt[merged_gt['corr_value'] > corr_threshold].reset_index(drop=True)
            merged_pred = merged_pred[merged_pred['protein_pair'].isin(merged_gt['protein_pair'])].reset_index(drop=True)
        elif which_pairs == 'negative_corr':
            merged_gt = merged_gt[merged_gt['corr_value'] < -corr_threshold].reset_index(drop=True)
            merged_pred = merged_pred[merged_pred['protein_pair'].isin(merged_gt['protein_pair'])].reset_index(drop=True)

        # --- plotting merged, x axis marker pairs, y axis correlation ---
        mean_column = 'corr_value' 
        std_column = 'corr_std'
        color = adjusted_method_colors[method_setting]
        marker = method_markers.get(setting, 'o')
        linestyle = ':' if setting == 'SP' else '-'

        if j==0: # plot GT only once
            sns.lineplot(x='protein_pair', y=mean_column, data=merged_gt,
                        marker=marker, markersize=8, linestyle=linestyle,
                        color=adjusted_method_colors['GT'], label="GT-GT", ax=ax1)
            ax1.fill_between(merged_gt['protein_pair'],
                            merged_gt[mean_column] - 0.5*merged_gt[std_column],
                            merged_gt[mean_column] + 0.5*merged_gt[std_column],
                            color=adjusted_method_colors['GT'], alpha=0.2)
            ax1.axhline(y=0.0, color='gray', linestyle='--', linewidth=2)        
            ax1.set_xlabel('Protein_pair',  fontsize=25)
            ax1.set_ylabel('Spearman Correlation', fontsize=25)
            ax1.tick_params(axis='x', labelrotation=45, labelsize=12)
            ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
            ax1.legend(title='Method/Setting')        

        # Plot mean SCC values with points and possibly connecting lines
        sns.lineplot(x='protein_pair', y=mean_column, data=merged_pred,
                    marker=marker, markersize=8, linestyle=linestyle,
                    color=color, label=f"{method_setting}", ax=ax1)

        # Add shaded area for standard deviation (using 0.5*std)
        ax1.fill_between(merged_pred['protein_pair'],
                        merged_pred[mean_column] - 0.5*merged_pred[std_column],
                        merged_pred[mean_column] + 0.5*merged_pred[std_column],
                        color=color, alpha=0.2)

    fig.savefig(os.path.join(save_path, 'coexpression_patterns_'+ which_pairs  + '.png'), bbox_inches='tight', dpi=400)
    fig.savefig(os.path.join(save_path, 'coexpression_patterns_'+ which_pairs  + '.pdf'), bbox_inches='tight', dpi=400)
    fig.savefig(os.path.join(save_path, 'coexpression_patterns_'+ which_pairs  + '.svg'), bbox_inches='tight', dpi=400)

    plt.show()