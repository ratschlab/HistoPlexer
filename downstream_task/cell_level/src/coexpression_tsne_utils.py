import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def clip_to_q(vec, quants):
    vec[vec > quants] = quants[vec > quants]
    return vec


def scale_df(df):
    df_quants = df.quantile(0.99)
    df_scaled = df.apply(lambda x: clip_to_q(x, df_quants), axis=1)
    df_scaled = df_scaled.divide(df_quants, axis=1)
    return df_scaled

def clean_axes(ax):
    ax.set_aspect(0.5, adjustable='box')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
        
def load_data(gt_scdata_path, pred_scdata_path):
    # load pseudo cells for all ROIs for ground truth and predicted data
    pred_all = pd.DataFrame()
    gt_all = pd.DataFrame()
    pred_scdata_paths = sorted(glob.glob(pred_scdata_path + '/*.tsv'))
    
    for pred_path in pred_scdata_paths:
        roi_name = pred_path.split('/')[-1].split('.')[0]
        print('roi_name: ', roi_name)
        
        pred = pd.read_csv(pred_path, sep='\t', index_col=[0])        
        pred = pred.loc[:, ~pred.columns.isin(['X', 'Y', 'radius'])]
        pred = pred.loc[pred.isna().sum(axis=1) < 1, :]

        gt = pd.read_csv(os.path.join(gt_scdata_path, roi_name + '.tsv'), sep='\t', index_col=[0])
        gt = gt.loc[:, pred.columns]
        gt = gt.loc[gt.isna().sum(axis=1) < 1, :]

        pred_all = pd.concat([pred_all, pred]).reset_index(drop=True)
        gt_all = pd.concat([gt_all, gt]).reset_index(drop=True)

    return pred_all, gt_all

def plot_tsne_results(tsne_results_df, save_path, protein_sets=None):
    alpha = 0.8
    marker_size = 5
    for protein_set, proteins in protein_sets.items():
        fig, axes = plt.subplots(2, len(proteins), figsize=(14, 4))
        for ax in axes.flatten():
            ax.axis('off')

        for j, data_type in enumerate(sorted(tsne_results_df['type'].unique())):
            for i, prot in enumerate(proteins):
                g = sns.scatterplot(x='tSNE1', y='tSNE2', hue=prot, 
                                    data=tsne_results_df.loc[tsne_results_df['type'].isin([data_type]),:],
                                    # data=tsne_results_df.loc[tsne_results_df['type'] == data_type, :],
                                    ax=axes[j][i], alpha=alpha, s=marker_size, legend=False, 
                                    palette='Spectral_r')
                g.set_title(f"{prot}_{data_type}")
                g.axis(False)

        cax_left = 0.99 
        cax_bottom = 0.3  
        cax_width = 0.006
        cax_height = 0.4
        cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
        plt.colorbar(sm, cax=cax, orientation='vertical', label='Value', shrink=0.5)

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(save_path, f'tsne_{protein_set}.png'), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(save_path, f'tsne_{protein_set}.pdf'), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(save_path, f'tsne_{protein_set}.svg'), bbox_inches='tight', dpi=300)
        plt.show()
