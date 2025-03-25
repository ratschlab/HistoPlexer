import os
import numpy as np
import pandas as pd
from pathlib import Path
import json

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from src.celltyping_utils import plt_ax_adjust, get_density_bins

# python -m bin.celltyping_visualise --cell_types='tumor_CD8_CD4_CD20'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply RF on tabular predicted pseudo-single-cell data.')
    parser.add_argument('--pred_celltype_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/test_cell_types', 
                        help='Path to predicted cell types per pseudo-cell')
    parser.add_argument('--cell_types', type=str, required=False, default='all', help='which cell type setting was used. Helps in merging cell types')
    parser.add_argument('--gt_celltypes', type=str, required=False, default='/raid/sonali/project_mvs/data/tupro/imc_updated/coldata.tsv', help='metadata per cell segmented from IMC using CellProfiler (includes coordinates X,Y and cell-type)')    
    parser.add_argument('--save_path', type=str, required=False, default=None, help='Path to save predictions')
    parser.add_argument('--he_path', type=str, required=False, default='/raid/sonali/project_mvs/data/tupro/binary_he_rois', help='Path to he numpy rois')
    args = parser.parse_args()

    pred_celltype_path = os.path.join(args.pred_celltype_path, args.cell_types)
    if args.save_path is None:
        save_path = args.pred_celltype_path.replace('cell_types', 'ct_pics')
    else:
        save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    CELL_TYPES = ['Bcells', 'Tcells.CD4', 'Tcells.CD8', 'myeloid', 'other','tumor', 'vessels']

    # overlay cell type density maps 
    level = 2
    bin_lim = 1000//(2**(level-2))
    axmax = 1024//(2**(level-2))
    desired_resolution_px = 32//(2**(level-2))
    x_bins, y_bins = get_density_bins(desired_resolution_px, bin_lim, axmax)
    max_density = 1/((bin_lim/desired_resolution_px)**2)/10 # 1/(n_bins)
    cts = CELL_TYPES #[x for x in CELL_TYPES if x !='other']

    bin_lim, axmax, desired_resolution_px, axmax//desired_resolution_px, max_density
    
    # gt ct
    df_cts = pd.read_csv(args.gt_celltypes, sep='\t', index_col=[0])[['X', 'Y', 'cell_type', 'sample_roi']].reset_index()
    df_cts = df_cts.rename(columns={'cell_type': 'pred_cell_type'})
    print(df_cts.head())
    
    # color palette for cell types
    color_palette = {}
    color_palette['tumor'] =  np.array([1.0, 0, 0, 1.]) # red
    color_palette['Tcells.CD8'] =  np.array([0.223, 1.0, 0.196, 1.]) # green
    color_palette['Bcells'] =  np.array([0, 0.4, 1, 1.]) # blue
    color_palette['Tcells.CD4'] =  np.array([1. , 0.498, 0., 1.]) # orange

    # plot settings 
    offset = 0 
    marker_size = 10 # size of the point
    alpha = 0.4 # transparency of the point
    cts_sel = ['tumor', 'Bcells', 'Tcells.CD8', 'Tcells.CD4'] # which cell-types to plot
    sns.set_style("white")

    save_rois = ['MEGEGUJ_C1', 'MYBYFUW_F2', 'MAHEFOG_F2', 'MUFYDUM_F2', 'MEBIGAL_C1']
    
    for s_roi in save_rois:
        print(s_roi)
        he = np.load(os.path.join(args.he_path, s_roi+'.npy'))
        gt = df_cts[df_cts['sample_roi'] == s_roi][['X', 'Y', 'pred_cell_type']]
        gt = gt.loc[gt['pred_cell_type'].isin(cts_sel),:].sort_values(by='pred_cell_type')
        gt['present'] = 1

        pred = pd.read_csv(os.path.join(pred_celltype_path, s_roi+'.tsv'), sep='\t', index_col=[0])
        pred = pred.loc[pred['pred_cell_type'].isin(cts_sel),:].sort_values(by='pred_cell_type')
        pred['present'] = 1
        
        fig, axes = plt.subplots(1, 3, figsize=(9,3))
        fig.patch.set_facecolor('white')  # For the figure background

        axes[0].imshow(he, origin='lower')
        plt_ax_adjust(axes[0], title="H&E")

        sns.scatterplot(x='X', y='Y', data=gt, hue='pred_cell_type', ax=axes[1], s=marker_size, legend=False,
                    palette=color_palette, alpha=alpha)
        axes[1].set_ylim(0-offset,1000+offset)
        axes[1].set_xlim(0-offset,1000+offset)
        plt_ax_adjust(axes[1], title='GT cell-types')

        sns.scatterplot(x='Y', y='X', data=pred, hue='pred_cell_type', ax=axes[2], s=marker_size, legend=True,
                    palette=color_palette, alpha=alpha)
        axes[2].set_ylim(0-offset,1000+offset)
        axes[2].set_xlim(0-offset,1000+offset)
        plt_ax_adjust(axes[2], title='Predicted cell-types')
        plt.legend(bbox_to_anchor=(1,1))

        fig.subplots_adjust(wspace=0.05, hspace=-0.25)
        
        # save figure
        save_path_roi = os.path.join(save_path, s_roi)
        if not os.path.exists(save_path_roi):
            os.makedirs(save_path_roi)

        plt.savefig(os.path.join(save_path, save_path_roi, s_roi +  '-all.png'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(save_path, save_path_roi, s_roi +  '-all.pdf'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(save_path, save_path_roi, s_roi +  '-all.svg'), bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

        # --- save individual axes --- 
        # save HE image
        plt.figure(figsize=(5, 5))
        plt.imshow(he, origin='lower') 
        plt.axis('off')
        plt.savefig(os.path.join(save_path_roi, s_roi +'-he.png'), bbox_inches='tight', dpi=300, pad_inches=0)
        plt.savefig(os.path.join(save_path_roi, s_roi +'-he.pdf'), bbox_inches='tight', dpi=300, pad_inches=0)
        plt.savefig(os.path.join(save_path_roi, s_roi +'-he.svg'), bbox_inches='tight', dpi=300, pad_inches=0)
        plt.show()
        plt.close()

        # plotting gt and predicted cell-types
        plt_dict = {'GT_celltypes': [gt, 'X', 'Y'], 'pred_celltypes': [pred, 'Y', 'X']}

        for i, (key, col) in enumerate(plt_dict.items()):
            fig, ax_single = plt.subplots(figsize=(5, 5))
            sns.scatterplot(x=col[1], y=col[2], data=col[0], hue='pred_cell_type', ax=ax_single, s=marker_size*2, legend=False,
                        palette=color_palette, alpha=alpha)
            ax_single.set_ylim(0-offset,1000+offset)
            ax_single.set_xlim(0-offset,1000+offset)
            ax_single.axis('off')
            ax_single.set_xlabel('')
            ax_single.set_ylabel('')
            # plt_ax_adjust(ax_single, title='')
            fig.savefig(os.path.join(save_path_roi, s_roi + '-' + key + '.png'), bbox_inches='tight', dpi=300, pad_inches=0)
            fig.savefig(os.path.join(save_path_roi, s_roi + '-' + key + '.pdf'), bbox_inches='tight', dpi=300, pad_inches=0)
            fig.savefig(os.path.join(save_path_roi, s_roi + '-' + key + '.svg'), bbox_inches='tight', dpi=300, pad_inches=0)
            plt.show()
            plt.close()

        plt.show()
