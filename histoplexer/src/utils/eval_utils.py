import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
import torch
import torchvision.transforms.functional as ttf
import torchvision.transforms as tt
import torch.nn as nn

from histoplexer.utils.constants import *

def get_protein_list(protein_set_name):
    '''Get list of protein names correspondig to protein_set_name
    protein_set_name: Name of the protein set {full, reduced, reduced_ext, single_protein_name}
    '''
    if protein_set_name=='full':
        protein_set = PROTEIN_LIST  # predict all proteins
    elif protein_set_name=='reduced':
        protein_set = celltype_relevant_prots # predict 10 proteins as in report
    elif protein_set_name=='selected':
        protein_set = selected_prots # predict 10 proteins, replaced SOX10 with SOX9 in reduced set 
    elif protein_set_name=='selected_snr':
        protein_set = selected_prots_snr # predict 11 proteins, chosen by Ruben on 03.02.23
    elif protein_set_name=='selected_snr_dgm4h':
        protein_set = selected_prots_snr_dgm4h # predict 10 proteins, same as miccai but w/o glut1; used for dgm4h
    elif protein_set_name=='selected_snr_nature':
        protein_set = selected_prots_snr_nature # predict 11 proteins, same as dgm4h + sox10
    elif protein_set_name=='selected_snr_v2':
        protein_set = selected_prots_snr_v2 # predict 12 proteins, replaced GLUT1 with ki-67 (proliferation marker) from selected_prots_snr and added SMA
    elif protein_set_name=='prots_pseudo_multiplex':
        protein_set = prots_pseudo_multiplex # predict 10 proteins, for miccai
    elif protein_set_name=='prots_tls':
        protein_set = prots_tls # predict all proteins for tls 
    elif protein_set_name=='prots_cd16_correlates':
        protein_set = prots_cd16_correlates # predict all proteins for cd16 
    elif protein_set_name=='prots_ki67_correlates':
        protein_set = prots_ki67_correlates # predict all proteins for Ki-67 
    elif protein_set_name=='selected_snr_ext':
        protein_set = selected_prots_snr_ext # predict 16 proteins, chosen by Ruben on 03.02.23
    elif protein_set_name=='reduced_ext':
        protein_set = celltype_relevant_prots_ext # predict extended list of proteins
    elif protein_set_name=='reduced_ext_ir':
        protein_set = celltype_relevant_prots_ext_ir # predict extended list of proteins + Ir
    elif protein_set_name == 'tumor_prots':
        protein_set = tumor_prots
    elif protein_set_name == 'immune_prots':
        protein_set = immune_prots
    elif protein_set_name == 'tcell_prots':
        protein_set = tcell_prots
    elif protein_set_name == 'ir_cd8':
        protein_set = ir_cd8
    elif protein_set_name == 'tumor_cd8':
        protein_set = tumor_cd8
    elif protein_set_name == 'cd3_cd8':
        protein_set = cd3_cd8
    elif protein_set_name == 'tumor_cd8_cd3':
        protein_set = tumor_cd8_cd3
    else:
        protein_set = [protein_set_name] # predict a single protein
    return protein_set

def standardize_img(img, global_avg=0, global_stdev=1):
    '''Center and standardize image using global_avg and global_stdev
    '''
    return (img - global_avg)/global_stdev

def destandardize_img(img, global_avg=0, global_stdev=1):
    '''Revert standardization and centering of the image using global_avg and global_stdev
    '''
    return (img*global_stdev) + global_avg


def min_max_scale(x, min_cohort, max_cohort):
    return (x-min_cohort)/(max_cohort - min_cohort)

def standardize(x, mean_cohort, std_cohort):
    return (x-mean_cohort)/std_cohort


def suppress_to_zero(x, q=0.25):
    ''' Set value considered noise (below quantile q) to 0
    x: vector with values
    q: quantile [0,1]
    '''
    thrs = np.quantile(x,q)
    x[x<thrs] = 0
    return x


def get_ordered_epoch_names(model_path):
    ''' Function to return names of modelstate files ordered by the epoch (usual sorting function does not properly sort bcs of the epoch naming convention)
    model_path: absolute path to the tb_logs folder of a given job
    '''
    files = [x for x in os.listdir(model_path) if 'model_state' in x]
    df_steps = pd.DataFrame({'step_number':[int(x.split('K_')[0]) for x in files], 'step_name':[x.split('_')[0] for x in files]})
    df_steps = df_steps.sort_values(by=['step_number'], ascending=True)
    return df_steps.step_name.to_list()


def get_last_epoch_number(model_path):
    ''' Function to find the last epoch for a given job
    model_path: absolute path to the tb_logs folder of a given job
    '''
    ordered_step_names = get_ordered_epoch_names(model_path)
    return ordered_step_names[-1]


def get_best_epoch_w_imgs(project_path, submission_id, level=2, data_set='valid'):
    ''' Extract the best epoch from chkpt selection files and check if corresponding inferred images exist
    project_path: Path where all data, results etc for project reside
    submission_id: job submission id
    level: resolution level ({2,4,6})
    data_set: data set {train, valid, test}
    '''
    if isinstance(project_path, str):
        project_path = Path(project_path)
    chkpt_path = project_path.joinpath('results', submission_id, 'chkpt_selection')
    if os.path.exists(chkpt_path):
        chkpt_file = '-'.join(['best_epoch','level_'+str(level),data_set])+'.txt'
        best_step_name = json.load(open(chkpt_path.joinpath(chkpt_file)))['best_step']
    if os.path.exists(project_path.joinpath('results', submission_id, data_set+'_images', 
                                            'step_' + best_step_name, 'level_'+str(level))):
        return best_step_name
    else:
        print('No epoch with images found')
        return np.nan


def get_last_epoch_w_imgs(project_path, submission_id, level=2, data_set='valid'):
    ''' Find the last epoch for which corresponding inferred images exist
    project_path: Path where all data, results etc for project reside
    submission_id: job submission id
    level: resolution level ({2,4,6})
    data_set: data set {train, valid, test}
    '''
    if isinstance(project_path, str):
        project_path = Path(project_path)
    imgs_path = project_path.joinpath('results', submission_id, data_set+'_images')
    imgs_steps = [x for x in os.listdir(imgs_path)]
    df_steps = pd.DataFrame({'step_number':[int(x.split('_')[1].split('K')[0]) for x in imgs_steps], 'step_name':[x.split('_')[1] for x in imgs_steps]})
    df_steps = df_steps.sort_values(by=['step_number'], ascending=True)
    last_step_name = df_steps.step_name.to_list()[-1]

    if os.path.exists(project_path.joinpath('results', submission_id, data_set+'_images', 'step_' + last_step_name, 'level_'+str(level))):
        return last_step_name
    else:
        print('No epoch with images found')
        return np.nan
    
def preprocess_img(img, dev0, downsample_factor=1, kernel_width=32, blur_sigma=0, avg_kernel=32, avg_stride=1):
    ''' Function to preprocess an img for evaluations
    img: np array tensor [H,W,C]
    dev0: device to send the data for torch operations
    downsample_factor: if > 1 then downsamples the img to shape[0]//downsample_factor
    kernel_width: Gaussian kernel size
    blur_sigma: Gaussian kernl sigma
    avg_kernel: averaging square kernel size
    avg_stride: averaging kernel stride
    '''
    img = torch.from_numpy(img.copy().transpose(2,0,1))
    # send to device for performing torch operations
    img.to(dev0)
    # downsample GT if pred IMC downsampled
    if downsample_factor > 1:
        img = ttf.resize(img, img.shape[1]//downsample_factor)
    if avg_kernel>0:
        avg_pool = nn.AvgPool2d(kernel_size=avg_kernel, padding=int(np.floor(avg_kernel/2)), stride=avg_stride)
        img = avg_pool(img)
    if blur_sigma > 0:
        spatial_denoise = tt.GaussianBlur(kernel_width, sigma=blur_sigma)
        img = spatial_denoise(img)
    img = np.asarray(img).transpose(1,2,0)
    
    return img


def get_tumor_prots_signal(img_np, protein_list, tumor_prots=['MelanA', 'gp100', 'S100', 'SOX9', 'SOX10']):
    ''' Aggregate tumor signal by taking max across tumor markers
    img_np: numpy array with protein expression [H,W,C]
    protein_list: protein_list corresponding to the columns of img_np (in the same order)
    tumor_prots: tumor proteins to use for aggregation (only thos found in protein_list will be used!)
    '''
    sel_prots = [x for x in tumor_prots if x in protein_list]
    assert len(sel_prots)>0, 'No tumor prots found!'
    sel_prots_idx = [protein_list.index(prot) for prot in sel_prots]
    img_np_sel = img_np[:,:,sel_prots_idx]
    # take max across all tumor markers
    img_np_sel = np.apply_along_axis(np.max, 2, img_np_sel).reshape(img_np_sel.shape[0], img_np_sel.shape[1], 1)
    
    return img_np_sel