import os
import numpy as np
import pandas as pd
import json
import glob

import cv2
import joblib

from pathlib import Path
from tqdm import tqdm
from src.annotation_utils import *

class wsi_celltyping: 

    '''
    Class for cell-typing on a whole-slide-level using a random forest model.
    Parameters:
    - rf_path: Path to the random forest model file.
    - cell_types: Cell types to be used for cell typing (e.g., 'tumor_CD8', 'tumor_CD8_CD4', 'tumor_CD8_CD4_CD20', 'all').
    - radius: Radius of the cell in micrometers.
    '''
    
    def __init__(self, rf_path, cell_types, radius):
        self.radius = radius
        self.rf_path = rf_path
        self.cell_types = cell_types

        if cell_types == 'tumor_CD8':
            rf_cell_types = ['tumor', 'Tcells.CD8', 'other']
        elif cell_types == 'tumor_CD8_CD4':
            rf_cell_types = ['tumor', 'Tcells.CD8', 'Tcells.CD4', 'other']
        elif cell_types == 'tumor_CD8_CD4_CD20':
            rf_cell_types = ['tumor', 'Tcells.CD8', 'Tcells.CD4', 'Bcells', 'other']
        elif cell_types == 'all':
            rf_cell_types = ['myeloid', 'Tcells.CD8', 'Bcells', 'tumor', 'vessels', 'Tcells.CD4', 'other']
        self.rf_cell_types = sorted(rf_cell_types)
        
    def get_wsi_celltyping(self, centroids_wsi, wsi_multiplex):

        # --- wsi pixels to cell counts ---
        x_max, y_max, n_channels = wsi_multiplex.shape
        ct_labels = []

        # iterate over all identified nuclei coordinates
        for i, centroid in tqdm(enumerate(centroids_wsi), total=len(centroids_wsi), desc="Processing", unit="item"):
            x0 = centroid[0]
            y0 = centroid[1]

            # average protein expression within the circle
            protein_sum = np.zeros((1,n_channels))
            n_pixels = 0
            for x in range(abs(x0-self.radius),(x0+self.radius)):
                for y in range(abs(y0-self.radius),(y0+self.radius)):
                    if (((x-x0)**2 + (y-y0)**2) <= self.radius**2) and (x<x_max and y<y_max) and (x>0 and y >0):
                        protein_sum = protein_sum + wsi_multiplex[x,y,:]
                        n_pixels = n_pixels + 1
            print(protein_sum, n_pixels)
            if n_pixels == 0:
                ct_labels.append(np.nan)
            else:
                protein_mean = protein_sum/n_pixels
                # print(protein_mean)
                # get cell-type label
                assert n_channels==11, 'RF prediction atm is possible only if the full multiplex is predicted!'
                ct_label = self.get_ct_label_rf(protein_mean)
                ct_labels.append(ct_label)

        ct_labels = np.array(ct_labels)
        print(len(ct_labels), len(centroids_wsi))    
        return ct_labels

    def get_ct_label_rf(self, protein_mean):
        ''' Get cell-type label (Tcells.CD8, tumor, other) based on predefined thresholds (125 among cells with the given cell-type label in train set)
        rf_object: object with trained RF
        rf_cts: cell-type labels that RF was trained on / can predict
        protein_mean: average protein expression of the pseudo-cell
        '''        
        rf_object = joblib.load(self.rf_path)
        rf_cts = sorted(self.rf_cell_types)
        raw_result = np.zeros((protein_mean.shape[0], len(rf_cts)))
        for i in range(len(rf_object.estimators_)):
            raw_result += rf_object.estimators_[i].predict(protein_mean)
        ct_label = rf_cts[np.argmax(raw_result)]
        ct_label = 'other' if ct_label not in ['Tcells.CD8', 'tumor', 'Tcells.CD3'] else ct_label
        return ct_label


def get_cellcount_in_mask(mask, centroids_CD8):
    mask_values_annots = mask[centroids_CD8[:, 1], centroids_CD8[:, 0]]
    coordinates_within_annots = centroids_CD8[mask_values_annots != 0]
    return len(coordinates_within_annots)

def get_celltype_centroids(celltype, ct_labels, centroids_wsi_level):
    indices = np.where(ct_labels==celltype)
    if len(indices[0])!=0: 
        centroids = centroids_wsi_level[indices]
    else: 
        print(f"No {celltype} predicted")
        centroids = []
    return centroids

def get_stroma_tumor_mask(img_annots_compartment): 
    mask_tumor = (np.all(img_annots_compartment == [255, 0, 0], axis=-1)).astype(np.uint8)
    mask_stroma = (np.all(img_annots_compartment == [0, 0, 255], axis=-1)).astype(np.uint8)
    mask_positive_lymphocytes = (np.all(img_annots_compartment == [255, 255, 0], axis=-1)).astype(np.uint8)
    mask_stroma = cv2.bitwise_or(mask_stroma, mask_positive_lymphocytes)
    return mask_tumor, mask_stroma

def get_cellcount_area_density(mask_tumor, mask_stroma, centroids_celltype, resolution_level):
    iCD_count, sCD_count = 0, 0
    if len(centroids_celltype)!=0: 
        iCD_count = get_cellcount_in_mask(mask_tumor, centroids_celltype)
        sCD_count = get_cellcount_in_mask(mask_stroma, centroids_celltype)

    # annotated area calculation 
    n_tumor_pixels = np.count_nonzero(mask_tumor)
    n_stroma_pixels = np.count_nonzero(mask_stroma)

    iCD_area = (n_tumor_pixels*(resolution_level**2)) # in um2 
    sCD_area = (n_stroma_pixels*(resolution_level**2)) # in um2

    # density calculation 
    iCD_density = 0 if iCD_area==0 else iCD_count/iCD_area
    sCD_density = 0 if sCD_area==0 else sCD_count/sCD_area
    # iCD_density = iCD_count/iCD_area
    # sCD_density = sCD_count/sCD_area
    return iCD_count, sCD_count, iCD_area, sCD_area, iCD_density, sCD_density


def process_and_save_annotations(tissue_regions_path, annotation_path, save_annotation_path):
    '''
    Process and save annotations for tumor and invasive margin regions from HE images.
    Parameters:
    - tissue_regions_path: Path to the directory containing tissue region annotations in .tif format.
    - annotation_path: Path to the directory containing tumor and invasive margin annotations.
    - save_annotation_path: Path to save the processed annotations.
    Returns:
    - None'''
    os.makedirs(save_annotation_path, exist_ok=True)
    for tif_annotation_path in glob.glob(str(tissue_regions_path) + '/*.tif'):
        sample = tif_annotation_path.split('/')[-1].split('-')[0].split('_')[0]
        region_annotation_path = glob.glob(str(annotation_path) + '/' + sample + '*.annotations')[0]

        save_path = os.path.join(save_annotation_path, f"{sample}_annotated_tumor_IM.npz")
        if not os.path.exists(save_path):
            print(f"Processing sample: {sample}")
            img_annotation = get_annotation_mask_HE(tif_annotation_path, get_color_code(), downsample=16)
            img_annotation = cv2.resize(img_annotation, (0, 0), fx=(0.3/0.23), fy=(0.3/0.23))
            img_annots_tumor, img_annots_IM = get_region_masks(img_annotation, region_annotation_path, downsample_factor=16, coutour_thickness=-1)
            np.savez(save_path, img_annots_tumor=img_annots_tumor, img_annots_IM=img_annots_IM)
        else:
            print(f"Sample {sample} already processed.")

def perform_wsi_cell_typing(args):
    '''
    Perform cell typing on whole slide images using the provided random forest model and save the results.
    Parameters:
    - args: Command line arguments containing paths and parameters for processing.
    Returns:
    - None
    '''
    imc_pred_basepath = os.path.join(args.wsi_pred_path, f'level_{args.level}')
    imc_pred_paths = glob.glob(imc_pred_basepath + '/*npy')
    save_path_analysis = os.path.join(args.wsi_pred_path, 'immune_phenotyping', f'rf_{args.cell_types}')
    os.makedirs(save_path_analysis, exist_ok=True)
    csv_save_path = os.path.join(save_path_analysis, f'immune_phenotype-pred-rf_{args.cell_types}.csv')

    resolution_level = args.resolution_he * (2**args.level)
    radius_level = round(args.radius / resolution_level)
    rf_path = glob.glob(os.path.join(args.rf_base_path, args.cell_types, '*.joblib'))[0]

    celltyping_obj = wsi_celltyping(rf_path, args.cell_types, radius_level)
    meta = pd.read_csv(args.meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]

    df_columns = ['Case_ID', 'iCD8_density_TC', 'sCD8_density_TC', 'iCD8_area_TC', 'sCD8_area_TC', 'iCD8_count_TC', 'sCD8_count_TC', 
                  'iCD8_density_IM', 'sCD8_density_IM', 'iCD8_area_IM', 'sCD8_area_IM', 'iCD8_count_IM', 'sCD8_count_IM',
                  'iCD3_density_TC', 'sCD3_density_TC', 'iCD3_area_TC', 'sCD3_area_TC', 'iCD3_count_TC', 'sCD3_count_TC',
                  'iCD3_density_IM', 'sCD3_density_IM', 'iCD3_area_IM', 'sCD3_area_IM', 'iCD3_count_IM', 'sCD3_count_IM']

    if os.path.exists(csv_save_path):
        df_pred = pd.read_csv(csv_save_path)
    else:
        df_pred = pd.DataFrame(columns=df_columns)
        df_pred.to_csv(csv_save_path, index=False)

    for imc_pred_path in imc_pred_paths:
        sample = os.path.basename(imc_pred_path).split('.')[0]
        if sample in df_pred['Case_ID'].values:
            print(f"Sample {sample} already processed.")
            continue

        process_sample(imc_pred_path, sample, meta, celltyping_obj, resolution_level, save_path_analysis, csv_save_path, df_columns, args.hovernet_path, args.level, args.radius)

def get_hovernet_file(hovernet_path, sample):
    """
    Retrieve the HoverNet file for a given sample.
    Parameters:
    - hovernet_path: Path to the directory containing HoverNet output files.
    - sample: Sample ID.
    Returns:
    - Path to the HoverNet file for the sample.
    """
    hovernet_files = glob.glob(os.path.join(hovernet_path, f"{sample}*.gz"))
    if not hovernet_files:
        raise FileNotFoundError(f"No HoverNet file found for sample: {sample}")
    return hovernet_files[0]

def process_sample(imc_pred_path, sample, meta, celltyping_obj, resolution_level, save_path_analysis, csv_save_path, df_columns, hovernet_path, level, radius):
    """
    Process a single sample for cell typing and save the results.
    Parameters:
    - imc_pred_path: Path to the IMC prediction file for the sample.
    - sample: Sample ID.
    - meta: Metadata DataFrame containing ground truth immune phenotype.
    - celltyping_obj: Instance of the wsi_celltyping class for cell typing.
    - resolution_level: Resolution level for processing.
    - save_path_analysis: Path to save the analysis results.
    - csv_save_path: Path to save the CSV file with results.
    - df_columns: List of columns for the results DataFrame.
    - hovernet_path: Path to the directory containing HoverNet output files.
    - level: Level of the WSI predictions.
    - radius: Radius of the cell in micrometers.
    Returns:
    - None
    """
    print(imc_pred_path)
    sample_phenotype = meta.loc[meta['tupro_id'] == sample, 'cd8_phenotype_revised']
    print('GT immune phenotype: ', sample_phenotype)

    wsi_pred = np.load(imc_pred_path)
    print('wsi_pred: ', wsi_pred.shape)

    try:
        f_hovernet_sample = get_hovernet_file(hovernet_path, sample)
    except FileNotFoundError as e:
        print(e)
        return

    centroids_wsi_level = get_hovernet_centroids(f_hovernet_sample, level)
    print(radius, round(radius / resolution_level))

    ct_labels = celltyping_obj.get_wsi_celltyping(centroids_wsi_level, wsi_pred)
    labels, counts = np.unique(ct_labels, axis=0, return_counts=True)
    print(labels, counts)

    centroids_CD8 = get_celltype_centroids(celltype='Tcells.CD8', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)
    centroids_CD3 = get_celltype_centroids(celltype='Tcells.CD3', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)
    centroids_tumor = get_celltype_centroids(celltype='tumor', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)

    print('centroids CD8: ', len(centroids_CD8))
    print('centroids CD3: ', len(centroids_CD3))
    print('centroids tumor: ', len(centroids_tumor))

    annotation_path_sample = os.path.join(save_path_analysis, f"{sample}_annotated_tumor_IM.npz")
    if not os.path.exists(annotation_path_sample):
        print(f"Sample {sample} does not have annotations")
        return

    img_annots = np.load(annotation_path_sample)
        
    mask_TC_tumor, mask_TC_stroma = get_stroma_tumor_mask(img_annots['img_annots_tumor'])
    mask_IM_tumor, mask_IM_stroma = get_stroma_tumor_mask(img_annots['img_annots_IM'])

    print('mask TC stroma and tumor ', mask_TC_stroma.shape, mask_TC_tumor.shape)
    print('mask IM stroma and tumor: ', mask_IM_stroma.shape, mask_IM_tumor.shape)

    iCD8_count_TC, sCD8_count_TC, iCD8_area_TC, sCD8_area_TC, iCD8_density_TC, sCD8_density_TC = get_cellcount_area_density(mask_TC_tumor, mask_TC_stroma, centroids_CD8, resolution_level)
    iCD3_count_TC, sCD3_count_TC, iCD3_area_TC, sCD3_area_TC, iCD3_density_TC, sCD3_density_TC = get_cellcount_area_density(mask_TC_tumor, mask_TC_stroma, centroids_CD3, resolution_level)

    iCD8_count_IM, sCD8_count_IM, iCD8_area_IM, sCD8_area_IM, iCD8_density_IM, sCD8_density_IM = get_cellcount_area_density(mask_IM_tumor, mask_IM_stroma, centroids_CD8, resolution_level)
    iCD3_count_IM, sCD3_count_IM, iCD3_area_IM, sCD3_area_IM, iCD3_density_IM, sCD3_density_IM = get_cellcount_area_density(mask_IM_tumor, mask_IM_stroma, centroids_CD3, resolution_level)

    print('CD8 TC: ', iCD8_count_TC, sCD8_count_TC, iCD8_area_TC, sCD8_area_TC, iCD8_density_TC, sCD8_density_TC)
    print('CD8 IM: ', iCD8_count_IM, sCD8_count_IM, iCD8_area_IM, sCD8_area_IM, iCD8_density_IM, sCD8_density_IM)
    print('CD3 TC: ', iCD3_count_TC, sCD3_count_TC, iCD3_area_TC, sCD3_area_TC, iCD3_density_TC, sCD3_density_TC)
    print('CD3 IM: ', iCD3_count_IM, sCD3_count_IM, iCD3_area_IM, sCD3_area_IM, iCD3_density_IM, sCD3_density_IM)

    new_row = [sample, iCD8_density_TC, sCD8_density_TC, iCD8_area_TC, sCD8_area_TC, iCD8_count_TC, sCD8_count_TC, 
                    iCD8_density_IM, sCD8_density_IM, iCD8_area_IM, sCD8_area_IM, iCD8_count_IM, sCD8_count_IM, 
                    iCD3_density_TC, sCD3_density_TC, iCD3_area_TC, sCD3_area_TC, iCD3_count_TC, sCD3_count_TC,
                    iCD3_density_IM, sCD3_density_IM, iCD3_area_IM, sCD3_area_IM, iCD3_count_IM, sCD3_count_IM]
    
    sample_row_df = pd.DataFrame([new_row], columns=df_columns)
    sample_row_df.to_csv(csv_save_path, mode='a', header=False)

    np.savez(os.path.join(save_path_analysis, f"{sample}-cell_coordinated.npz"), centroids_CD8=centroids_CD8, centroids_CD3=centroids_CD3, centroids_tumor=centroids_tumor, density_analysis=new_row)
