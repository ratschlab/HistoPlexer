import argparse
import os
import numpy as np
import pandas as pd
import json
import glob
import cv2
from hta.stats import HTA
from shapely.geometry import Point
import openslide
import matplotlib.pyplot as plt
from src.annotation_utils import get_region_masks


def load_metadata(meta_path, immune_gt_path):
    '''
    Load metadata and ground truth data for immune phenotyping.
    
    Parameters:
    meta_path (str): Path to the metadata file.
    immune_gt_path (str): Path to the immune ground truth file.
    
    Returns:
    pd.DataFrame: DataFrame containing merged metadata and ground truth data.
    
    '''
    meta = pd.read_csv(meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]
    xl = pd.ExcelFile(immune_gt_path)
    df_gt = xl.parse("Tabelle1")
    df_gt = df_gt[['Case_ID', 'Analysis_Region', 'Revised immune diagnosis', 'Density Tumor', 'Density Stroma total', 'Tumor Area (um²)',
    'Positive Lymphocytes Area (um²)', 'Stroma Area (um²)', 'Tumor:_AP_Positive_Cells', 'Stroma:_AP_Positive_Cells', 'Positive_Lymphocytes:_AP_Positive_Cells']]
    df_gt['Total_Stroma:_AP_Positive_Cells'] = df_gt['Stroma:_AP_Positive_Cells'] + df_gt['Positive_Lymphocytes:_AP_Positive_Cells']
    df_gt['Total Stromal area'] = df_gt['Stroma Area (um²)'] + df_gt['Positive Lymphocytes Area (um²)']
    df_gt.drop(columns=['Stroma Area (um²)', 'Positive Lymphocytes Area (um²)', 'Stroma:_AP_Positive_Cells', 'Positive_Lymphocytes:_AP_Positive_Cells'], inplace=True)
    df_gt = df_gt.drop(df_gt[df_gt['Analysis_Region'] == 'IM'].index)#.reset_index()
    df_gt = df_gt.drop(df_gt[df_gt['Analysis_Region'] == 'Peritumoral'].index)#.reset_index()
    df_gt.rename(columns={'Case_ID': 'tupro_id'}, inplace=True)
    df_gt = pd.merge(df_gt, meta, on='tupro_id')[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]
    return df_gt


def load_he_image(sample_HE_path, level):
    '''
    Load HE image from a whole slide image (WSI) file.
    
    Parameters:
    sample_HE_path (str): Path to the WSI file.
    level (int): Level of the image to be loaded.
    
    Returns:
    np.ndarray: Loaded HE image as a NumPy array.
    '''
    wsi = openslide.OpenSlide(sample_HE_path)
    he_img = wsi.read_region((0, 0), level, (wsi.level_dimensions[level]))
    return np.array(he_img.convert('RGB')).astype(np.uint8)


def overlay_contours(he_img, annotation_path, level):
    '''
    Overlay contours on the HE image based on the provided annotation path.
    
    Parameters:
    he_img (np.ndarray): HE image as a NumPy array.
    annotation_path (str): Path to the annotation file.
    level (int): Level of the image to be loaded.
    
    Returns:
    np.ndarray: HE image with contours overlaid.
    '''
    img_annots_tumor, _ = get_region_masks(he_img, annotation_path, downsample_factor=2**level, coutour_thickness=10, plot=False)
    gray_image = cv2.cvtColor(img_annots_tumor, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(he_img, [contour], -1, (200, 200, 200), 10)
    return he_img


def prune_centroids(data, TC_img, key):
    '''
    Prune centroids based on the TC image.
    
    Parameters:
    data (dict): Dictionary containing centroid data.
    TC_img (np.ndarray): TC image as a NumPy array.
    key (str): Key to access centroid data in the dictionary.
    
    Returns:
    list: List of pruned centroids.
    '''
    
    center_points = [Point(center) for center in data[key]]
    non_zero_points = [point for point in center_points if TC_img[int(point.y), int(point.x)][0] != 0]
    return [[int(point.x), int(point.y)] for point in non_zero_points]


def draw_centroids(he_img, centroids, color):
    '''
    Draw centroids on the HE image.
        
    Parameters:
    he_img (np.ndarray): HE image as a NumPy array.
    centroids (list): List of centroids to be drawn.
    color (tuple): Color for the centroids.
    
    Returns:
    np.ndarray: HE image with centroids drawn.
    '''
    for point in centroids:
        he_img = cv2.circle(he_img, point, 3, color, 1)
    return he_img


def save_plots(figs_save_sample, sample, sample_phenotype, plots):
    '''
    Save plots as SVG and PNG files.
    
    Parameters:
    figs_save_sample (str): Path to save the plots.
    sample (str): Sample name.
    sample_phenotype (list): List of sample phenotypes.
    plots (dict): Dictionary containing plot names and images. 
    
    Returns:
    None
    '''
    os.makedirs(figs_save_sample, exist_ok=True)
    for plot_name, plot_img in plots.items():
        plt.figure(figsize=(40, 20))
        plt.imshow(plot_img)
        plt.axis('off')
        plt.savefig(os.path.join(figs_save_sample, f'{sample}_{sample_phenotype[0].split(" ")[-1]}_{plot_name}.svg'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(figs_save_sample, f'{sample}_{sample_phenotype[0].split(" ")[-1]}_{plot_name}.png'), bbox_inches='tight', dpi=300)
        plt.close()