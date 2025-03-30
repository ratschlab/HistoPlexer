import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import glob
from tqdm import tqdm

import cv2
import openslide
import tifffile
import math
import matplotlib.pyplot as plt
import time 
from PIL import Image
from sklearn.cluster import KMeans
import seaborn as sns
import xml.etree.ElementTree as ET
import joblib
import sys 

# ---------------------------------------------# 
# Class for cell-typing on a whole-slide-level # 
# ---------------------------------------------# 
class wsi_celltyping: 

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
            if n_pixels == 0:
                ct_labels.append(np.nan)
            else:
                protein_mean = protein_sum/n_pixels
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
    
    
    
import argparse
import os
import numpy as np
import pandas as pd
import json
import glob
import cv2
from hta.stats import HTA
from shapely.geometry import Point

parser = argparse.ArgumentParser(description="Configurations for getting HTA heterogeneity score")
parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
parser.add_argument("--wsi_celltyping_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000/immune_phenotyping/rf_tumor_CD8', help="Path to where cell typing for immune phenotyping are saved")
parser.add_argument("--saved_annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/tumor_IM_annotations', help="Path to saved tumor IM npz files saved in immune phenotyping")

args = parser.parse_args()
wsi_celltyping_path  = args.wsi_celltyping_path
meta_path = args.meta_path
meta = pd.read_csv(meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]

# HTA for cells inside tumor compartment
hta_save_path = os.path.join(wsi_celltyping_path, 'hta_samples_tumor-center.csv')
hta_cols = ['sample', 'immune_subtype', 'HTA', 'HTA_pval']

if os.path.exists(hta_save_path):
    df_hta = pd.read_csv(hta_save_path)
else:
    df_hta = pd.DataFrame(columns=hta_cols)  # Create an empty df
    df_hta.to_csv(hta_save_path, index=False)

region_size = 100  

for coords_path in glob.glob(wsi_celltyping_path + '/*-cell_coordinated.npz'):

    sample = coords_path.split('/')[-1].split('-')[0]

    if sample in df_hta['sample'].values:
        print(f"sample: {sample} already processed")
        continue
     
    immune_subtype = meta.loc[meta['tupro_id']==sample, 'cd8_phenotype_revised'].values[0]

    print(immune_subtype, sample)

    coords = np.load(coords_path)
    coords_tumor = coords['centroids_tumor']
    coords_CD8 = coords['centroids_CD8']

    data = np.load(coords_path)
    print(data.files)

    # load region annotations to prune coordinates
    sample_annotation_path = args.saved_annotation_path + '/' + sample + '_annotated_tumor_IM.npz'
    TC_img = np.load(sample_annotation_path)['img_annots_tumor']

    # Get the non-zero pixel coordinates from the image
    non_zero_indices = np.transpose(np.nonzero(np.any(TC_img != [0, 0, 0], axis=-1)))
    print(len(non_zero_indices))

    # Get the centroids of the CD8+ cells inside tumor center
    center_points = [Point(center) for center in coords_CD8]
    non_zero_points = [point for point in center_points if TC_img[int(point.y), int(point.x)][0] != 0]
    coords_CD8 = [[int(point.x), int(point.y)] for point in non_zero_points]
    print(len(coords['centroids_CD8']), len(coords_CD8))

    # Get the centroids of the Tumor cells inside tumor center
    center_points = [Point(center) for center in coords_tumor]
    non_zero_points = [point for point in center_points if TC_img[int(point.y), int(point.x)][0] != 0]
    coords_tumor = [[int(point.x), int(point.y)] for point in non_zero_points]
    print(len(coords['centroids_tumor']), len(coords_tumor))

    coords_tumor = np.array(coords_tumor)
    coords_CD8 = np.array(coords_CD8)

    if len(coords_tumor) == 0 or len(coords_CD8) == 0:
        print('No cells in tumor center')
        sample_hta_df = pd.DataFrame([[sample, immune_subtype, 0, 0]], columns=hta_cols)
        sample_hta_df.to_csv(hta_save_path, mode='a', header=False)
        continue

    # get max x and y coordinates for both tumor and CD8 cells
    max_x = max(np.max(coords_tumor[:,0]), np.max(coords_CD8[:,0]))
    max_y = max(np.max(coords_tumor[:,1]), np.max(coords_CD8[:,1]))

    # get a new array with the same shape as the image with two channels for tumor and CD8 cells
    # in each channel, if a pixel is occupied by a cell, the value is 1, otherwise 0
    image = np.zeros((max_x+1, max_y+1, 2))
    for i in range(coords_tumor.shape[0]):
        x, y = coords_tumor[i]
        image[x, y, 0] = 1
    for i in range(coords_CD8.shape[0]):
        x, y = coords_CD8[i]
        image[x, y, 1] = 1

    # image as int 
    image = image.astype(int)
    print(image.shape)

    # resize
    factor = 10
    new_size = (image.shape[1] // factor, image.shape[0] // factor)  # (width, height)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    print(image.shape)

    hta = HTA(image, region_size)
    hta_stat, hta_pval = hta.calc()
    print('HTA {:.2f},  p-val: {:.2e}'.format(hta_stat, hta_pval))

    sample_hta_df = pd.DataFrame([[sample, immune_subtype, round(hta_stat, 3), round(hta_pval, 5)]], columns=hta_cols)
    sample_hta_df.to_csv(hta_save_path, mode='a', header=False)
    
    
    
    
    
    
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
from src.annotation_utils import *


parser = argparse.ArgumentParser(description="Configurations for visualizing cell typing results on HE wsis")

parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
parser.add_argument("--wsi_celltyping_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000/immune_phenotyping/rf_tumor_CD8', help="Path to where cell typing for immune phenotyping are saved")
parser.add_argument("--saved_annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/tumor_IM_annotations', help="Path to saved tumor IM npz files saved in immune phenotyping")
parser.add_argument("--immune_gt_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/CD8_quantification/20220209_TuProcohort_CD8.xlsx', help="Path to excel files with CD8 quantification")
parser.add_argument("--he_basepath", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/HE_new_wsi', help="Path to where HE wsi files are saved")
parser.add_argument("--annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/regions_annotations', help="Path to tumor and invasive margin annotations on HE wsi")
parser.add_argument("--save_path", type=str, required=False, default='/raid/sonali/project_mvs/results/final_results/dummy_', help="Path where HE images with cell typing results are saved")

args = parser.parse_args()
wsi_celltyping_path  = args.wsi_celltyping_path
meta_path = args.meta_path
immune_gt_path = args.immune_gt_path
saved_annotation_path = args.saved_annotation_path
he_basepath = args.he_basepath
annotation_path = args.annotation_path
save_path = args.save_path
level = 4

# ---- loading metadata ----
# df metadata
meta = pd.read_csv(meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]

# getting GT immune phenotypes for samples 
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


# --- plotting for samples ---
samples_of_interest = ['MYGIFUD', 'MIDOBOL', 'MISYPUP', # inflamed
                       'MUGAHEK', 'MEWORAT', 'MOBICUN', 'MIDEKOG', 'MEFYFAK', 'MUBYJOF', # excluded 
                       'MEKAKYD'] # desert 

# for coordinates_file in glob.glob(save_path_cells + '/*cell_coordinated.npz'): 
for sample_of_interest in samples_of_interest: 
    print(sample_of_interest)
    coordinates_file = glob.glob(wsi_celltyping_path + '/' + sample_of_interest +  '*cell_coordinated.npz')[0]
    print(coordinates_file)

    sample = coordinates_file.split('/')[-1].split('-')[0]
    print(sample)
    data = np.load(coordinates_file)
    print(data.files)

    # --- getting phenotype for sample ---
    sample_phenotype = df_gt.loc[df_gt['tupro_id'] == sample][['cd8_phenotype_revised', 'pathology_immune_diagnosis']].values[0]#, 'Revised immune diagnosis']]
    print(sample_phenotype)

    # --- loading HE at desired level ---
    sample_HE_path = glob.glob(he_basepath + '/*' + sample + '*.ndpi')[0]
    wsi = openslide.OpenSlide(sample_HE_path)
    he_img = wsi.read_region((0, 0), level, (wsi.level_dimensions[level]))
    he_img = np.array(he_img.convert('RGB')).astype(np.uint8)
    print(he_img.shape)

    # --- HE with TC boundary overlay ---
    region_annotation_path = glob.glob(str(annotation_path) + '/' + sample_of_interest + '*.annotations')[0]
    img_annots_tumor, _ = get_region_masks(he_img, region_annotation_path, downsample_factor=2**level, coutour_thickness=10, plot=False)

    # gettting contour from TC mask 
    gray_image = cv2.cvtColor(img_annots_tumor, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Find contours

    # Draw the contours based on their hierarchy
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:# External boundary (no parent)
            cv2.drawContours(he_img, [contour], -1, (200,200,200), 10)  # Solid line for outer contours

    # --- drawing cell coordinates on he img ---
    # load annotations for the sample qnd overlay on HE
    sample_annotation_path = saved_annotation_path + '/' + sample + '_annotated_tumor_IM.npz'
    TC_img = np.load(sample_annotation_path)['img_annots_tumor']

    # Get the non-zero pixel coordinates from the image
    non_zero_indices = np.transpose(np.nonzero(np.any(TC_img != [0, 0, 0], axis=-1)))
    print(len(non_zero_indices))

    # Get the centroids of the CD8+ cells inside tumor center
    center_points = [Point(center) for center in data['centroids_CD8']]
    non_zero_points = [point for point in center_points if TC_img[int(point.y), int(point.x)][0] != 0]
    centroids_CD8_pruned = [[int(point.x), int(point.y)] for point in non_zero_points]
    print(len(data['centroids_CD8']), len(centroids_CD8_pruned))

    # Get the centroids of the Tumor cells inside tumor center
    center_points = [Point(center) for center in data['centroids_tumor']]
    non_zero_points = [point for point in center_points if TC_img[int(point.y), int(point.x)][0] != 0]
    centroids_Tumor_pruned = [[int(point.x), int(point.y)] for point in non_zero_points]
    print(len(data['centroids_tumor']), len(centroids_Tumor_pruned))

    # --- drawing cell coordinates on he img ---
    he_img_CD8_centroids = he_img.copy()
    for point in centroids_CD8_pruned:
        he_img_CD8_centroids = cv2.circle(he_img_CD8_centroids, point, 3, (57, 255, 50), 1)
    he_img_CD8_tumor_centroids = he_img_CD8_centroids.copy()
    for point in centroids_Tumor_pruned:
        he_img_CD8_tumor_centroids = cv2.circle(he_img_CD8_tumor_centroids, point, 3, (255, 0, 0), 1)

    # -- region classification overlay on he ---
    label_to_new_color = {
        'Whitespace': (255, 255, 255),     # white
        'Positive Lymphocytes': tuple([int(color*255) for color in cm.get_cmap('YlOrBr')(0.3)[0:3]]), # yellow
        "Stroma": tuple([int(color*255) for color in cm.get_cmap('bwr')(0.1)[0:3]]),   # blue
        'Tumor': tuple([int(color*255) for color in cm.get_cmap('Reds')(0.7)[0:3]]),  # red
        'others': (0, 0, 0) # black
    }
    
    color_code = {(255, 255, 255): "Whitespace", # white
        (255, 255, 0): "Positive Lymphocytes", # yellow   
        (0, 0, 255): "Stroma", # blue
        (255, 0, 0): "Tumor", # red           
        (0, 0, 0): "others" # pigment etc 
                }
    pre_defined_colors = [list(t) for t in list(color_code.keys())]
    pre_defined_colors = np.array(pre_defined_colors).astype('double')

    TC_img_overlay = map_colors(TC_img.copy(), pre_defined_colors)
    colors, counts = np.unique(TC_img_overlay.reshape(-1, TC_img_overlay.shape[-1]), axis=0, return_counts=True)
    print('Unique colors and counts in annotated image: ', len(colors))

    # Iterate over the original colors and labels
    for original_color, label in color_code.items():
        new_color = label_to_new_color[label]
        mask = np.all(TC_img_overlay == original_color, axis=-1)    
        TC_img_overlay[mask] = new_color

    TC_img_overlay = cv2.resize(TC_img_overlay, (he_img.shape[1], he_img.shape[0]))
    mask = np.all(TC_img_overlay == [0, 0, 0], axis=-1)
    TC_img_overlay[mask] = he_img[mask]

    # --- plotting ---
    fig, ax = plt.subplots(1,4, figsize=(40,20))
    ax[0].imshow(he_img)
    ax[1].imshow(TC_img_overlay)
    ax[2].imshow(he_img_CD8_centroids)
    ax[3].imshow(he_img_CD8_tumor_centroids)

    figs_save_sample = os.path.join(save_path, sample)
    os.makedirs(figs_save_sample, exist_ok=True)

    for a in ax: # axis off
        a.axis('off')
    plt.savefig(os.path.join(figs_save_sample, f'{sample}_{sample_phenotype[0].split(" ")[-1]}_HE_regions_predCD8.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(figs_save_sample, f'{sample}_{sample_phenotype[0].split(" ")[-1]}_HE_regions_predCD8.png'), bbox_inches='tight', dpi=300)
    plt.show()

    # saving 
    plot_imgs = {'1_HE': he_img, '2_regions': TC_img_overlay, '3_predCD8': he_img_CD8_centroids, '4_pred-CD8-tumor':he_img_CD8_tumor_centroids}
    for plot_name, plot_img in plot_imgs.items():
        figure = plt.figure(figsize=(40, 20))
        plt.imshow(plot_img)
        plt.axis('off')
        plt.savefig(os.path.join(figs_save_sample, f'{sample}_{sample_phenotype[0].split(" ")[-1]}_{plot_name}.svg'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(figs_save_sample, f'{sample}_{sample_phenotype[0].split(" ")[-1]}_{plot_name}.png'), bbox_inches='tight', dpi=300)
        plt.show()





import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
# cm 
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser(description="Configurations for getting immune phenotyping stratification")
parser.add_argument("--saved_path_analysis", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000/immune_phenotyping/rf_tumor_CD8', help="Path to where cell typing for immune phenotyping are saved")
parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
parser.add_argument("--immune_gt_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/CD8_quantification/20220209_TuProcohort_CD8.xlsx', help="Path to excel files with CD8 quantification")
parser.add_argument("--gt_labels_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/CD8_quantification/immune_type-Marta.xlsx', help="Path to excel files with CD8 quantification")
parser.add_argument("--annotation_meta_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/240418_TuPro_Mastersheet.xlsx', help="Including metadata for HALO annotations, eg quality of annotations")
parser.add_argument("--save_path", type=str, required=False, default='/raid/sonali/project_mvs/results/final_results/dummy__', help="Path where box plots and results are saved")


args = parser.parse_args()
saved_path_analysis = args.saved_path_analysis
meta_path = args.meta_path
immune_gt_path = args.immune_gt_path
gt_labels_path = args.gt_labels_path
annotation_meta_path = args.annotation_meta_path
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)  

csv_path = glob.glob(saved_path_analysis + '/immune_phenotype-pred-' + '*.csv')[0]
print(csv_path)
df_pred = pd.read_csv(csv_path, index_col=None) 
df_pred.rename(columns={'Case_ID': 'tupro_id'}, inplace=True)
print(df_pred.head())


# comparing with ground truth 
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

df_gt = pd.merge(df_gt, meta, on='tupro_id')

df_gt_subset = df_gt[['tupro_id', 'Analysis_Region', 'Revised immune diagnosis', 'Density Tumor', 'Density Stroma total', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]
df_pred_subset = df_pred[['tupro_id', 'iCD8_density_TC', 'sCD8_density_TC', 'iCD3_density_TC', 'sCD3_density_TC']]
df_merged_subset = pd.merge(df_pred_subset, df_gt_subset, on='tupro_id')

# comparing with ground truth 
xl = pd.ExcelFile(gt_labels_path)
df_gt_labels = xl.parse("Sheet1")
df_gt_labels.rename(columns={'sample': 'tupro_id'}, inplace=True)
df_gt = pd.merge(df_gt, df_gt_labels, on='tupro_id')

# including annotaiton quality score
xl = pd.ExcelFile(annotation_meta_path)
df_annotations = xl.parse('Sheet 1 - melanoma-merged_clini')
df_annotations.columns = df_annotations.iloc[0]
df_annotations = df_annotations[1:]
df_annotations = df_annotations[['sample_id', 'S', 'biopsy_localisation', 'Comment', 'Exclude']]

df_annotations = df_annotations.rename(columns={'sample_id': 'tupro_id', 'S': 'annotation_quality'})
score_counts = df_annotations['annotation_quality'].value_counts()

df_annotations = df_annotations[~df_annotations['annotation_quality'].str.contains('*', regex=False, na=False)]
score_counts = df_annotations['annotation_quality'].value_counts()
print(score_counts)

df_gt_annotations = pd.merge(df_gt, df_annotations, on='tupro_id')
df_merged = pd.merge(df_pred, df_gt_annotations, on='tupro_id')

df_merged = df_merged[~((df_merged['Exclude'] == 1))]

print(df_merged['annotation_quality'].value_counts())
print(df_merged['cd8_phenotype_revised'].value_counts())


# density correlation
# iCD8
spearman_corr_iCD8_density = df_merged['iCD8_density_TC'].corr(df_merged['Density Tumor'], method='spearman')
print("sCorr iCD8 density:", spearman_corr_iCD8_density)

pearson_corr_iCD8_density = df_merged['iCD8_density_TC'].corr(df_merged['Density Tumor'], method='pearson')
print("pCorr iCD8 density:", pearson_corr_iCD8_density)
print('\n')

# sCD8
spearman_corr_sCD8_density = df_merged['sCD8_density_TC'].corr(df_merged['Density Stroma total'], method='spearman')
print("sCorr sCD8 density:", spearman_corr_sCD8_density)

pearson_corr_sCD8_density = df_merged['sCD8_density_TC'].corr(df_merged['Density Stroma total'], method='pearson')
print("pCorr sCD8 density:", pearson_corr_sCD8_density)

# density correlation

annotation_quality_labels = {2: 'best', 1: 'acceptable', 0: 'poor'}

for i in range(3):# in ['immune desert', 'immune excluded', 'inflamed']:
    print('annotation_quality: ', annotation_quality_labels[i])
    condition = df_merged['annotation_quality'] == i
    filtered_df = df_merged[condition]

    spearman_corr_iCD8_density = filtered_df['iCD8_density_TC'].corr(filtered_df['Density Tumor'], method='spearman')
    pearson_corr_iCD8_density = filtered_df['iCD8_density_TC'].corr(filtered_df['Density Tumor'], method='pearson')
    print("sCorr and pCorr iCD8 density:", round(spearman_corr_iCD8_density, 3), round(pearson_corr_iCD8_density, 3))

    # sCD8
    spearman_corr_sCD8_density = filtered_df['sCD8_density_TC'].corr(filtered_df['Density Stroma total'], method='spearman')
    pearson_corr_sCD8_density = filtered_df['sCD8_density_TC'].corr(filtered_df['Density Stroma total'], method='pearson')
    print("sCorr and pCorr sCD8 density:", round(spearman_corr_sCD8_density, 3), round(pearson_corr_sCD8_density, 3))
    print('\n')


df_merged['total_density_TC'] = (df_merged['iCD8_count_TC'] + df_merged['sCD8_count_TC']) / (df_merged['iCD8_area_TC'] + df_merged['sCD8_area_TC'])
df_merged['total_density_IM'] = (df_merged['iCD8_count_IM'] + df_merged['sCD8_count_IM']) / (df_merged['iCD8_area_IM'] + df_merged['sCD8_area_IM'])
df_merged['total_density'] = (df_merged['iCD8_count_TC'] + df_merged['sCD8_count_TC'] + df_merged['iCD8_count_IM'] + df_merged['sCD8_count_IM']) / (df_merged['iCD8_area_TC'] + df_merged['sCD8_area_TC'] + df_merged['iCD8_area_IM'] + df_merged['sCD8_area_IM'])

phenotype_mapping= {
    'immune desert': 'desert',
    'immune excluded': 'excluded',
    'inflamed': 'inflamed'
}

df_merged['cd8_phenotype_revised'] = df_merged['cd8_phenotype_revised'].map(phenotype_mapping)


custom_order = ['desert', 'excluded', 'inflamed']
custom_colors = [cm.get_cmap('terrain')(0.14), cm.get_cmap('terrain')(0.3), cm.get_cmap('jet')(0.75)] # colormaps
annotation_quality_labels = {2: 'best', 1: 'acceptable', 0: 'poor'}
plot_col_list = ['iCD8_density_TC', 'sCD8_density_TC']

for col in plot_col_list:
    print(col)
    fig, axes = plt.subplots(1, 4, figsize=(4 * 7, 5))
    sns.boxplot(x='cd8_phenotype_revised', y=col, data=df_merged, order=custom_order, palette=custom_colors, ax=axes[0])
    axes[0].set_xlabel('Immune Phenotypes')
    axes[0].set_ylabel(col)
    print(len(df_merged))

    # saving 
    fig_single, ax_single = plt.subplots(figsize=(5, 5))
    sns.boxplot(x='cd8_phenotype_revised', y=col, data=df_merged, order=custom_order, palette=custom_colors, ax=ax_single)
    ax_single.set_title('')
    ax_single.set_xlabel('')
    ax_single.set_ylabel('Density')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use scientific notation
 
    fig_single.savefig(os.path.join(save_path, f'3class_{col}_all.svg'), bbox_inches='tight', dpi=300)
    fig_single.savefig(os.path.join(save_path, f'3class_{col}_all.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_single)

    i = 1
    for key, value in annotation_quality_labels.items():
        print(key, value)
        condition = df_merged['annotation_quality'] == key
        filtered_df = df_merged[condition]
        print('annotation_quality: ', value, len(filtered_df))

        # plt.figure(figsize=(10, 6))
        sns.boxplot(x='cd8_phenotype_revised', y=col, data=filtered_df, order=custom_order, palette=custom_colors, ax=axes[i])

        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'annotation score: {value}')  
        i+=1

        # saving 
        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='cd8_phenotype_revised', y=col, data=filtered_df, order=custom_order, palette=custom_colors, ax=ax_single)
        ax_single.set_title('')
        ax_single.set_xlabel('')
        ax_single.set_ylabel('Density', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use scientific notation
 
        fig_single.savefig(os.path.join(save_path, f'3class_{col}_{value}.svg'), bbox_inches='tight', dpi=300)
        fig_single.savefig(os.path.join(save_path, f'3class_{col}_{value}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig_single)


mapping_dict = {'desert': 'cold', 'excluded': 'cold', 'inflamed': 'hot'}
color_cold = tuple((x + y) / 2 for x, y in zip(cm.get_cmap('terrain')(0.14), cm.get_cmap('terrain')(0.3)))
color_dict = {'cold': color_cold, 'hot': cm.get_cmap('jet')(0.75)}

# custom_colors = [cm.get_cmap('terrain')(0.14), cm.get_cmap('terrain')(0.3), cm.get_cmap('jet')(0.75)] # colormaps

custom_order = ['cold', 'hot']
annotation_quality_labels = {2: 'best', 1: 'acceptable', 0: 'poor'}

df_merged['cd8_phenotype_revised_'] = df_merged['cd8_phenotype_revised'].map(mapping_dict)

plot_col_list = ['iCD8_density_TC', 'sCD8_density_TC']

for col in plot_col_list:
    print(col)
    fig, axes = plt.subplots(1, 4, figsize=(4 * 7, 5))
    sns.boxplot(x='cd8_phenotype_revised_', y=col, data=df_merged, order=custom_order, ax=axes[0], palette=color_dict)
    axes[0].set_xlabel('Immune Phenotypes')
    axes[0].set_ylabel(col)
    print(len(df_merged))

    # saving 
    fig_single, ax_single = plt.subplots(figsize=(5, 5))
    sns.boxplot(x='cd8_phenotype_revised_', y=col, data=df_merged, order=custom_order, ax=ax_single, palette=color_dict)
    ax_single.set_title('')
    ax_single.set_xlabel('')
    ax_single.set_ylabel('Density')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) 
    
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use scientific notation
 
    fig_single.savefig(os.path.join(save_path, f'2class_{col}_all.svg'), bbox_inches='tight', dpi=300)
    fig_single.savefig(os.path.join(save_path, f'2class_{col}_all.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_single)

    i = 1
    for key, value in annotation_quality_labels.items():
        print(key, value)
        condition = df_merged['annotation_quality'] == key
        filtered_df = df_merged[condition]
        print('annotation_quality: ', value, len(filtered_df))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cd8_phenotype_revised_', y=col, data=filtered_df, order=custom_order, ax=axes[i], palette=color_dict)

        axes[i].set_xlabel('Immune Phenotypes')
        axes[i].set_ylabel(col)
        axes[i].set_title(f'annotation score: {value}')  
        i+=1

        # saving 
        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='cd8_phenotype_revised_', y=col, data=filtered_df, order=custom_order, ax=ax_single, palette=color_dict)
        ax_single.set_title('')
        ax_single.set_xlabel('')
        ax_single.set_ylabel('Density')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14) 

        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use scientific notation
 
        fig_single.savefig(os.path.join(save_path, f'2class_{col}_{value}.svg'), bbox_inches='tight', dpi=300)
        fig_single.savefig(os.path.join(save_path, f'2class_{col}_{value}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig_single)

    plt.show()









import argparse
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import glob
from tqdm import tqdm

import openslide
import tifffile
import math
import matplotlib.pyplot as plt
import time 
from PIL import Image
from sklearn.cluster import KMeans
from skimage.transform import resize
import cv2
import seaborn as sns
import xml.etree.ElementTree as ET
import joblib
import sys 
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from hta.stats import HTA
from shapely.geometry import Point

from src.annotation_utils import *
from src.wsi_celltyping import *
from src.immune_phenotyping_utils import *

parser = argparse.ArgumentParser(description="Configurations for getting immune phenotyping stratification")

parser.add_argument("--annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/regions_annotations', help="Path to tumor and invasive margin annotations on HE wsi")
parser.add_argument("--tissue_regions_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/classifier_mask', help="Path to tif files from halo with tissue regions")

parser.add_argument("--wsi_pred_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000', help="Path to wsi predictions from the model")
parser.add_argument("--level", type=int, required=False, default=4, help="which level to use from wsi predictions")
parser.add_argument("--hovernet_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/hovernet_output', help="Path to hovernet output for HE wsi")
parser.add_argument("--rf_base_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/cell_typing/rf/split3/', help="Path to random forest model for cell typing")

parser.add_argument("--cell_types", type=str, required=False, default='tumor_CD8', help="Which cell types to use for cell typing: tumor_CD8, tumor_CD8_CD4, tumor_CD8_CD4_CD20, all")
parser.add_argument("--radius", type=int, required=False, default=5, help="radius of the cell in um")
parser.add_argument("--resolution_he", type=float, required=False, default=0.23, help="resoluton of HE image in um/px")

parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")

parser.add_argument("--save_annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/tumor_IM_annotations', help="Path to saved tumor IM annotations using halo annotations")

args = parser.parse_args()
annotation_path = args.annotation_path
tissue_regions_path = args.tissue_regions_path

wsi_pred_path = args.wsi_pred_path
hovernet_path = args.hovernet_path
rf_base_path = args.rf_base_path
cell_types = args.cell_types
level = args.level
radius = args.radius
resolution_he = args.resolution_he

meta_path = args.meta_path
save_annotation_path = args.save_annotation_path

os.makedirs(save_annotation_path, exist_ok=True)

# ---- process and save annotation coordinated as annotation masks -----
for tif_annotation_path in glob.glob(str(tissue_regions_path) + '/*.tif'):
    sample = tif_annotation_path.split('/')[-1].split('-')[0].split('_')[0]
    region_annotation_path = glob.glob(str(annotation_path) + '/' + sample + '*.annotations')[0]
    # print(region_annotation_path)

    if not os.path.exists(save_annotation_path + '/' + sample + '_annotated_tumor_IM.npz'):
        print(f"sample: {sample}")
        print(f"tif annotation path: {tif_annotation_path}")
        print(f"region annotation path: {region_annotation_path}")

        img_annotation = get_annotation_mask_HE(tif_annotation_path, get_color_code(), downsample=16)
        print(f"img_annotation: {img_annotation.shape}")

        # resizing as tif annotated image is saved at resolutuon of 0.3um/px and HE is at resolution 0.23um/px -- so resizing to match the HE image
        img_annotation = cv2.resize(img_annotation, (0,0), fx=(0.3/0.23), fy=((0.3/0.23)))
        print(f"img_annotation: {img_annotation.shape}")

        img_annots_tumor, img_annots_IM = get_region_masks(img_annotation, region_annotation_path, downsample_factor=16, coutour_thickness=-1)
        print(f"img_annots_tumor: {img_annots_tumor.shape}, img_annots_IM: {img_annots_IM.shape}")

        # saving the annotated images
        np.savez(save_annotation_path + '/' + sample + '_annotated_tumor_IM.npz', img_annots_tumor=img_annots_tumor, img_annots_IM=img_annots_IM)
    else: 
        # print(f"sample: {sample} already exists")
        continue


# ---- get wsi cell typing -----
# loading wsi predictions
imc_pred_basepath = os.path.join(wsi_pred_path, 'level_' + str(level))
imc_pred_paths = glob.glob(imc_pred_basepath + '/*npy')
print(len(imc_pred_paths), imc_pred_paths[0])

# save paths for analysis
save_path_analysis = os.path.join(wsi_pred_path, 'immune_phenotyping', 'rf_' + cell_types)
os.makedirs(save_path_analysis, exist_ok=True)
csv_save_path = os.path.join(save_path_analysis, 'immune_phenotype-pred-' + 'rf_' + cell_types + '.csv')

resolution_level = resolution_he * (2**level)
radius_level = round(radius/resolution_level)
print('radius level: ', radius_level)
rf_path = glob.glob(rf_base_path + '/' + cell_types + '/*.joblib')[0]

celltyping_obj = wsi_celltyping(rf_path, cell_types, radius_level)
meta = pd.read_csv(meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]

df_columns = ['Case_ID', 'iCD8_density_TC', 'sCD8_density_TC', 'iCD8_area_TC', 'sCD8_area_TC', 'iCD8_count_TC', 'sCD8_count_TC', 
                'iCD8_density_IM', 'sCD8_density_IM', 'iCD8_area_IM', 'sCD8_area_IM', 'iCD8_count_IM', 'sCD8_count_IM',
                'iCD3_density_TC', 'sCD3_density_TC', 'iCD3_area_TC', 'sCD3_area_TC', 'iCD3_count_TC', 'sCD3_count_TC',
                'iCD3_density_IM', 'sCD3_density_IM', 'iCD3_area_IM', 'sCD3_area_IM', 'iCD3_count_IM', 'sCD3_count_IM']

if os.path.exists(csv_save_path):
    df_pred = pd.read_csv(csv_save_path)
else:
    df_pred = pd.DataFrame(columns=df_columns) # Create an empty df
    df_pred.to_csv(csv_save_path, index=False)

for imc_pred_path in imc_pred_paths: 

    print(imc_pred_path)
    sample = imc_pred_path.split('/')[-1].split('.')[0]

    if sample in df_pred['Case_ID'].values:
        print(f"sample: {sample} already processed")
        continue 

    # get GT immune phenotype
    sample_phenotype = meta.loc[meta['tupro_id']==sample, 'cd8_phenotype_revised']
    print('GT immune phenotype: ', sample_phenotype)

    # load predicted wsi
    wsi_pred = np.load(imc_pred_path)
    print('wsi_pred: ', wsi_pred.shape)

    # get hovernet cell centroids
    f_hovernet_sample = glob.glob(str(hovernet_path) + '/' + sample + '*.gz')[0]
    centroids_wsi_level = get_hovernet_centroids(f_hovernet_sample, level)
    print(radius, radius_level)

    # --- wsi pixels to cell counts ---
    ct_labels = celltyping_obj.get_wsi_celltyping(centroids_wsi_level, wsi_pred)
    labels, counts = np.unique(ct_labels, axis=0, return_counts=True)
    print(labels, counts)

    # get celltype centroids
    centroids_CD8 = get_celltype_centroids(celltype='Tcells.CD8', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)
    centroids_CD3 = get_celltype_centroids(celltype='Tcells.CD3', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)
    centroids_tumor = get_celltype_centroids(celltype='tumor', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)

    print('centroids CD8: ', len(centroids_CD8))
    print('centroids CD3: ', len(centroids_CD3))
    print('centroids tumor: ', len(centroids_tumor))

    # loading annotation masks: tumor and stroma within tumor compartment
    annotation_path_sample = save_path_processed_annotations + '/' + sample + '_annotated_tumor_IM.npz'
    if not os.path.exists(save_path_processed_annotations + '/' + sample + '_annotated_tumor_IM.npz'):
        print(f"sample: {sample} does not have annotations")
        continue

    img_annots = np.load(annotation_path_sample)
        
    # tumor and stroma mask for tumor compartment and invasive margin
    mask_TC_tumor, mask_TC_stroma = get_stroma_tumor_mask(img_annots['img_annots_tumor']) # for tumor compartment
    mask_IM_tumor, mask_IM_stroma = get_stroma_tumor_mask(img_annots['img_annots_IM']) # for invasive margin

    print('mask TC stroma and tumor ', mask_TC_stroma.shape, mask_TC_tumor.shape)
    print('mask IM stroma and tumor: ', mask_IM_stroma.shape, mask_IM_tumor.shape)

    # getting cell count in regions 
    # for TC
    iCD8_count_TC, sCD8_count_TC, iCD8_area_TC, sCD8_area_TC, iCD8_density_TC, sCD8_density_TC = get_cellcount_area_density(mask_TC_tumor, mask_TC_stroma, centroids_CD8, resolution_level)
    iCD3_count_TC, sCD3_count_TC, iCD3_area_TC, sCD3_area_TC, iCD3_density_TC, sCD3_density_TC = get_cellcount_area_density(mask_TC_tumor, mask_TC_stroma, centroids_CD3, resolution_level)

    # for IM
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

    # save coordinates visualisation
    np.savez(os.path.join(save_path_analysis, sample + '-cell_coordinated.npz'), centroids_CD8=centroids_CD8, centroids_CD3=centroids_CD3, centroids_tumor=centroids_tumor, density_analysis=new_row)


































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
            if n_pixels == 0:
                ct_labels.append(np.nan)
            else:
                protein_mean = protein_sum/n_pixels
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

        process_sample(imc_pred_path, sample, meta, celltyping_obj, resolution_level, save_path_analysis, csv_save_path, df_columns)

def process_sample(imc_pred_path, sample, meta, celltyping_obj, resolution_level, save_path_analysis, csv_save_path, df_columns):
    '''
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
    Returns:
    - None
    '''
    print(imc_pred_path)
    sample_phenotype = meta.loc[meta['tupro_id'] == sample, 'cd8_phenotype_revised']
    print('GT immune phenotype: ', sample_phenotype)

    wsi_pred = np.load(imc_pred_path)
    print('wsi_pred: ', wsi_pred.shape)

    f_hovernet_sample = glob.glob(str(hovernet_path) + '/' + sample + '*.gz')[0]
    centroids_wsi_level = get_hovernet_centroids(f_hovernet_sample, level)
    print(radius, radius_level)

    ct_labels = celltyping_obj.get_wsi_celltyping(centroids_wsi_level, wsi_pred)
    labels, counts = np.unique(ct_labels, axis=0, return_counts=True)
    print(labels, counts)

    centroids_CD8 = get_celltype_centroids(celltype='Tcells.CD8', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)
    centroids_CD3 = get_celltype_centroids(celltype='Tcells.CD3', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)
    centroids_tumor = get_celltype_centroids(celltype='tumor', ct_labels=ct_labels, centroids_wsi_level=centroids_wsi_level)

    print('centroids CD8: ', len(centroids_CD8))
    print('centroids CD3: ', len(centroids_CD3))
    print('centroids tumor: ', len(centroids_tumor))

    annotation_path_sample = save_path_processed_annotations + '/' + sample + '_annotated_tumor_IM.npz'
    if not os.path.exists(save_path_processed_annotations + '/' + sample + '_annotated_tumor_IM.npz'):
        print(f"sample: {sample} does not have annotations")
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

    np.savez(os.path.join(save_path_analysis, sample + '-cell_coordinated.npz'), centroids_CD8=centroids_CD8, centroids_CD3=centroids_CD3, centroids_tumor=centroids_tumor, density_analysis=new_row)
