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
from src.visualize_utils import *

def main():
    
    parser = argparse.ArgumentParser(description="Configurations for visualizing cell typing results on HE wsis")
    parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
    parser.add_argument("--wsi_celltyping_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000/immune_phenotyping/rf_tumor_CD8', help="Path to where cell typing for immune phenotyping are saved")
    parser.add_argument("--saved_annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/tumor_IM_annotations', help="Path to saved tumor IM npz files saved in immune phenotyping")
    parser.add_argument("--immune_gt_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/CD8_quantification/20220209_TuProcohort_CD8.xlsx', help="Path to excel files with CD8 quantification")
    parser.add_argument("--he_basepath", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/HE_new_wsi', help="Path to where HE wsi files are saved")
    parser.add_argument("--annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/regions_annotations', help="Path to tumor and invasive margin annotations on HE wsi")
    parser.add_argument("--save_path", type=str, required=False, default='/raid/sonali/project_mvs/results/final_results/immune_plots', help="Path where HE images with cell typing results are saved")
    args = parser.parse_args()

    df_gt = load_metadata(args.meta_path, args.immune_gt_path)
    samples_of_interest = ['MYGIFUD', 'MIDOBOL', 'MISYPUP', # inflamed
                           'MUGAHEK', 'MEWORAT', 'MOBICUN', 'MIDEKOG', 'MEFYFAK', 'MUBYJOF', # excluded 
                           'MEKAKYD'] # desert 
    
    print(df_gt.head())
    level = 4

    for sample_of_interest in samples_of_interest:
        print(sample_of_interest)
        coordinates_file = glob.glob(args.wsi_celltyping_path + '/' + sample_of_interest + '*cell_coordinated.npz')[0]
        sample = coordinates_file.split('/')[-1].split('-')[0]
        data = np.load(coordinates_file)
        sample_phenotype = df_gt.loc[df_gt['tupro_id'] == sample][['cd8_phenotype_revised', 'pathology_immune_diagnosis']].values[0]

        sample_HE_path = glob.glob(args.he_basepath + '/*' + sample + '*.ndpi')[0]
        he_img = load_he_image(sample_HE_path, level)

        region_annotation_path = glob.glob(str(args.annotation_path) + '/' + sample_of_interest + '*.annotations')[0]
        he_img = overlay_contours(he_img, region_annotation_path, level)

        sample_annotation_path = args.saved_annotation_path + '/' + sample + '_annotated_tumor_IM.npz'
        TC_img = np.load(sample_annotation_path)['img_annots_tumor']

        centroids_CD8_pruned = prune_centroids(data, TC_img, 'centroids_CD8')
        centroids_Tumor_pruned = prune_centroids(data, TC_img, 'centroids_tumor')

        he_img_CD8_centroids = draw_centroids(he_img.copy(), centroids_CD8_pruned, (57, 255, 50))
        he_img_CD8_tumor_centroids = draw_centroids(he_img_CD8_centroids.copy(), centroids_Tumor_pruned, (255, 0, 0))

        plots = {
            '1_HE': he_img,
            '2_regions': TC_img,
            '3_predCD8': he_img_CD8_centroids,
            '4_pred-CD8-tumor': he_img_CD8_tumor_centroids
        }
        save_plots(os.path.join(args.save_path, sample), sample, sample_phenotype, plots)


if __name__ == "__main__":
    main()

