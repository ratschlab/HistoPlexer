import argparse
import os
import numpy as np
import pandas as pd
import json
import glob
import cv2
from hta.stats import HTA
from shapely.geometry import Point
from src.hta_utils import load_metadata, get_hta_sample, initialize_hta_file

def main():
    parser = argparse.ArgumentParser(description="Configurations for getting HTA heterogeneity score")
    parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
    parser.add_argument("--wsi_celltyping_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000/immune_phenotyping/rf_tumor_CD8', help="Path to where cell typing for immune phenotyping are saved")
    parser.add_argument("--saved_annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/tumor_IM_annotations', help="Path to saved tumor IM npz files saved in immune phenotyping")
    args = parser.parse_args()

    meta = load_metadata(args.meta_path)
    hta_save_path, hta_cols = initialize_hta_file(args.wsi_celltyping_path)
    region_size = 100

    for coords_path in glob.glob(os.path.join(args.wsi_celltyping_path, '*-cell_coordinated.npz')):
        sample = os.path.basename(coords_path).split('-')[0]
        if sample in pd.read_csv(hta_save_path)['sample'].values:
            print(f"sample: {sample} already processed")
            continue
        get_hta_sample(sample, coords_path, meta, args, hta_save_path, hta_cols, region_size)

if __name__ == "__main__":
    main()