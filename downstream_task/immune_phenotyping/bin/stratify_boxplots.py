import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, RocCurveDisplay
import statistics
from src.stratify_utils import *


def main():
    
    parser = argparse.ArgumentParser(description="Configurations for getting immune phenotyping stratification")
    parser.add_argument("--saved_path_analysis", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000/immune_phenotyping/rf_tumor_CD8', help="Path to where cell typing for immune phenotyping are saved")
    parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
    parser.add_argument("--immune_gt_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/CD8_quantification/20220209_TuProcohort_CD8.xlsx', help="Path to excel files with CD8 quantification")
    parser.add_argument("--gt_labels_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/CD8_quantification/immune_type-Marta.xlsx', help="Path to excel files with CD8 quantification")
    parser.add_argument("--annotation_meta_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/240418_TuPro_Mastersheet.xlsx', help="Including metadata for HALO annotations, eg quality of annotations")
    parser.add_argument("--save_path", type=str, required=False, default='/raid/sonali/project_mvs/results/final_results/immune_plots', help="Path where box plots and results are saved")
    args = parser.parse_args()

    df_merged = load_data(args)
    calculate_correlations(df_merged)

    phenotype_mapping = {'immune desert': 'desert', 'immune excluded': 'excluded', 'inflamed': 'inflamed'}
    df_merged['cd8_phenotype_revised'] = df_merged['cd8_phenotype_revised'].map(phenotype_mapping)
    plot_boxplots(df_merged, args.save_path, '3class')

    mapping_dict = {'desert': 'cold', 'excluded': 'cold', 'inflamed': 'hot'}
    df_merged['cd8_phenotype_revised_'] = df_merged['cd8_phenotype_revised'].map(mapping_dict)
    plot_boxplots(df_merged, args.save_path, '2class')

    # Run classification
    run_classification(df_merged)


if __name__ == "__main__":
    main()

