import argparse
import os
import numpy as np
import pandas as pd
import json
import glob
import cv2
from hta.stats import HTA
from shapely.geometry import Point


def load_metadata(meta_path):
    return pd.read_csv(meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]

def initialize_hta_file(wsi_celltyping_path):
    hta_save_path = os.path.join(wsi_celltyping_path, 'hta_samples_tumor-center.csv')
    hta_cols = ['sample', 'immune_subtype', 'HTA', 'HTA_pval']
    if not os.path.exists(hta_save_path):
        pd.DataFrame(columns=hta_cols).to_csv(hta_save_path, index=False)
    return hta_save_path, hta_cols

def load_coordinates(coords_path):
    coords = np.load(coords_path)
    return coords['centroids_tumor'], coords['centroids_CD8']

def prune_coordinates(coords, TC_img):
    center_points = [Point(center) for center in coords]
    non_zero_points = [point for point in center_points if TC_img[int(point.y), int(point.x)][0] != 0]
    return np.array([[int(point.x), int(point.y)] for point in non_zero_points])

def create_image(coords_tumor, coords_CD8):
    max_x = max(np.max(coords_tumor[:, 0]), np.max(coords_CD8[:, 0]))
    max_y = max(np.max(coords_tumor[:, 1]), np.max(coords_CD8[:, 1]))
    image = np.zeros((max_x + 1, max_y + 1, 2))
    for x, y in coords_tumor:
        image[x, y, 0] = 1
    for x, y in coords_CD8:
        image[x, y, 1] = 1
    return image.astype(int)

def resize_image(image, factor=10):
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

def get_hta_sample(sample, coords_path, meta, args, hta_save_path, hta_cols, region_size):
    immune_subtype = meta.loc[meta['tupro_id'] == sample, 'cd8_phenotype_revised'].values[0]
    print(immune_subtype, sample)

    coords_tumor, coords_CD8 = load_coordinates(coords_path)
    sample_annotation_path = os.path.join(args.saved_annotation_path, f"{sample}_annotated_tumor_IM.npz")
    TC_img = np.load(sample_annotation_path)['img_annots_tumor']

    coords_tumor = prune_coordinates(coords_tumor, TC_img)
    coords_CD8 = prune_coordinates(coords_CD8, TC_img)

    if len(coords_tumor) == 0 or len(coords_CD8) == 0:
        print('No cells in tumor center')
        pd.DataFrame([[sample, immune_subtype, 0, 0]], columns=hta_cols).to_csv(hta_save_path, mode='a', header=False)
        return

    image = create_image(coords_tumor, coords_CD8)
    image = resize_image(image)

    hta = HTA(image, region_size)
    hta_stat, hta_pval = hta.calc()
    print('HTA {:.2f},  p-val: {:.2e}'.format(hta_stat, hta_pval))

    pd.DataFrame([[sample, immune_subtype, round(hta_stat, 3), round(hta_pval, 5)]], columns=hta_cols).to_csv(hta_save_path, mode='a', header=False)
