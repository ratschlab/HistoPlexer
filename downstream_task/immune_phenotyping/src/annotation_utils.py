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

def get_color_code():
    # color code for annotation masks 
    color_dict = {(255, 255, 255): "Whitespace", # white
                  (255, 255, 0): "Positive Lymphocytes", # yellow   
                  (0, 0, 255): "Stroma", # blue
                  (255, 0, 0): "Tumor", # red           
                  (0, 0, 0): "others" # pigment etc 
                 }
    color_code = [list(t) for t in list(color_dict.keys())]
    color_code = np.array(color_code).astype('double')
    return color_code

# ----- Functions for annotation masks -----
def map_colors(image, colors):
    kmeans = KMeans(n_clusters=len(get_color_code()), random_state=0)
    kmeans.fit(colors)
    kmeans.cluster_centers_ = colors
    
    pixels = image.reshape(-1, 3)
    cluster_assignment = kmeans.predict(pixels.astype('double'))    
    pixel_labels = list(map(lambda x: colors[x], cluster_assignment))
    mapped_image = np.array(pixel_labels).reshape(image.shape).astype(np.uint8)

    return mapped_image


def get_hovernet_centroids(f_hovernet_sample, level):  
    # load h&e cell centers from hovernet output  
    df_hovernet_simple = pd.read_csv(
        f_hovernet_sample,
        usecols=['centroid_x', 'centroid_y']
    )

    centroids_wsi = df_hovernet_simple[['centroid_x', 'centroid_y']].values.tolist()
    print(len(centroids_wsi), centroids_wsi[0])

    # hovernet cell centriods and radius for level 
    centroids_wsi_level = (np.array(centroids_wsi)//(2**(level))).astype(int)
    print(centroids_wsi[0])
    return centroids_wsi_level


def get_annotation_masks(annots_path):
    # loading annotation masks: tumor and stroma within tumor compartment
    # reading using tifffile 
    img_annots = tifffile.imread(annots_path, key=0)
    print(wsi_pred.shape, img_annots.shape)

    # downsampling annotations as otherwise color mapping is slow  
    img_annots = cv2.resize(img_annots, (0,0), fx=0.25, fy=0.25) 
    print(img_annots.shape)

    # map colors correctly to annotations 
    img_annots = map_colors(img_annots, get_color_code())
    print(img_annots.shape)

    colors, counts = np.unique(img_annots.reshape(-1, img_annots.shape[-1]), axis=0, return_counts=True)
    print('Unique colors and counts in annotated image: ', len(colors))#, colors, counts)
    assert len(colors)<=6, "more than 6 colors/annotations found in the image"

    # get tumor and stroma mask 
    mask_tumor = (np.all(img_annots == [255, 0, 0], axis=-1)).astype(np.uint8)
    mask_stroma = (np.all(img_annots == [0, 128, 0], axis=-1)).astype(np.uint8)
    mask_positive_lymphocytes = (np.all(img_annots == [255, 0, 255], axis=-1)).astype(np.uint8)
    mask_stroma = cv2.bitwise_or(mask_stroma, mask_positive_lymphocytes)
    print('masks: ', mask_tumor.shape, mask_stroma.shape)

    # make sure shape matches of annotations and wsi_pred 
    height, width = wsi_pred.shape[:2]
    mask_tumor = cv2.resize(mask_tumor, (width, height))
    mask_stroma = cv2.resize(mask_stroma, (width, height))

    print('masks: ', mask_tumor.shape, mask_stroma.shape)
    return mask_tumor, mask_stroma


# ---- Functions for annotation masks -----
def get_annotation_mask_HE(annot_path, pre_defined_colors, downsample):

    """
    For the input "annot_path", loads the tif file with region annotations at desired level "level_annots"
    Uses k-means to fix the annotation by mapping to nearest color from the pre_defined_colors. 
    Returns annotation image at the desired level and the fixed colors.
    """
         
    wsi_annots = openslide.OpenSlide(annot_path) # tif file 

    print(wsi_annots.level_dimensions)
    print(wsi_annots.level_downsamples)

    level = wsi_annots.get_best_level_for_downsample(downsample)
    print('level: ', level)
    print(wsi_annots.level_dimensions[level])

    img_annots = wsi_annots.read_region((0, 0), level, (wsi_annots.level_dimensions[level]))
    img_annots = np.array(img_annots.convert('RGB')).astype(np.uint8)

    print(img_annots.shape)

    # map colors correctly to annotations 
    img_annots = map_colors(img_annots, pre_defined_colors)

    colors, counts = np.unique(img_annots.reshape(-1, img_annots.shape[-1]), axis=0, return_counts=True)
    print('Unique colors and counts in annotated image: ', len(colors))#, colors, counts)
    assert len(colors)<=6, "more than 6 colors/annotations found in the image"

    print(colors, counts)

    return img_annots


def get_region_masks(img_annots, f_regions, downsample_factor=16, coutour_thickness=-1, plot=True): 

    mask_annotated = np.zeros((img_annots.shape), np.uint8)
    img_annots_tumor = np.zeros((img_annots.shape), np.uint8)
    img_annots_IM = np.zeros((img_annots.shape), np.uint8)

    tree = ET.parse(f_regions)
    Annotation = tree.findall('Annotation')
    # labels: TumorBorder is all of tumor; CoreTumor is TumorBorder minus 500um; InvasiveMargin is approx strip of 1mm from CoreTumor
    labels_dict = {"CoreTumor":(255,0,0), "InvasiveMargin":(0,255,0)}#, "TumorBorder":(0,0,255)}

    print(Annotation)

    for j in range(len(Annotation)):
        label = Annotation[j].get('Name')
        print('label: ', label)

        mask_inclusion = np.zeros((mask_annotated.shape), np.uint8)
        mask_exclusion = np.zeros((mask_annotated.shape), np.uint8)

        if label in labels_dict.keys(): 
            n_regions = len(Annotation[j].findall('Regions/Region'))

            for i in range(n_regions): 
                region = Annotation[j].findall('Regions/Region')[i]
                exclusion = region.get('NegativeROA')
                vertices = region.findall('Vertices/V')

                # get vertices for the region
                loc_temp = []
                for counter, x in enumerate(vertices):
                    loc_X = int(float(x.attrib['X']))
                    loc_Y = int(float(x.attrib['Y']))
                    loc_temp.append([loc_X, loc_Y])
                loc_temp = np.asarray(loc_temp)
                loc_temp = loc_temp / downsample_factor # just to plot the coordinates on a downsampled image
                loc_temp = loc_temp.astype(int)

                if int(exclusion)==1: 
                    mask_exclusion = cv2.drawContours(mask_exclusion, [loc_temp], 0, labels_dict[label], coutour_thickness)

                elif int(exclusion)!=1: 
                    mask_inclusion = cv2.drawContours(mask_inclusion, [loc_temp], 0, labels_dict[label], coutour_thickness)               

            # for label merge inclusion exclusion masks 
            mask_label = mask_inclusion
            mask_label[np.where(mask_inclusion==mask_exclusion)]=0
            mask_annotated = np.maximum(mask_annotated, mask_label)

            if label == 'CoreTumor': 
                mask_label[mask_label!=0] = 255
                img_annots_tumor = cv2.bitwise_and(img_annots, img_annots, mask=cv2.cvtColor(mask_label, cv2.COLOR_BGR2GRAY))
            elif label == 'InvasiveMargin': 
                mask_label[mask_label!=0] = 255
                img_annots_IM = cv2.bitwise_and(img_annots, img_annots, mask=cv2.cvtColor(mask_label, cv2.COLOR_BGR2GRAY))

    overlay = cv2.addWeighted(img_annots,0.3,mask_annotated,0.7,0)

    if plot: 
        # plotting img_annots_fixed and img_annots_ side by side 
        fig, ax = plt.subplots(1,5, figsize=(20,20))
        ax[0].imshow(img_annots/255)
        ax[1].imshow(mask_annotated/255)
        ax[2].imshow(overlay/255)
        ax[3].imshow(img_annots_tumor/255)
        ax[4].imshow(img_annots_IM/255)

        plt.show()

    return img_annots_tumor, img_annots_IM