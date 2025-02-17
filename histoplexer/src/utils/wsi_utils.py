import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import numpy as np
import cv2
import openslide
from shapely.geometry import Polygon, box
from shapely.affinity import scale
from shapely.ops import unary_union
from PIL import ImageDraw
import h5py
import gzip
import tqdm
import time 
import os
import pandas as pd

# functions 
def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(
        img_med, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours


def construct_tissue_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # If we have multiple polygons, we merge any overlap between them using unary_union().
    # This will result in a Polygon or MultiPolygon with most tissue masks.
    return unary_union(polys)


def make_tile_QC_fig(tile_sets, slide, level, line_width_pix):
    # Render the tiles on an image derived from the specified zoom level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    draw = ImageDraw.Draw(img, 'RGBA')
    colors = ['red', 'lightgreen']
    assert len(tile_sets) <= len(colors), 'define more colors'
    for tiles, color in zip(tile_sets, colors):
        for tile in tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline=color, width=line_width_pix)

    img = img.convert('RGB')
    return img


def create_tissue_mask(wsi,
                       seg_level,
                       min_rel_surface_area=500
                       ):
    # Determine the best level to determine the segmentation on
    level_dims = wsi.level_dimensions[seg_level]

    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))
    contours, hierarchy = segment_tissue(img)
    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)

    # Get the total surface area of the slide level that was used
    level_area = level_dims[0] * level_dims[1]

    # Minimum surface area of tissue polygons (in pixels)
    min_area = level_area / min_rel_surface_area

    tissue_mask = construct_tissue_polygon(
        foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0))

    return tissue_mask_scaled


def create_tiles_in_mask(wsi, tissue_mask_scaled, tile_size_pix, stride, padding=0):
    # Generate tiles covering the entire mask
    minx, miny, _, _ = tissue_mask_scaled.bounds

    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    maxx, maxy = wsi.level_dimensions[0]
    cols = range(int(minx), int(maxx-tile_size_pix), stride)
    rows = range(int(miny), int(maxy-tile_size_pix), stride)
    rects = []
    for x in cols:
        for y in rows:
            # (minx, miny, maxx, maxy)
            rect = box(
                x - padding,
                y - padding,
                x + tile_size_pix + padding,
                y + tile_size_pix + padding,
            )

            # Retain only the tiles that partially overlap with the tissue mask.
            if tissue_mask_scaled.intersects(rect):
                rects.append(rect)

    return rects


def crop_rect_from_slide(slide, rect):
    minx, miny, maxx, maxy = rect.bounds
    # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
    # but in the slide it is y = 0. Hence: miny instead of maxy.
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))


class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles, normalizer=None):
        self.wsi = wsi
        self.tiles = tiles

        self.stain_normalizer = normalizer
        self.to_tensor = transforms.ToTensor()
        self.to_byte = transforms.Lambda(lambda x: x*255)
        self.to_unit = transforms.Lambda(lambda x: x / 255.0)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = crop_rect_from_slide(self.wsi, tile)

        # Convert from RGBA to RGB
        img = img.convert('RGB')

        # Ensure we have a square tile in our hands.
        width, height = img.size
        assert width == height, 'input image is not a square'

        # Turn the PIL image into a (C x H x W) torch.FloatTensor (32 bit by default)
        img = self.to_tensor(img)

        if self.stain_normalizer:
            img = self.to_byte(img)
            img, _, _ = self.stain_normalizer.normalize(I=img, stains=False) # return shape: [H, W, C], range: [0, 255]
            img = img.permute(2, 0, 1) # [H, W, C] --> [C, H, W]
            img = self.to_unit(img)

        # img = img * 255 # to range 0 to 255 
        coords = np.array(tile.bounds).astype(np.int32)
        return img, coords

def infer_batch(batch_imgs, model, device):
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    with torch.no_grad():
        pred_dict = model(batch_imgs)
        # Restructure the tensor: move the 'values' to the last dimension.
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()]
             for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)
    return pred_output.cpu().numpy()


# ---------------

def create_dir(path): 
    if not os.path.exists(path):
        os.makedirs(path)
        
    
def get_chunks(wsi, seg_level, chunk_size, chunk_padding=0):
    tissue_mask = create_tissue_mask(wsi, seg_level)

    chunks = create_tiles_in_mask(
        wsi,
        tissue_mask,
        tile_size_pix=chunk_size,
        stride=chunk_size,
        padding=chunk_padding,
    )
    return tissue_mask, chunks

def get_qc_img(wsi, chunks, seg_level):

    # Build a figure for quality control purposes; to check if the tiles are where we expect them.
    qc_img = make_tile_QC_fig([chunks], wsi, seg_level, 1)
    qc_img_target_width = 1920
    qc_img = qc_img.resize((qc_img_target_width, int(
        qc_img.height / (qc_img.width / qc_img_target_width))))
    return qc_img

def get_wsi_inference(sample, wsi, seg_level, chunk_size, batch_size, n_proteins, loader_kwargs, dev0, network, save_path_imgs, normalizer): 

    # imc multiplex to be generated at 3 downsamples
    print(wsi.level_downsamples)
    print(wsi.level_dimensions)
    if 'TCGA' in sample: 
        level =  wsi.get_best_level_for_downsample(4.01)
    else: 
        level =  wsi.get_best_level_for_downsample(4)

    level_dims = wsi.level_dimensions[level]
    print(level, level_dims)
    
    imc_pred_level2 = np.zeros((level_dims[1],level_dims[0], n_proteins), dtype=np.float32)
    imc_pred_level4 = np.zeros((int(level_dims[1]//(2**2)),int(level_dims[0]//(2**2)), n_proteins), dtype=np.float32)
    imc_pred_level6 = np.zeros((int(level_dims[1]//(2**4)),int(level_dims[0]//(2**4)), n_proteins), dtype=np.float32)

    print(imc_pred_level2.shape, imc_pred_level4.shape, imc_pred_level6.shape)

    tissue_mask, chunks = get_chunks(wsi, seg_level, chunk_size) # get chunks
    qc_img = get_qc_img(wsi, chunks, seg_level) # get wsi with tiling for qc 
    
    # saving coordinates of tiles
    path_coords = save_path_imgs.joinpath('coords')
    create_dir(path_coords)
    pd.DataFrame([rect.bounds for rect in chunks], columns=['xmin', 'ymin', 'xmax', 'ymax']).to_csv(path_coords.joinpath(sample + 'coords.csv'), index=False)
    print(len(chunks))

    # saving qc image 
    path_qc = save_path_imgs.joinpath('QC')
    create_dir(path_qc)
    qc_img.save(path_qc.joinpath(sample + '_tile_QC.png')) 
        
    loader = DataLoader(
        dataset=BagOfTiles(wsi, chunks, normalizer),
        batch_size=batch_size,
        **loader_kwargs,
    )

    for batch_id, (batch, coord) in enumerate(loader):
        # predict for batch 
        with torch.no_grad():
            # batch = (batch/255.0).to(dev0)
            batch = batch.to(dev0)
            print(batch_id, batch.shape, batch[0,0,0,5])
            imc_batch = network(batch)

        coord = (coord.detach().cpu().numpy())#.astype(int)
        for i, c in enumerate(coord):
            if any(x<0 for x in c) == True:
                pass
            else:
                c_6 = c//(2**6)
                c_4 = c//(2**4)
                c_2 = c//(2**2)
                imc_pred_level6[c_6[1]: c_6[3], c_6[0]: c_6[2], :] = (torch.permute(imc_batch[0][i], (1, 2, 0))).detach().cpu().numpy()
                imc_pred_level4[c_4[1]: c_4[3], c_4[0]: c_4[2], :] = (torch.permute(imc_batch[1][i], (1, 2, 0))).detach().cpu().numpy()
                imc_pred_level2[c_2[1]: c_2[3], c_2[0]: c_2[2], :] = (torch.permute(imc_batch[2][i], (1, 2, 0))).detach().cpu().numpy()
                    
    # save the generated multiplex 
    path_level6 = save_path_imgs.joinpath('level_6')
    path_level4 = save_path_imgs.joinpath('level_4')
    path_level2 = save_path_imgs.joinpath('level_2')

    create_dir(path_level6)
    create_dir(path_level4)
    create_dir(path_level2)
    
    np.save(path_level6.joinpath(sample + '.npy'), imc_pred_level6)    
    np.save(path_level4.joinpath(sample + '.npy'), imc_pred_level4)    
    np.save(path_level2.joinpath(sample + '.npy'), imc_pred_level2)  
