import openslide
import numpy as np
import json
import cv2
import scipy.ndimage
import argparse

from histoplexer.utils.constants import *
from histoplexer.utils.constants import *


def get_he_res(slide):
    ''' Simple convenience getter function.
    '''
    return float(slide.properties["openslide.mpp-x"])


def get_manually_aligned_he_region(align_result: dict, level: int, verbose=0):
    ''' Reads a certain image tile from an OpenSlide object and returns it as a numpy array,
        after applying the manual alignment transform stored in alig_result.
    align_result: Dictionary holding ROI info (WSI path, ROI coordinate json path, etc.) as well as alignment parameters, as produced by ManualAlignment.ipynb
    level: The level to read the image tile at
    verbose: Level of verbosity
    '''
    # grab original ROI coordinates
    roi_name_to_coords = get_rois_names_and_coords(align_result["json_file"])
    roi_coords = roi_name_to_coords[align_result["ROI"]]
    original_width_lvl0 = roi_coords[2][0] - roi_coords[0][0]
    # grab manual adjustments
    x_shift, y_shift = align_result["x_shift"], align_result["y_shift"]
    # open WSI slide
    slide = openslide.open_slide(align_result["he_slide"])
    # grab resolution
    resolution = slide.properties['openslide.mpp-x']

    # alignment is done at 1px per 1um, ROI coordinates are stored as level 0 pixel coordinates,
    # so we have to take resolution into account
    if align_result["he_slide"].endswith("ndpi"):
        roi_coords += (np.array([-x_shift, y_shift]).astype(float) * 2**2 * (1.0 / (4.0 * float(resolution)))).astype(int)
    elif align_result["he_slide"].endswith("tif"):
        roi_coords += (np.array([-y_shift, -x_shift]).astype(float) * 2**2 * (1.0 / (4.0 * float(resolution)))).astype(int)
    else:
        print("Warning: weird WSI format detected: ", align_result["he_slide"])

    # need a margin, so we can rotate and then cut out:
    margin_factor = 0.5
    margin = margin_factor * original_width_lvl0
    roi_coords[0] = roi_coords[0] + [-margin, -margin]
    roi_coords[1] = roi_coords[1] + [margin, -margin]
    roi_coords[2] = roi_coords[2] + [margin, margin]
    roi_coords[3] = roi_coords[3] + [-margin, margin]
    roi_coords[4] = roi_coords[4] + [-margin, -margin]

    margin_he_img = get_he_region(slide, level, roi_coords)

    rot_margin_he_img = scipy.ndimage.rotate(margin_he_img, 360.0 - align_result["angle"], reshape=False, mode='nearest')
    # cut out center square at original dimensions:
    original_width_chosen_lvl = int(round(original_width_lvl0 / 2**level))
    cut_start = int(round((rot_margin_he_img.shape[0] - original_width_chosen_lvl) / 2.0))
    final_he_img = rot_margin_he_img[cut_start: cut_start + original_width_chosen_lvl,
                                     cut_start: cut_start + original_width_chosen_lvl, :]

    if align_result["he_slide"].endswith("ndpi"):
        final_he_img = np.fliplr(final_he_img)
    elif align_result["he_slide"].endswith("tif"):
        final_he_img = np.fliplr(np.rot90(final_he_img))
    else:
        print("Warning: weird WSI format detected: ", align_result["he_slide"])

    return final_he_img


def get_rois_names_and_coords(roi_locations_path):
    ''' Returns ROI coordinates by name (as a dict).
    roi_locations_path: Absolute path to a json file containing ROI locations
    '''
    roi_coordinates = []
    roi_names = []
    with open(roi_locations_path) as f:
        data = json.load(f)
        rois = data['features']
        for i in range(len(rois)):
            roi_i = rois[i]
            roi_names.append(roi_i['properties']['name'])
            roi_coordinates.append(roi_i['geometry']['coordinates'][0])

    rois_name_to_coords = dict(zip(np.array(roi_names), np.array(roi_coordinates).astype(int)))

    return rois_name_to_coords


def get_he_region(slide, lvl_in_he: int, roi_coords, verbose=0):
    ''' Reads a certain image tile from an OpenSlide object and returns it as a numpy array.
    slide: OpenSlide object
    lvl_in_he: The level to read the image tile at
    roi_coords: The coordinates of the region to be read (has to be level 0 !)
    verbose: Level of verbosity
    '''
    roi_coords_in_lvl = (roi_coords / (2**lvl_in_he)).astype(int)
    
    width_imc_in_lvl = roi_coords_in_lvl[2][0] - roi_coords_in_lvl[0][0]
    height_imc_in_lvl = roi_coords_in_lvl[2][1] - roi_coords_in_lvl[0][1]

    if verbose >= 1:
        print(width_imc_in_lvl, height_imc_in_lvl)
    
    # *** IMP : need to provide the coordinates for read region in level 0, eg roi_location[0] for the highest level 0 
    he_roi = np.array(slide.read_region((roi_coords[0]), lvl_in_he, (width_imc_in_lvl, height_imc_in_lvl)).convert("RGB"))

    if verbose >= 1:
        print('he roi has shape ', he_roi.shape)
    
    return he_roi


def read_raw_protein_txt(imc_roi_txt_path: str, transformation=(lambda x: x), protein_subset=PROTEIN_LIST, cut_to_square=1000, verbose=0):
    ''' Reads a raw IMC measurement file and returns it as a numpy array. The order of channels is as defined in utils.constants.py
    imc_roi_txt_path: Absolute path to a raw IMC measurement txt file
    transformation: Function to apply to raw values, e.g. arcsinh with cofactor 1: (lambda x: math.log(x + math.sqrt(x**2 + 1)))
    protein_subset: A list of protein_names, like PROTEIN_LIST or ["MelanA", "SOX10"].
                    These channels will have valid data (granted that the raw files contains readouts for these proteins).
                    The rest of the channels will have zeros.
    cut_to_square: Sometimes the raw files contain values outside the expected area. This value defines the dimensions of the square output
    verbose: Level of verbosity
    '''
    assert imc_roi_txt_path.endswith(".txt"), "This function is only intended for raw text files !"

    lines = []
    with open(imc_roi_txt_path, 'r') as fh:
        lines = fh.readlines()
        fh.close()

    # first line is description:
    columns = [col.strip() for col in lines[0].split('\t')]
    lines = lines[1:]  # remove description

    for idx in range(len(columns)):
        # have to convert raw name, because the exact string sometimes does not match due to different metal tags
        # separation between protein name and metal tag can be brace or underscore
        if columns[idx] in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z']:
            continue

        underscore_pos = columns[idx].find('_')
        brace_pos = columns[idx].find('(')
        if brace_pos < 0:  # weird case, every actual measurement has at least one opening brace in its name, ignore
            continue
        else:
            if underscore_pos < 0:
                columns[idx] = columns[idx][:brace_pos]
            else:
                columns[idx] = columns[idx][:min(brace_pos, underscore_pos)]

    relevant_cols_idxs = []
    for idx, col in enumerate(columns):
        if col in prot_names_raw2deriv.keys() and prot_names_raw2deriv[col] in protein_subset:
            relevant_cols_idxs.append(idx)

    if len(relevant_cols_idxs) < len(protein_subset) and verbose >= 1:
        print("Warning: Looks like we are missing some proteins in file ", imc_roi_txt_path)

    if len(relevant_cols_idxs) > len(protein_subset):
        print("CRITICAL: Something went fundamentally wrong in read_raw_protein_txt !")
        quit(-1)

    result = np.zeros((cut_to_square, cut_to_square, len(protein_subset)))

    for strline in lines:
        # split long string into tab-separated smaller strings
        line = [entry.strip() for entry in strline.split('\t')]
        xidx = int(line[3])  # x and y have fixed position
        yidx = int(line[4])
        if xidx < cut_to_square and yidx < cut_to_square:
            relevant_vals = [transformation(float(line[col_idx])) for col_idx in relevant_cols_idxs]
            for val_idx, col_idx in enumerate(relevant_cols_idxs):
                prot_name_raw = columns[col_idx]
                prot_name_deriv = prot_names_raw2deriv[prot_name_raw]
                #prot_idx = protein2index[prot_name_deriv]  # index of the protein, as defined in constants.py
                protein2index = {prot: protein_subset.index(prot) for prot in protein_subset}
                prot_idx = protein2index[prot_name_deriv]
                # channel order for numpy / OpenCV is H-W-C (Y-X-Z)
                result[yidx, xidx, prot_idx] = relevant_vals[val_idx]

    # have to flip Y axis of image for some reason to match SCE readouts
    result = np.flipud(result)

    return result

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def apply_otsu_thresholding(imc_np_roi, sigma, return_blurred=False): 
    ''' Perform OTSU thresholding on a blurred image
    imc_np_roi: numpy array corresponding to all ROIs concatenated for a sample
    sigma: sigma of the Gaussian kernel used for blurring (if returne_blurred=False, then blurring used only for computation of the mask)
    return_blurred: whether to return an image with applied Gaussian blur
    returns thresholded image and the mask
    '''
    thresholded_img = imc_np_roi.copy()    
    otsu_mask = np.zeros(imc_np_roi.shape, dtype=np.uint8)
    
    for i in range(imc_np_roi.shape[2]): 
        # 1. Convert image to unit8 
        imc_np_roi_ = imc_np_roi[:,:,i].astype(np.float64) / np.max(imc_np_roi[:,:,i]) # normalize the data to 0 - 1
        imc_np_roi_unit8 = (255 * imc_np_roi_).astype(np.uint8) # Now scale by 255

        # 2. Blur the image with desired sigma (Gaussian blur) for improved threshold computation
        blurred = cv2.GaussianBlur(imc_np_roi_unit8,(sigma,sigma),0)
        # 3. Get mask and the threshold value
        _, otsu_mask[:,:,i] = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # 4. Apply mask on the image 
        if return_blurred:
            thresholded_img[:,:,i] = cv2.bitwise_and(blurred, blurred, mask = otsu_mask[:,:,i])
        else:
            thresholded_img[:,:,i] = cv2.bitwise_and(imc_np_roi[:,:,i],imc_np_roi[:,:,i], mask = otsu_mask[:,:,i])
        
    return thresholded_img, otsu_mask