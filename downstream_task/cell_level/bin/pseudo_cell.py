import numpy as np
import pandas as pd
from skimage import draw
from pathlib import Path
import argparse
import os
import json
import glob

# python -m bin.pseudo_cell --experiment_type='ours' --experiment_name='tupro_ours_channels-all_seed-3' --overwrite

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate signal across pseudocells (circles around nuclei centroids).')
    parser.add_argument('--base_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results', help='Path where all results for project reside')
    parser.add_argument('--experiment_type', type=str, required=False, default='ours', help='experiment type eg ours-FM, ours, pix2pix, pyramidp2p, cycleGAN') 
    parser.add_argument('--experiment_name', type=str, required=False, default=None, help='experiment name')
    parser.add_argument('--data_set', type=str, required=False, default="test", help='Which set from split to use {test, train}')
    parser.add_argument('--input_path', type=str, required=False, default=None, help='Path to the predicted images')
    parser.add_argument('--he_nuclei_path', type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/hovernet/hovernet_nuclei-coordinates_all-samples.csv', 
                        help='path to csv file with XY coordinates of nuclei from hovernet')
    parser.add_argument("--overwrite", action="store_true", help='pass if overwrite existing output files')
    parser.add_argument('--radius', type=int, required=False, default=5, help='Radius of the circle to be drawn around the XY cell coordinates')
    parser.add_argument('--level', type=int, required=False, default=2, help='2 for tupro data, 0 for deepliif. Depends on the dataset and diff in resolution between HE and IMC')
    parser.add_argument('--markers_list', type=list, default=["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"], help='List of markers to use')
    parser.add_argument('--save_path', type=str, required=False, default=None, help='Path to save the results')

    args = parser.parse_args() 
    
    # get predicted data
    if not args.input_path:
        if args.experiment_type == 'cycleGAN':
            input_path = os.path.join(args.base_path, args.experiment_type, args.experiment_name, 'results')
            pred_paths = sorted(glob.glob(input_path + '/*.npy'))
        else: 
            input_path = sorted(glob.glob(os.path.join(args.base_path, args.experiment_type, args.experiment_name, 'test_images') + '/step*'))[-1]
            print('input_path: ', input_path)
            pred_paths = sorted(glob.glob(input_path + '/*.npy'))
    else:
        pred_paths = sorted(glob.glob(args.input_path + '/*.npy'))
    print('pred_paths: ', pred_paths[0], len(pred_paths))
    
    # save path 
    if not args.save_path:
        save_path = os.path.join(args.base_path, args.experiment_type, args.experiment_name, args.data_set + '_scdata')  
    else:
        save_path = args.save_path          
    os.makedirs(save_path, exist_ok=True)
    print('save_path: ', save_path)
    
    # load XY cell coordinates
    he_coords = pd.read_csv(args.he_nuclei_path)
    he_coords['X'] = he_coords['X']//(2**(args.level))
    he_coords['Y'] = he_coords['Y']//(2**(args.level))    
    radius = args.radius*(4//(2**(args.level)))
    
    for pred_path in pred_paths:
        roi_name = pred_path.split('/')[-1].split('.')[0]
        print('roi_name: ', roi_name)
        save_path_roi = os.path.join(save_path, roi_name+'.tsv')
        if os.path.exists(save_path_roi) and not args.overwrite:
            print('Already exists: ', save_path_roi)
            continue
        pred_imc = np.load(pred_path, mmap_mode='r')
        x_max, y_max, n_channels = pred_imc.shape
        df = he_coords.loc[he_coords.sample_roi==roi_name,:]
        
        sc_df = pd.DataFrame(index=df.index.to_list(), columns=markers)
        mask = np.zeros((x_max, y_max,1))
        for i in range(df.shape[0]):
            object_df = df.iloc[i,:]
            x0 = object_df['X']
            y0 = object_df['Y']
            rr, cc = draw.circle_perimeter(x0, y0, radius=radius, shape=mask.shape, method='andres')
            # circle contour
            mask[rr, cc,0] = i
            # fill the circle
            for x in range(abs(x0-radius),(x0+radius)):
                for y in range(abs(y0-radius),(y0+radius)):
                    if (((x-x0)**2 + (y-y0)**2) <= radius**2) and (x<x_max and y<y_max) and (x>0 and y >0):
                        mask[x,y,0] = i

        unq,ids,count = np.unique(mask.flatten(),return_inverse=True,return_counts=True)
        # average pixel signal within cells (per channel)
        for j,prot in enumerate(args.markers_list):
            try:
                out = np.column_stack((unq,np.bincount(ids,pred_imc[:,:,j].flatten())/count))
            except:
                import pdb; pdb.set_trace()
            mean_dict = dict(zip(out[:,0].astype(int), out[:,1]))
            for i in mean_dict.keys():
                sc_df.loc[df.index.to_list()[i],prot] = mean_dict[i]
        sc_df['sample_roi'] = roi_name
        sc_df = sc_df.merge(df.loc[:,['X','Y']], left_index=True, right_index=True, how='left')
        sc_df['radius'] = radius
        sc_df.to_csv(save_path_roi, sep='\t')
        print('Saved: ', roi_name)