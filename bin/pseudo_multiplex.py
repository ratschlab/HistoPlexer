# the script is used to merge predictions from singleplex model to pseudo-multiplexed images which are later used for eval and co-expression analysis

import numpy as np
import os
import shutil
import json
import glob
import argparse

# python -m bin.pseudo_multiplex --method=cycleGAN --markers CD16 CD20 CD3 CD31 CD8a gp100 HLA-ABC HLA-DR MelanA S100 SOX10 --real_multiplex=tupro_cyclegan_channels-all_seed-2
# python -m bin.pseudo_multiplex --method=ours-FM --markers CD16 CD20 CD3 CD31 CD8a gp100 HLA-ABC HLA-DR MelanA S100 SOX10 --real_multiplex=tupro-patches_ours-FM_channels-all_seed-2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for HistoPlexer pseudo multiplex")
    
    parser.add_argument("--results_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/', help="Path to the results where all multiplex and singleplex experiments are stored")    
    parser.add_argument("--method", type=str, required=False, default='cycleGAN', help="name of the modes: ours, pix2pix, pyramidp2p, cycleGAN, ours-FM")    
    parser.add_argument("--real_multiplex", type=str, required=True, default=None, help="Name of the real multiplex experiment")    
    parser.add_argument("--markers", nargs="+", default=[], help="List of markers (optional), useful if running only evaluation")   
    parser.add_argument("--checkpoint_step", type=int, default=150000, help="checkpoint step for inference")   

    args = parser.parse_args()
    
    args.results_path = os.path.join(args.results_path, args.method)
    assert os.path.exists(args.results_path), f"Path {args.results_path} does not exist"
        
    real_multiplex = args.real_multiplex 
    pseudo_multiplex = real_multiplex.replace('all', 'all-pseudoplex')
    
    print('real_multiplex: ', real_multiplex)
    print('pseudo_multiplex: ', pseudo_multiplex)

    real_multiplex_path = os.path.join(args.results_path, real_multiplex)
    pseudo_multiplex_path = os.path.join(args.results_path, pseudo_multiplex)
    print('real_multiplex_path: ', real_multiplex_path)
    print('pseudo_multiplex_path: ', pseudo_multiplex_path)
    os.makedirs(pseudo_multiplex_path, exist_ok=True)

    # copying args/config.json file from real multiple to pseudo multiplex experiment
    if args.method == 'cycleGAN':
        source_args = os.path.join(real_multiplex_path, 'train_opt.txt')
        target_args = os.path.join(pseudo_multiplex_path, 'train_opt.txt')
    else: 
        source_args = os.path.join(real_multiplex_path, 'config.json')
        target_args = os.path.join(pseudo_multiplex_path, 'config.json')
    shutil.copy2(source_args, target_args)
    
    if args.method == 'cycleGAN':
        # Define the parameters to change
        changes = {
            "name": pseudo_multiplex,
        }
        updated_lines = []
        with open(target_args, "r") as file:
            for line in file:
                for key, new_value in changes.items():
                    if line.strip().startswith(key + ":"):
                        line = f"{key}: {new_value}\n"
                updated_lines.append(line)

        # Write the changes back to the file
        with open(target_args, "w") as file:
            file.writelines(updated_lines)
        markers = args.markers
        results_folder = 'results'
        save_path = os.path.join(pseudo_multiplex_path, results_folder)
        os.makedirs(save_path, exist_ok=True)
    else: 
        # change experiment name and change path
        with open(target_args, "r") as jsonFile:
            data = json.load(jsonFile)
        data["experiment_name"] = pseudo_multiplex
        data["save_path"] = data["save_path"].replace(real_multiplex, pseudo_multiplex)
        markers = data["markers"]   
        with open(target_args, 'w') as f:
            json.dump(data, f, indent=2)
        
        results_folder = 'test_images/step_' + str(args.checkpoint_step)
        save_path = os.path.join(pseudo_multiplex_path, results_folder)
        os.makedirs(save_path, exist_ok=True)

    # get paths of all relevant singleplex exps 
    exp_results = []
    for marker in markers: 
        exp_path = real_multiplex_path.replace("all", marker)
        assert os.path.exists(exp_path), f"Path {exp_path} does not exist"
        exp_result = glob.glob(exp_path + '/test_images/step_*')[-1]
        exp_results.append(exp_result)
    print(len(exp_results), exp_results)
    
    pred_paths = glob.glob(exp_results[0] + '/*.npy')
    print(len(pred_paths), pred_paths[0:2])

    for pred_path in pred_paths:
        roi = pred_path.split('/')[-1]
        print('roi: ', roi)
        roi_width = np.load(pred_path).shape[0]
        roi_multiplex = np.zeros((roi_width, roi_width, len(markers)), dtype=np.float32)
            
        for i, marker in enumerate(markers):
            print(exp_results[i], roi)
            pred = np.load(os.path.join(exp_results[i], roi))
            roi_multiplex[:,:,i] = pred[:,:,0]
        
        roi_multiplex = np.save(os.path.join(save_path, roi), roi_multiplex)
        print('saved: ', roi)