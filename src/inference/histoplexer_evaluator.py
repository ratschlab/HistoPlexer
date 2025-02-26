import random
import os
import re
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from math import sqrt, ceil
from tqdm import tqdm
from copy import deepcopy
import glob

from pathlib import Path
import json 
from types import SimpleNamespace
from src.models.generator import unet_translator
import concurrent.futures
from src.dataset.dataset import make_dataset, InferenceDataset, EvalDataset
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

# args: checkpoint_path, src_folder, trg_folder, flag (inference, eval), mode test, device 'cuda:4'
class HistoplexerEval(): 
    def __init__(self, args):
        self.args = args
        self.checkpoint_path = args.checkpoint_path
        self.device = args.device

        # getting config file path from experiment 
        config_path = os.path.dirname(self.checkpoint_path) + '/config.json'        
        with open(config_path, "r") as f:
            self.config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
                
        # path for he rois
        src_folder = self.config.src_folder if not self.args.src_folder else self.args.src_folder
        self.src_paths = sorted(make_dataset(src_folder, args.mode, self.config.split))

        print(self.src_paths[0:2], len(self.src_paths))

        # initialize model in cpu
        self.model = unet_translator(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            use_high_res=self.config.use_high_res,
            use_multiscale=self.config.use_multiscale,
            ngf=self.config.ngf,
            depth=self.config.depth,
            encoder_padding=self.config.encoder_padding,
            decoder_padding=self.config.decoder_padding, 
            device="cpu"
        )

        # load model weights all in cpu
        self.checkpoint_name = os.path.basename(self.checkpoint_path).split('-')[1].split('.')[0]
        print(f"Checkpoint name: {self.checkpoint_name}")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['trans_ema_state_dict']) # trans_state_dict
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded!")

        # save path for predictions 
        self.save_path = os.path.join(os.path.dirname(self.checkpoint_path), args.mode + '_images', self.checkpoint_name)
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Save path: {self.save_path}")

        # run inference and save predictions 
        self.test_dataset = InferenceDataset(input_paths=self.src_paths)
        print("Test dataset created of size ", len(self.test_dataset))
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1, 
                                      shuffle=False,
                                      pin_memory=True, 
                                      num_workers=1, 
                                      drop_last=False)
        # if number of npy files in save_path is equal to number of src_paths, then skip inference
        if not len(glob.glob(self.save_path + '/*npy')) == len(self.src_paths):
            print("Running inference!")
            self.run_inference()
        else:
            print("Inference already done!")

        # get performance using eval metrics ---
        if args.measure_metrics:
            print("Evaluating metrics!")            
            self.eval()


    # TODO also write dataset class for inference -- make sure can input diff file formats -- npy, tif, png, jpg -- make sure all in same range -- to tensor
    # have a dummy he on github to test and viz -- outputs always in npy or also image -- flag for viz -- save figure with channels etc -- have a demo folder with notebook 
    # make sure inference also works for single image and also 1 marker -- have a dummy image on github to test
    
    def run_inference(self):
        for input_img, img_name, input_size in self.test_loader:
            input_img = input_img.to(self.device)
            print(input_img.shape, img_name, input_size)
            pred_imc = self.model(input_img)
            print(len(pred_imc))
            print(pred_imc[0].shape, pred_imc[1].shape, pred_imc[2].shape, pred_imc[-1].shape)

            if self.config.use_high_res:
                pred_shape = int(input_size // 2**2)
                print(input_size, pred_shape, pred_imc[-1].shape, pred_imc[-1].squeeze(0).shape)
                pred_imc = torchvision.transforms.CenterCrop([pred_shape, pred_shape])(pred_imc[-1].squeeze(0))

            else:
                pred_imc = torchvision.transforms.CenterCrop([input_size, input_size])(pred_imc[-1].squeeze(0))
            print(pred_imc.shape, pred_imc[0].shape, len(pred_imc))
            pred_imc = pred_imc.detach().cpu().numpy().transpose((1, 2, 0))
            np.save(os.path.join(self.save_path, img_name[0] + ".npy"), pred_imc)


    def eval(self):
        # TODO how to agg across markers and exps 

        # GT IMC 
        tgt_folder = self.config.tgt_folder if not self.args.tgt_folder else self.args.tgt_folder
        tgt_gt_paths = sorted(make_dataset(tgt_folder, self.args.mode, self.config.split))
        print(tgt_gt_paths[0:2], len(tgt_gt_paths))
        
        # pred IMC
        tgt_pred_paths = sorted(glob.glob(self.save_path + '/*npy'))
        print(tgt_pred_paths[0:2], len(tgt_pred_paths))
        
        # Metrics initialisation
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, gaussian_kernel=True, kernel_size=25, sigma=1).to(self.device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device) # psnr reduction none does not work, so looping though samples and channels
        rmse = RootMeanSquaredErrorUsingSlidingWindow(window_size=8).to(self.device)
        
        # dataset for evaluation
        eval_dataset = EvalDataset(tgt_pred_paths, tgt_gt_paths, len(self.config.markers))
        print("Eval dataset created of size ", len(eval_dataset))
        eval_loader = DataLoader(eval_dataset,
                                batch_size=8, 
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=8, 
                                drop_last=False)
        
        markers = self.config.markers
        msssim_dict = {marker: [] for marker in markers}
        psnr_dict = {marker: [] for marker in markers}
        rmse_dict = {marker: [] for marker in markers}

        all_metrics_list = []
        
        for _, batch in enumerate(eval_loader): # loop through batches
            print(batch['sample'], batch['imc_pred'].shape, batch['imc_gt'].shape)
            sample = batch['sample']
            imc_pred = batch['imc_pred'].to(self.device).float() # [8, 11, H, W]
            imc_gt = batch['imc_gt'].to(self.device).float() # [8, 11, H, W]
                
            for i in range(imc_pred.shape[1]): # loop through channels
                imc_pred_ = imc_pred[:, i, :, :].unsqueeze(1) # (8,H,W) --> (8,1,H,W)
                imc_gt_ =  imc_gt[:, i, :, :].unsqueeze(1)
                
                imc_pred_ = imc_pred_.repeat(1, 3, 1, 1) # [8, 1, H, W] --> [8, 3, H, W]
                imc_gt_ = imc_gt_.repeat(1, 3, 1, 1) 
                
                for j in range(len(imc_pred_)): # loop through samples
                    imc_pred__ = imc_pred_[j].unsqueeze(0) # [3,H,W] --> [1,3,H,W]
                    imc_gt__ =  imc_gt_[j].unsqueeze(0)
                    ms_ssim_score = round(ms_ssim(imc_pred__, imc_gt__).item(), 4)
                    psnr_score = round(psnr(imc_pred__, imc_gt__).item(), 4)
                    
                    # TODO: think if need to apply gaussian blurr or downsample imgs to compute rmse
                    rmse_score = round(rmse(imc_pred__, imc_gt__).item(), 4)
                    
                    print(markers[i], sample[j], ms_ssim_score, psnr_score, rmse_score)
                    all_metrics_list.append([markers[i], sample[j], ms_ssim_score, psnr_score, rmse_score])

                    # used later to get avg metrics over samples
                    msssim_dict[markers[i]].append(ms_ssim_score)
                    psnr_dict[markers[i]].append(psnr_score)
                    rmse_dict[markers[i]].append(rmse_score)

        # save path for eval 
        save_path_eval = os.path.join(os.path.dirname(self.checkpoint_path), self.args.mode + '_eval', self.checkpoint_name)
        os.makedirs(save_path_eval, exist_ok=True)
        print(f"Save path eval: {save_path_eval}")
                            
        # saving merics per sample per marker
        df_per_sample = pd.DataFrame(all_metrics_list)
        df_per_sample.columns = ['Marker', 'sample', 'MSSSIM', 'PSNR', 'RMSE']
        df_per_sample['dataset'] = self.args.mode
        df_per_sample['submission_id'] = self.checkpoint_path.split('/')[-2]
        df_per_sample.to_csv(os.path.join(save_path_eval, 'all_metrics_per_sample.csv'),  index=False) 

        # aggregate over samples metrics to pandas df 
        agg_msssim = {marker: round(sum(values)/len(values), 4) if values else 0 for marker, values in msssim_dict.items()}
        agg_psnr = {marker: round(sum(values)/len(values), 4) if values else 0 for marker, values in psnr_dict.items()}
        agg_rmse = {marker: round(sum(values)/len(values), 4) if values else 0 for marker, values in rmse_dict.items()}
        
        list_of_dicts = [agg_msssim, agg_psnr, agg_rmse]
        df_agg = pd.DataFrame(list_of_dicts, index=['MSSSIM', 'PSNR', 'RMSE']).T
        df_agg = df_agg.reset_index().rename(columns={'index': 'Marker'})
        df_agg['dataset'] = self.args.mode
        df_agg['submission_id'] = self.args.checkpoint_path.split('/')[-2]
        df_agg.to_csv(os.path.join(save_path_eval, 'all_metrics_agg.csv'),  index=False) 
        print("Metrics saved!")

# TODOS
# one number of an exp 
# flag to rewrite pred or not 
# flag to run metrics or not
# introduce more metrics 
# optimize code 