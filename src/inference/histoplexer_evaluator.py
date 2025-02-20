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

from pathlib import Path
import json 
from types import SimpleNamespace
from src.models.generator import unet_translator
import concurrent.futures
from src.dataset.dataset import make_dataset, InferenceDataset

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

        if args.measure_metrics:
            tgt_folder = self.config.tgt_folder if not self.args.tgt_folder else self.args.tgt_folder
            self.tgt_paths = sorted(make_dataset(tgt_folder, args.mode, self.config.split))
            print(self.tgt_paths[0:2], len(self.tgt_paths))

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
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['trans_ema_state_dict']) # trans_state_dict
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded!")

        # save path for predictions 
        self.save_path = os.path.join(os.path.dirname(self.checkpoint_path), args.mode + '_new_images')
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
        
        print("Running inference!")
        self.run_inference()

        # init metrics 

        # get performance using eval metrics 
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
        # TODO make sure the file name corresponds b/w src and target 
        pass
