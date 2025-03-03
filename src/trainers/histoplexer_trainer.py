import random
import os
import re
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from math import sqrt, ceil
from tqdm import tqdm
from copy import deepcopy

from src.trainers.base_trainer import BaseTrainer
from src.models.generator import unet_translator
from src.models.discriminator import Discriminator
from src.models.patch_sampler import PatchSampleF
from src.models.vgg import VGG19
from src.utils.loss.gp_loss import GaussPyramidLoss
from src.utils.loss.nce_loss import PatchNCELoss
from src.utils.logger.tb_logger import TBLogger
from src.utils.logger.colormap import colormap


def infinite_loader(loader):
    """Generates an infinite stream of data from a given DataLoader.
    
    Args:
        loader (DataLoader): The DataLoader instance from which data batches are to be loaded.

    Yields:
        Any: A batch of data from the DataLoader. 
    """
    while True:
        for data in loader:
            yield data

class HistoplexerTrainer(BaseTrainer):
    def __init__(self,
                args,
                datasets):
        super(HistoplexerTrainer, self).__init__(args)
        
        self.logger = TBLogger(self.config.save_path) 
               
        self.protein2index = {protein: i for i, protein in enumerate(self.config.markers)} # dict key as marker name, value as index
        if self.config.channels:
            print(f"Train/val on selected proteins {[self.config.markers[i] for i in self.config.channels]}")
        else:
            print(f"Train/val on all proteins {self.config.markers}")
        
        print('train_dataset', len(datasets[0])) 
        self.train_loader = DataLoader(datasets[0],
                                batch_size=self.config.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=self.config.num_workers, 
                                drop_last=True)
        self.train_loader_iter = infinite_loader(self.train_loader)

        if self.config.val: 
            self.val_loader = DataLoader(datasets[1],
                                         batch_size=self.config.batch_size, 
                                         shuffle=False, 
                                         pin_memory=True, 
                                         num_workers=self.config.num_workers, 
                                         drop_last=False)
            self.val_loader_iter = infinite_loader(self.val_loader)

        # init models and weights 
        self.G = self._init_G() 
        self.D = self._init_D() 
        self.G_ema = deepcopy(self.G) 
        self.G_ema.requires_grad(False) 
        
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.config.lr_G, betas=(self.config.beta_0, self.config.beta_1)) 

        if self.config.w_R1 > 0: 
            self.lazy_c = self.config.r1_interval / (self.config.r1_interval + 1) if self.config.r1_interval > 1 else 1
            self.opt_D = optim.Adam(self.D.parameters(), lr=self.config.lr_D*self.lazy_c, betas=(self.config.beta_0**self.lazy_c, self.config.beta_1**self.lazy_c))
        else:
            self.opt_D = optim.Adam(self.D.parameters(), lr=self.config.lr_D, betas=(self.config.beta_0, self.config.beta_1)) # changed
    
        if self.config.blur_gt:
            self.spatial_denoise = torchvision.transforms.GaussianBlur(3, sigma=1)

        if self.config.use_multiscale:
            self.imc_sizes = [self.config.patch_size // (2 ** (2 * (j + 1))) for j in reversed(range(self.config.depth // 2 - 1))]

        self.L1loss = GaussPyramidLoss() if self.config.use_gp else torch.nn.L1Loss() # changed
            
        if self.config.w_ASP > 0: 
            # TODO: add more options for feature encoderss
            self.E = VGG19(vgg_path=self.config.vgg_path, device=self.device) if self.config.use_feat_enc else None
            self.F = PatchSampleF(device=self.device)
            self.F.init_model(
                dummy_input=self.E(torch.randn([self.config.batch_size, 1, self.config.patch_size, self.config.patch_size], device=self.device))
            )
            self.opt_F = optim.Adam(self.F.parameters(), lr=self.config.lr_F, betas=(self.config.beta_0, self.config.beta_1))
            self.NCEloss = PatchNCELoss(batch_size=self.config.batch_size, total_step=self.config.total_steps, n_step_decay=10000)
            
        self.latest_step = -1
        if self.config.resume_path is not None:
            self._resume_checkpoint()

    def _init_D(self):
        D = Discriminator(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            use_high_res=self.config.use_high_res,
            use_multiscale=self.config.use_multiscale,
            ngf=self.config.ngf,
            depth=self.config.depth, 
            device=self.device
        )
        D.init_weights(init_type=self.config.discriminator_init_type) # added init of weights
        return D
        
    def _init_G(self):
        G = unet_translator(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            use_high_res=self.config.use_high_res,
            use_multiscale=self.config.use_multiscale,
            ngf=self.config.ngf,
            depth=self.config.depth,
            encoder_padding=self.config.encoder_padding,
            decoder_padding=self.config.decoder_padding, 
            device=self.device, 
            extra_feature_size=self.config.fm_feature_size 
        )
        G.init_weights(init_type=self.config.encoder_init_type) # added init of weights
        return G


    def _G_step(self, step: int, src: torch.Tensor, tgt: torch.Tensor, src_feats: torch.Tensor):
        self.G.requires_grad(True)
        self.D.requires_grad(False)
        self.opt_G.zero_grad(set_to_none=True)
        
        real_imc = tgt
        fake_imcs = self.G(src, src_feats)

        if self.config.use_gp: # changed
            loss_L1 = self.config.w_GP * self.L1loss(fake_imcs[-1], real_imc)
        else:
            loss_L1 = self.config.w_L1 * self.L1loss(fake_imcs[-1], real_imc)
                        
        if self.config.w_ASP > 0:
            fake_imc_feats = []
            real_imc_feats = []
            if self.E:
                self.E.eval()
                with torch.no_grad():
                    for i in range(real_imc.shape[1]):
                        fake_imc_feats.append(self.E(fake_imcs[-1][:, i:i+1, :, :].repeat(1, 3, 1, 1))) # only calculate ASP loss for high-res output
                        real_imc_feats.append(self.E(real_imc[:, i:i+1, :, :].repeat(1, 3, 1, 1)))
            else:
                self.G.eval()
                with torch.no_grad():
                    for i in range(real_imc.shape[1]):
                        fake_imc_feats.append(self.G(fake_imcs[-1][:, i:i+1, :, :].repeat(1, 3, 1, 1), encode_only=True))
                        real_imc_feats.append(self.G(real_imc[:, i:i+1, :, :].repeat(1, 3, 1, 1), encode_only=True))
                self.G.train()
                        
            loss_ASP = self.config.w_ASP * self._get_asp_loss(real_imc_feats, fake_imc_feats, step)
        else:
            loss_ASP = 0.0
        
        if self.config.p_dis_add_noise:
            fake_imcs = [self._add_noise_prob(x, self.config.p_dis_add_noise) for x in fake_imcs]

        # translator loss 
        fake_score_maps = self.D((src, fake_imcs))
        fake_score_means = [fake_score_map.mean(dim=(1, 2, 3)) for fake_score_map in fake_score_maps]
        fake_score_mean = sum(fake_score_means)/len(fake_score_means)  
        
        real_label = torch.ones(fake_score_mean.size(), device=self.device)
        loss_G = 0.5 * torch.square(fake_score_mean - real_label).mean()
        
        loss_G_total = loss_G + loss_L1 + loss_ASP # NOTE: in old version we did 0.5 * loss_ASP

        loss_G_total.backward()
        self.opt_G.step()
        if self.config.w_ASP > 0:
            self.opt_F.step()

        # update G_ema
        with torch.no_grad():
            decay=0.9999 if step >= self.config.ema_warmup else 0 
            # lin. interpolate and update
            for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                p_ema.copy_(p.lerp(p_ema, decay))
            # copy buffers
            for (b_ema_name, b_ema), (b_name, b) in zip(self.G_ema.named_buffers(), self.G.named_buffers()):
                if "num_batches_tracked" in b_ema_name:
                    b_ema.copy_(b)
                else:
                    b_ema.copy_(b.lerp(b_ema, decay))
        
        return {
            "loss_G": loss_G.item(),
            "loss_L1": loss_L1.item() if loss_L1 != 0.0 else 0.0,
            "loss_ASP": loss_ASP.item() if loss_ASP != 0.0 else 0.0,
        }

    def _D_step(self, step: int, src: torch.Tensor, tgt: torch.Tensor, src_feats: torch.Tensor):
        self.G.requires_grad(False) 
        self.D.requires_grad(True) 
        self.opt_D.zero_grad(set_to_none=True) 
        
        real_imcs = []
        if self.config.blur_gt: 
            tgt = self.spatial_denoise(tgt)
        if self.config.use_multiscale:
            real_imcs.extend(self._resize_tensor(tgt, self.imc_sizes))
        real_imcs.append(tgt)
        
        with torch.no_grad():
            fake_imcs = self.G(src, src_feats)
            
        if self.config.p_dis_add_noise: 
            fake_imcs = [self._add_noise_prob(x, self.config.p_dis_add_noise) for x in fake_imcs]
            real_imcs = [self._add_noise_prob(x, self.config.p_dis_add_noise) for x in real_imcs]
        
        fake_score_maps = self.D((src, fake_imcs))  
        real_score_maps = self.D((src, real_imcs))
        
        # retrain batch dimension
        real_score_means = [real_score_map.mean(dim=(1, 2, 3))for real_score_map in real_score_maps]
        fake_score_means = [fake_score_map.mean(dim=(1, 2, 3))for fake_score_map in fake_score_maps]
        real_score_mean = sum(real_score_means)/len(real_score_means)
        fake_score_mean = sum(fake_score_means)/len(fake_score_means)  

        # LS-GAN loss, we choose a = 0 and b = c = 1 (shown to perform well and focus on generating realistic examples)
        real_label = torch.ones(real_score_mean.size(), device=self.device)
        
        loss_D = 0.5 * torch.square(real_score_mean - real_label).mean() + 0.5 * torch.square(fake_score_mean).mean()
        
        if self.config.w_R1 > 0 and (step % self.config.r1_interval == 0):
            loss_R1 = self.config.w_R1 * self._get_r1(src, real_imcs, gamma_0=self.config.r1_gamma, lazy_c=self.lazy_c)
        else:
            loss_R1 = 0.0
        
        loss_D_total = loss_D + loss_R1
        loss_D_total.backward()
        self.opt_D.step()
        
        return {
            "loss_D": loss_D.item(),
            "loss_R1": loss_R1.item() if loss_R1 != 0.0 else 0.0
        }, fake_score_means

    def train(self):
        self.G.train()
        self.D.train()
        if self.config.w_ASP > 0:
            self.F.train()
            
        step_G = 0
        step_D = 0

        for step in tqdm(range(self.latest_step + 1, self.config.total_steps), initial=self.latest_step + 1):
            batch = next(self.train_loader_iter) 
            
            he = batch["he_patch"].to(self.device) 
            imc = batch["imc_patch"].to(self.device) 
            
            if "fm_features" not in batch:
                feats_uni = None
            else:
                feats_uni = batch["fm_features"].to(self.device)
            
            losses_D, fake_score_means = self._D_step(step, he, imc, feats_uni) 
            if step_D % self.config.log_interval == 0:
                self.logger.run(func_name="log_scalars", metric_dict=losses_D, step=step_D)
            step_D += 1

            update_G = self._update_translator_bool(
                rule=self.config.update_rule, 
                dis_fake_loss=fake_score_means,
                update_interval=self.config.update_interval, 
                current_step=step
            )

            if update_G:
                losses_G = self._G_step(step, he, imc, feats_uni)
                if step_G % self.config.log_interval == 0:  
                    self.logger.run(func_name="log_scalars", metric_dict=losses_G, step=step_G)
                step_G += 1

            if self.config.val and ((step+1) % self.config.log_img_interval == 0):
                batch_val = next(self.val_loader_iter)
                self._log_sample(step=step, 
                                 src=batch_val["he_patch"].to(self.device), 
                                 tgt=batch_val["he_patch"].to(self.device), 
                                 sample=batch_val["sample"], src_feats=batch_val["fm_features"].to(self.device))
            
            
            if (step+1) % self.config.save_interval == 0:
                self._save_checkpoint(step+1, self.config.save_path)
        
        print("Training finished!")
        
        # save the final model
        print(f"Saving final checkpoint: step={step}...")
        self._save_checkpoint(step+1, self.config.save_path)
        print("Saving finished!")
        self.logger.close() # added

    def _log_sample(self, step: int, src: torch.Tensor, tgt: torch.Tensor, sample: str, src_feats: torch.Tensor):
        self.G.eval() # set G to eval mode
        log_idx = random.randint(0, len(src) - 1) # only log one sample within the batch
        with torch.no_grad():
            imc_curr = self.G(src, src_feats)[-1][log_idx]
            imc_ema = self.G_ema(src, src_feats)[-1][log_idx]
            imc_real = tgt[log_idx]
            he = src[log_idx]
            sample = sample[log_idx]
            self.logger.run(
                func_name="add_image", 
                tag=f"pred_IMC_G", 
                img_tensor=self._prepare_imc_for_log(
                    imc_curr, 
                    size=(self.config.vis_size, self.config.vis_size),
                    vmin=self.config.vis_vmin,
                    vmax=self.config.vis_vmax), 
                global_step=step
            )
            self.logger.run(
                func_name="add_image", 
                tag=f"pred_IMC_G_ema", 
                img_tensor=self._prepare_imc_for_log(
                    imc_ema, 
                    size=(self.config.vis_size, self.config.vis_size),
                    vmin=self.config.vis_vmin,
                    vmax=self.config.vis_vmax), 
                global_step=step
            )
            self.logger.run(
                func_name="add_image", 
                tag=f"real_IMC", 
                img_tensor=self._prepare_imc_for_log(
                    imc_real, 
                    size=(self.config.vis_size, self.config.vis_size),
                    vmin=self.config.vis_vmin,
                    vmax=self.config.vis_vmax), 
                global_step=step
            )
            self.logger.run(
                func_name="add_image", 
                tag=f"H&E", 
                img_tensor=self._prepare_he_for_log(he, size=(self.config.vis_size, self.config.vis_size)), 
                global_step=step
            )
        self.G.train() # set G back to train mode
            
    def _get_asp_loss(self, real_imc_feats, fake_imc_feats, step):
        total_asp_loss = []
        for feat_k, feat_q in zip(real_imc_feats, fake_imc_feats):
            total_asp_loss_per_channel = 0.0
            n_layers = len(feat_q)
            
            feat_k_pool, sample_ids = self.F(feat_k, 256, None)
            feat_q_pool, _ = self.F(feat_q, 256, sample_ids)
            
            for f_q, f_k in zip(feat_q_pool, feat_k_pool):
                loss = self.NCEloss(f_q, f_k, step)
                total_asp_loss_per_channel += loss.mean()

            total_asp_loss_per_channel /= n_layers
            total_asp_loss.append(total_asp_loss_per_channel)

        total_asp_loss = torch.tensor(total_asp_loss, device=self.device).mean()
        return total_asp_loss
    
    def _get_r1(self, src, tgt_list, gamma_0, lazy_c):
        """
        Compute the R1 regularization loss for the discriminator.

        Args:
            src (torch.Tensor): Tensor of high-resolution images.
            tgt_list (list of torch.Tensor): List of intermediate convolution layers.
            gamma_0 (float): Gamma value for R1 regularization.
            lazy_c (float): Coefficient for lazy regularization.

        Returns:
            torch.Tensor: The total R1 regularization loss.
        """
        total_r1_loss = 0
        
        src.detach().requires_grad_(True)
        tgt_list = [tgt.detach().requires_grad_(True) for tgt in tgt_list]
        
        score_maps = self.D((src, tgt_list))

        for i, score_map in enumerate(score_maps):
            tgt = tgt_list[len(tgt_list) - i - 1]
            bs = tgt.shape[0]
            img_size = tgt.shape[-1]
            r1_gamma = gamma_0 * (img_size ** 2 / bs)
            
            r1_grad = torch.autograd.grad(outputs=[score_map.sum()],
                                          inputs=[tgt],
                                          create_graph=True,
                                          only_inputs=True)
            r1_penalty = 0
            for grad in r1_grad:
                r1_penalty += grad.square().sum(dim=[1, 2, 3])
                r1_loss = r1_penalty.mean() * (r1_gamma / 2) * lazy_c

            total_r1_loss += r1_loss

        return total_r1_loss
    
    def _save_checkpoint(self, step, save_path): 
        """
        Saves the state dictionaries of models and optimizers to a checkpoint.

        Parameters:
        - step: The current training step, used to name the checkpoint file.
        - save_path: The directory where the checkpoint will be saved.
        """
        
        checkpoint = {
            'trans_state_dict': self.G.state_dict(),
            'trans_ema_state_dict': self.G_ema.state_dict(),
            'dis_state_dict': self.D.state_dict(),
            'trans_optimizer_state_dict': self.opt_G.state_dict(),
            'dis_optimizer_state_dict': self.opt_D.state_dict(),
        }

        # Optionally add opt_F if W_ASP > 0
        if hasattr(self, 'W_ASP') and self.W_ASP > 0 and hasattr(self, 'opt_F'):
            checkpoint['opt_F'] = self.opt_F.state_dict()

        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Construct the checkpoint file path
        checkpoint_path = os.path.join(save_path, f'checkpoint-step_{step}.pt')

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def _resume_checkpoint(self):
        
        folder_path = self.config.resume_path
        regex = re.compile(r'checkpoint-step_(\d+).pt')
        latest_checkpoint = None

        for filename in os.listdir(folder_path):
            match = regex.search(filename)
            if match:
                step = int(match.group(1))
                if step > self.latest_step:
                    self.latest_step = step
                    latest_checkpoint = filename

        if latest_checkpoint: 
            checkpoint_path = os.path.join(folder_path, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path) # TODO: see of need to add map_location to device 

            self.G.load_state_dict(checkpoint['trans_state_dict']) # changed from 'model_G' to 'trans_state_dict'
            self.G_ema.load_state_dict(checkpoint['trans_ema_state_dict']) # changed from 'model_G_ema' to 'trans_ema_state_dict'
            self.D.load_state_dict(checkpoint['dis_state_dict']) # changed from 'model_D' to 'dis_state_dict'
            self.opt_G.load_state_dict(checkpoint['trans_optimizer_state_dict']) # changed from 'opt_G' to 'trans_optimizer_state_dict'
            self.opt_D.load_state_dict(checkpoint['dis_optimizer_state_dict']) # changed from 'opt_D' to 'dis_optimizer_state_dict'
            if 'opt_F' in checkpoint and hasattr(self, 'opt_F'):
                self.opt_F.load_state_dict(checkpoint['opt_F'])

            print(f"Resumed training from step {self.latest_step}.")
        else:
            print("No checkpoint found to resume training.")
    
    @staticmethod
    def _prepare_imc_for_log(img, size=(256, 256), vmin=None, vmax=None):
        assert img.ndim == 3, f"Only support input of shape [C, H, W], but got {img.shape}."
        img = img.unsqueeze(dim=0) # [C, H, W] --> [1, C, H, W]
        n_ch = img.shape[1]
        img = torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)
        img = img.permute(1, 0, 2, 3)  # [1, C, H, W] --> [C, 1, H, W]
        img = colormap(img, vmin=vmin, vmax=vmax)
        img = torchvision.utils.make_grid(img, int(sqrt(n_ch)), ceil(n_ch / int(sqrt(n_ch))))
        return img
    
    @staticmethod
    def _prepare_he_for_log(img, size=(256, 256)):
        img = img.unsqueeze(0) if img.ndim == 3 else img
        assert img.ndim == 4, f"Image must have the shape of [B, C, H, W], but got {img.shape}"
        img = torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)
        img = torchvision.utils.make_grid(img, 1, 1)
        return img
    
    @staticmethod
    def _resize_tensor(tensor, output_size):
        """
        Resize a tensor to a given output size.

        Args:
            tensor (torch.Tensor): The tensor to be resized.
            output_size (int or list): The desired output size or list of sizes for sequential resizing.

        Returns:
            list[torch.Tensor]: List of resized tensors.
        """
        if not isinstance(output_size, list):
            output_size = [output_size]

        tensors_resized = []
        for osz in output_size:
            resized_tensor = torchvision.transforms.functional.resize(tensor, osz)
            tensors_resized.append(resized_tensor)

        return tensors_resized
    
    @staticmethod
    def _add_noise_prob(tensor, factor=0.1, p=0.5):
        """
        Add noise to a tensor with a given probability.

        Args:
            tensor (torch.Tensor): The tensor to add noise to.
            factor (float): Factor determining the amount of noise.
            p (float): Probability of adding noise.

        Returns:
            torch.Tensor: Tensor with added noise, or the original tensor.
        """
        if random.random() < p:
            tensor_noise = factor * torch.rand(tensor.size())
            if tensor.device != tensor_noise.device:
                tensor_noise = tensor_noise.to(tensor.device)
            tensor_w_noise = tensor + tensor_noise
        else:
            tensor_w_noise = tensor

        return tensor_w_noise
    
    @staticmethod
    def _update_translator_bool(rule='prob', dis_fake_loss=None, update_interval=1, current_step=0):
        """
        Determine whether to update the translator based on a specified rule and interval.

        Args:
            rule (str): Rule for updating. Options are 'always', 'prob', 'dis_loss', and 'interval'.
            dis_fake_loss (float, optional): Fake score mean value from discriminator (used only if rule=='dis_loss').
            update_interval (int): Specifies the interval for updating G relative to D. For example, an interval of 2 means
                                   update G every 2 iterations of updating D.
            current_step (int): The current training step or iteration number.

        Returns:
            bool: Indicator whether to update the translator.
        """
        assert rule in ['prob', 'dis_loss', 'interval'], f'Only support: prob, dis_loss and interval, but got {rule}.'
        if rule == 'prob':
            if_update = random.choice([True, True, False])
        elif rule == 'dis_loss':
            assert dis_fake_loss is not None, 'dis_fake_loss not provided!'
            if_update = dis_fake_loss < 0.5
        elif rule == 'interval':
            if_update = current_step % update_interval == 0

        return if_update
