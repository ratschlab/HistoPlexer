import os
import numpy as np
import json
import random
import argparse
from pathlib import Path
from fractions import Fraction
from tqdm import tqdm
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from histoplexer.utils.constants import *
from histoplexer.utils.raw_utils import *
from histoplexer.training.training_helpers import *
from histoplexer.training.network import * 
from histoplexer.training.loaders import CGANDataset
from histoplexer.utils.eval_utils import get_protein_list
from histoplexer.utils.tb_logger import TBLogger
from histoplexer.utils.loss_utils import *
from histoplexer.utils.gauss_pyramid import Gauss_Pyramid_Conv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN model training')
    parser.add_argument('--project_path', type=str, required=True, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
    parser.add_argument('--data_path', type=str, required=True, default='/raid/sonali/project_mvs/data/tupro', help='path where all data for project reside')
    parser.add_argument('--submission_id', type=str, required=True, help='Job submission ID')
    parser.add_argument('--asymmetric', type=str2bool, required=False, default=True, help='Asymmertric or symmetric u-net generator, if IMC and H&E size same then symmetric setting')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Random seed')
    parser.add_argument('--n_step', type=int, required=False, default=500000, help='Number of steps to train')
    parser.add_argument('--cv_split', type=str, required=False, default='split3', help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--protein_set', type=str, required=False, default='full', help='which protein set to use')
    parser.add_argument('--continue_experiment', type=str, required=False, default=False, help='If we want to continue an experiment')
    parser.add_argument('--restore_model', type=str, required=False, default="NO_FILE", help='Model checkpoint dictionary to restore the model from')
    parser.add_argument('--imc_prep_seq', type=str, required=False, default='raw_clip99_arc_otsu3', help='Sequence of data preprocessing steps (needed to select form which folder to load the data)')
    parser.add_argument('--patch_size', type=int, required=False, default=256, help='Patch size')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size, works with single gpu for batch size upto 32')
    parser.add_argument('--save_every_x_epoch', type=int, required=False, default=20, help='Save logs every 1/x epoch.')
    parser.add_argument('--save_chkpt_every_x_epoch', type=int, required=False, default=2, help='Save model checkpoint every 1/x epoch.')
    parser.add_argument('--model_depth', type=int, required=False, default=6, help='depth of the translator and discriminator model, tested for depth 5 to 8')
    parser.add_argument('--factor_len_dataloader', type=float, required=False, default='8.0', help='factor for lenght of data loader')
    parser.add_argument('--weight_multiscale', type=str, required=False, default='1/3,1/3,1/3', help='weight of imc downsamples in multiscale realness score')
    parser.add_argument('--weight_L1', type=float, required=False, default=1.0, help='weight for L1 loss in translator loss term')
    parser.add_argument('--p_flip_jitter_hed_affine', type=str, required=False, default='0.5,0.5,0.5,0.5', help='probability for different augmentations. Order: flip/rot, he color jitter, hed color aug, he affine')
    parser.add_argument('--r1_interval', type=int, required=False, default=16, help='interval of calculating r1 loss')
    parser.add_argument('--log_interval', type=int, required=False, default=10, help='interval of logging loss and lr')
    parser.add_argument('--save_interval', type=int, required=False, default=5000, help='interval of saving state dict')
    parser.add_argument('--weight_ASP', type=float, required=False, default=1.0, help='weight of adaptive supervised patchNCE loss')
    parser.add_argument('--enable_RLW', type=str2bool, required=False, default=False, help='If true then use random task weighting')
    parser.add_argument('--weight_multiscale_L1', type=float, required=False, default=5.0, help='weight given to pyramid/GP loss')

    args = parser.parse_args()

    if args.continue_experiment:
        cluster = args.which_cluster
        n_step = args.n_step
        save_path = os.path.join(args.project_path, 'results', args.submission_id)

        with open(os.path.join(save_path, 'args.txt'), 'r') as file:
            args_dict = json.load(file)

        args = parser.parse_args(namespace=argparse.Namespace(**args_dict))
        args.which_cluster = cluster 
        args.n_step = n_step 

        if args.restore_model=="NO_FILE":
            pt_files = [file for file in os.listdir(save_path + '/tb_logs/') if file.endswith('_model_state.pt')]
            # Extract numeric parts from filenames and find the maximum
            if len(pt_files)!= 0: # checkpts found 
                max_saved_step = max(int(file.split('_')[0].split('K')[0]) for file in pt_files)
                print('max_saved_step: ', max_saved_step)
                args.restore_model = os.path.join(save_path, 'tb_logs', str(max_saved_step) +  'K_model_state.pt')
    
    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    # save path for logs, experiment results ...
    save_path = os.path.join(args.project_path, 'results', args.submission_id)
    surrounding_path = save_path
    print(f"Saving exps at:{surrounding_path}")
    
    # get current datetime up to seconds to create log folder
    datetime_full_iso = get_datetime()
    log_folder = os.path.join(surrounding_path, "tb_logs")
    print(f"Saving logs at:{log_folder}")
    if not os.path.exists(log_folder):
        Path(log_folder).mkdir(parents=True)
    tb_logger = TBLogger(log_dir=log_folder)
        
    # save argument values into a txt file
    with open(os.path.join(surrounding_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('Created output folders')
    
    # log parameters to tensorboard
    for arg_name, arg_val in vars(args).items():
        tb_logger.run(func_name="add_text", tag=arg_name, text_string=str(arg_val))
    
    protein_subset = get_protein_list(args.protein_set)
    
    # getting samples for desired train set
    train_aligns = get_aligns(args.project_path, cv_split=args.cv_split, protein_set=args.protein_set, aligns_set='train')
    print('Train set with '+str(len(train_aligns))+' ROIs')

    batch_size = args.batch_size
    patch_size = args.patch_size 
    assert patch_size==256, 'Atm patch size of 256 is the only implemented option.'
    
    # the size of the different IMC downsamples when multiscale setting
    imc_sizes = [] # eg: [patch_size//16, patch_size//4]
    for j in reversed(range(args.model_depth//2 -1)):
        imc_sizes.append(patch_size//(2**(2*(j+1))))
    weight_downsamples = [float(Fraction(x)) for x in args.weight_multiscale.split(',')]

    p_flip_jitter_hed_affine = list(map(float, args.p_flip_jitter_hed_affine.split(',')))
    train_ds = CGANDataset(args.project_path, align_results=train_aligns,
                        name="Train",
                        data_path=args.data_path,
                        patch_size=patch_size,
                        protein_subset=protein_subset,
                        imc_prep_seq=args.imc_prep_seq,
                        cv_split=args.cv_split,
                        standardize_imc=True,
                        scale01_imc=True,
                        factor_len_dataloader=args.factor_len_dataloader, 
                        which_HE='new', 
                        p_flip_jitter_hed_affine=p_flip_jitter_hed_affine, 
                        use_roi_weights = False)

    print('Loaded data')
    trainloader = DataLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=8, 
                             drop_last=True)
    trainloader_iter = enumerate(trainloader)
        
    print('HE_ROI_STORAGE: ', train_ds.HE_ROI_STORAGE)
    print('IMC_ROI_STORAGE: ', train_ds.IMC_ROI_STORAGE)
    
    # ----- create models -----
    trans = unet_translator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                            flag_asymmetric=args.asymmetric, flag_multiscale=True,
                            last_activation='relu',which_decoder='conv', encoder_padding=1, decoder_padding=1, eq_lr=False)   
    print(trans)
    # create ema model for translater 
    trans_ema = deepcopy(trans)
    trans_ema.to(device)
    freeze_params(trans_ema)
    trans_ema.eval()
    trans_opti = get_optimizer(trans, 'translator', 'fixedqeual')
    
    dis = Discriminator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                flag_asymmetric=args.asymmetric, flag_multiscale=True, mbdis=True, eq_lr=False)
    print(dis)

    lazy_c = args.r1_interval / (args.r1_interval + 1) if args.r1_interval > 1 else 1
    dis_opti = get_optimizer(dis, 'discriminator', 'fixedqeual', lazy_c)

    # ----- losses -----
    # L1 loss
    L1loss = torch.nn.L1Loss(reduction='none')
    
    # contrastive loss
    netF = None
    netF_opti = None
    if not np.isclose(args.weight_ASP, 0.0): 
        # initialise vgg19
        MODELDIR = os.path.join(args.project_path, PRETRAINED_DIR)
        vgg = VGG19(MODELDIR).to(device)
        freeze_params(vgg)
        vgg.eval()
             
        # initialise netF
        netF = PatchSampleF().to(device)
        dummy_feats = vgg(torch.zeros([args.batch_size, 1, 256, 256], device=device)) # use dummy data for netF initialisation
        _ = netF(dummy_feats, 256, None)
        patchNCELoss = PatchNCELoss(batch_size=args.batch_size, total_step=args.n_step, n_step_decay=10000).to(device)
        netF_opti = get_optimizer(netF, 'netF', 'fixedqeual')

    # Gaussian pyramid loss
    pyr_conv = Gauss_Pyramid_Conv(num_high=3)
    gp_weights = [0.0625, 0.125, 0.25, 1.0]

    # ----- restore model -----
    if args.restore_model == "NO_FILE":
        trans_lr_scheduler = get_lr_scheduler(trans_opti, 'fixedqeual')
        dis_lr_scheduler = get_lr_scheduler(dis_opti, 'fixedqeual')
        current_step = -1
    else:
        # restore model from checkpoint
        print('Restoring from a checkpoint.')
        checkpoint = torch.load(args.restore_model, map_location=device)
        trans.load_state_dict(checkpoint['trans_state_dict'])
        trans_opti.load_state_dict(checkpoint['trans_optimizer_state_dict'])
        optimizer_to(trans_opti, device)
        trans_lr_scheduler = checkpoint['trans_lr_scheduler']
        dis.load_state_dict(checkpoint['dis_state_dict'])
        dis_opti.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        optimizer_to(dis_opti, device)
        dis_lr_scheduler = checkpoint['dis_lr_scheduler']
        # epoch_number = checkpoint['step']+1
        current_step = int(str(max_saved_step)+'000')
        print('Model restored from epoch number: ', args.restore_model, current_step)
        
    trans.to(device)
    dis.to(device)
    
    unfreeze_params(dis)
    unfreeze_params(trans)
    trans.train()
    dis.train()
    
    # ----- training loop -----
    for step in tqdm(range(current_step+1, args.n_step)):
        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()
               
        lv2_real_imc = batch["imc_patch"].to(device)
        lv0_he_device = batch["he_patch"].to(device)
        
        # ----- updata discriminator -----                
        dis_opti.zero_grad(set_to_none=True)
        freeze_params(trans)
        with torch.no_grad():
            fake_imcs = trans(lv0_he_device)

        # prepare different resolutions of IMC and HE
        real_imcs = []
        with torch.no_grad():
            real_imcs.extend(resize_tensor(lv2_real_imc, imc_sizes))
            real_imcs.append(lv2_real_imc)
            
        # calculate scores 
        real_imcs = [real_imc.to(device) for real_imc in real_imcs]
        fake_score_maps = dis(lv0_he_device, fake_imcs)            
        real_score_maps = dis(lv0_he_device, real_imcs) 

        # retain batch dimension
        real_score_means = [real_score_map.mean(dim=(1, 2, 3))for real_score_map in real_score_maps]
        fake_score_means = [fake_score_map.mean(dim=(1, 2, 3))for fake_score_map in fake_score_maps]
        real_score_mean = sum(real_score_means)/len(real_score_means)
        fake_score_mean = sum(fake_score_means)/len(fake_score_means)     
        real_label = torch.ones(real_score_mean.size()).to(device)
            
        # r1 penalty loss
        if (step % args.r1_interval == 0):
            r1_loss = get_r1(dis, lv0_he_device, real_imcs, gamma_0=0.0002, lazy_c=lazy_c)
            tb_logger.run(func_name="log_scalars", metric_dict={"r1_loss": r1_loss.item()}, step=step)
        else:
            r1_loss = 0.0
        dis_loss = 0.5 * torch.square(real_score_mean - real_label).mean() + 0.5 * torch.square(fake_score_mean).mean()
        total_dis_loss = dis_loss + r1_loss
        total_dis_loss.backward()
        dis_opti.step()
        
        # ----- updata translator -----
        update_trans = update_translator_bool(rule='prob', dis_fake_loss=fake_score_mean)
        if update_trans:
            unfreeze_params(trans)
            trans_opti.zero_grad(set_to_none=True)
            if netF_opti is not None:
                netF_opti.zero_grad(set_to_none=True)
            fake_imcs = trans(lv0_he_device)
                            
            # calculate scores
            fake_score_maps = dis(lv0_he_device, fake_imcs)

            # retain batch dimension
            fake_score_means = [fake_score_map.mean(dim=(1, 2, 3)) for fake_score_map in fake_score_maps]
            fake_score_mean = sum(fake_score_means)/len(fake_score_means)            

            # define translator loss:
            trans_loss = 0.
            gen_loss = 0.5 * torch.square(fake_score_mean - real_label).mean()
            trans_loss += gen_loss
            
            n_channel = real_imcs[-1].shape[1] # num of markers
            batch_weight = F.softmax(torch.randn(n_channel), dim=-1).to(device) if args.enable_RLW else torch.ones(n_channel).to(device)
            
            if not np.isclose(args.weight_L1, 0.0): 
                loss_pyramid = [L1loss(pf, pr).mean(dim=(2, 3)) for pf, pr in zip(pyr_conv(fake_imcs[-1]), pyr_conv(real_imcs[-1]))]
                loss_pyramid = [l * w for l, w in zip(loss_pyramid, gp_weights)]
                loss_pyramid = [torch.mul(l, batch_weight).mean() for l in loss_pyramid]
                loss_GP = torch.mean(torch.stack(loss_pyramid))
                loss_GP *= args.weight_multiscale_L1
                if (step % args.log_interval == 0) & (step > 0):
                    tb_logger.run(func_name="log_scalars", metric_dict={"GP_loss": loss_GP.item()}, step=step)
                trans_loss += 0.5 * loss_GP
                    
            if not np.isclose(args.weight_ASP, 0.0):
                total_asp_loss = []
                # calculate ASP loss between real_B and fake_B
                for i in range(n_channel):
                    feat_real_B = vgg(real_imcs[-1][:, i:i+1, :, :])
                    feat_fake_B = vgg(fake_imcs[-1][:, i:i+1, :, :])
                    n_layers = len(feat_fake_B)
                    feat_q = feat_fake_B
                    feat_k = feat_real_B
                    feat_k_pool, sample_ids = netF(feat_k, 256, None)
                    feat_q_pool, _ = netF(feat_q, 256, sample_ids)
                    
                    total_asp_loss_per_channel = 0.0
                    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
                        loss = patchNCELoss(f_q, f_k, step)
                        total_asp_loss_per_channel += loss.mean()
                    total_asp_loss_per_channel /= n_layers
                    total_asp_loss.append(total_asp_loss_per_channel)
                
                total_asp_loss = torch.tensor(total_asp_loss, device=device)
                loss_asp = torch.mul(total_asp_loss, batch_weight).mean()
                loss_asp *= args.weight_ASP
                if (step % args.log_interval == 0) & (step > 0):
                    tb_logger.run(func_name="log_scalars", metric_dict={"ASP_loss": loss_asp.item()}, step=step)
                trans_loss += 0.5 * loss_asp
                        
            trans_loss.backward()
            trans_opti.step()
            
            if netF_opti is not None:
                netF_opti.step()
         
            # update trans_ema
            with torch.no_grad():
                decay=0.9999 if step >= 5000 else 0 
                # lin. interpolate and update
                for p_ema, p in zip(trans_ema.parameters(), trans.parameters()):
                    p_ema.copy_(p.lerp(p_ema, decay))
                # copy buffers
                for (b_ema_name, b_ema), (b_name, b) in zip(trans_ema.named_buffers(), trans.named_buffers()):
                    if "num_batches_tracked" in b_ema_name:
                        b_ema.copy_(b)
                    else:
                        b_ema.copy_(b.lerp(b_ema, decay))
                                
        # Record loss values every 10 steps
        if (step % args.log_interval == 0) & (step > 0):
            loss_dict = {
                "loss_dis": dis_loss.item(),
                "loss_trans": gen_loss.item(),
            }
            tb_logger.run(func_name="log_scalars", metric_dict=loss_dict, step=step)
            
            lr_dict = {
                "lr_dis": dis_opti.param_groups[0]['lr'],
                "lr_trans": trans_opti.param_groups[0]['lr']
            }
            tb_logger.run(func_name="log_scalars", metric_dict=lr_dict, step=step)
        
        dis_lr_scheduler.step()
        trans_lr_scheduler.step()
           
        # save checkpoint as dictionary every 5k steps
        if (step % args.save_interval == 0) & (step != 0):
            torch.save({'step': step,
                        'trans_state_dict':trans.state_dict(),
                        'trans_optimizer_state_dict':trans_opti.state_dict(),
                        'trans_lr_scheduler':'fixedqeual',
                        'trans_ema_state_dict':trans_ema.state_dict(),
                        'dis_state_dict':dis.state_dict(),
                        'dis_optimizer_state_dict':dis_opti.state_dict(),
                        'dis_lr_scheduler':dis_lr_scheduler,},
                        os.path.join(log_folder, f"{int(step/1000)}K_model_state.pt"))
            
            torch.save(trans_ema, os.path.join(log_folder, f"{int(step/1000)}K_translator.pth"))
            print('step: ', int(step/1000), 'K')