import numpy as np 
import os 
import glob 
import csv
import pandas as pd
import statistics


base_path = '/raid/sonali/project_mvs/nmi_results/cycleGAN'
markers_list = ["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"]
# save_path = '/raid/sonali/project_mvs/results/final_results/quatitative'
# os.makedirs(save_path, exist_ok=True)


keywords = {'cyclegan-MP': 'tupro_cyclegan_channels-all_seed'}
# keywords = {'ours-MP': '5GP+ASP_selected_snr_nature', 'pix2pix-MP': 'pix2pix_selected_snr_nature', 'pyramidp2p-MP': 'pyramidp2p_selected_snr_nature',
            # 'ours-SP': '5GP+ASP_selected_snr_nature_pseudo-multiplex', 'pix2pix-SP': 'pix2pix_selected_snr_nature_pseudo-multiplex', 'pyramidp2p-SP': 'pyramidp2p_selected_snr_nature_pseudo-multiplex'}
step = '495K'

list_csv = []

for exp_name in keywords.keys(): 
    print(exp_name)
    
    exps = glob.glob(os.path.join(base_path + '/*' + keywords[exp_name] + '*'))
    print(exps)
    print(len(exps))
 
    msssim = []
    psnr = []
    rmse = []

    for exp in exps:
        print(exp.split('/')[-1])
        df_exp = pd.read_csv(os.path.join(exp, 'test_eval', 'all_metrics_per_sample.csv'))
        df_exp = df_exp[~df_exp['PSNR'].isin([np.inf, -np.inf])]

        MSSSIM_mean = df_exp['MSSSIM'].mean()
        PSNR_mean = df_exp['PSNR'].mean()
        RMSE_mean = df_exp['RMSE'].mean()

        msssim.append(MSSSIM_mean)
        psnr.append(PSNR_mean)
        rmse.append(RMSE_mean)

        print(exp, MSSSIM_mean, PSNR_mean, RMSE_mean)

        print('msssim:', MSSSIM_mean)
        print('psnr:', PSNR_mean)
        print('rmse:', RMSE_mean)
        print('\n')
    msssim_mean = statistics.mean(msssim)
    msssim_stdev = statistics.stdev(msssim)
    psnr_mean = statistics.mean(psnr)
    psnr_stdev = statistics.stdev(psnr)
    rmse_mean = statistics.mean(rmse)
    rmse_stdev = statistics.stdev(rmse)
    
    print('msssim:', msssim_mean, msssim_stdev)
    print('psnr:', psnr_mean, psnr_stdev)
    print('rmse:', rmse_mean, rmse_stdev)
    list_csv.append([exp_name, msssim_mean, msssim_stdev, psnr_mean, psnr_stdev, rmse_mean, rmse_stdev])
    # list_csv.append([exp_name, msssim_mean, msssim_stdev, psnr_mean, psnr_stdev])
    print('\n')

