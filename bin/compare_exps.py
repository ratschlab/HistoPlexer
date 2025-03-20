import numpy as np 
import os 
import glob 
import csv
import pandas as pd
import statistics
import argparse


parser = argparse.ArgumentParser(description='Comparing different methods for HistoPlexer using MSSSIM, PSNR, RMSE metrics')
parser.add_argument('--base_path', type=str, required=False, default='/raid/sonali/project_mvs/nmi_results', help='Path to the experiments directory')
parser.add_argument('--save_path', type=str, required=False, default= '/raid/sonali/project_mvs/nmi_results/final_results/quantitative', help='Path to where save the csv file with metrics from all experiments')
parser.add_argument('--markers_list', type=list, default=["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"], help='List of markers to use')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
# keywords to find all experiments for different baselines 
keywords = {'ours-FM-MP': 'tupro-patches_ours-FM_channels-all_seed-',
            'ours-FM-SP': 'tupro-patches_ours-FM_channels-all-pseudoplex_seed',
            'ours-MP': 'tupro_ours_channels-all_seed-',
            'ours-SP': 'tupro_ours_channels-all-pseudoplex_seed',
            'pyramidp2p-MP': 'tupro_pyramidp2p_channels-all_seed-',
            'pyramidp2p-SP': 'tupro_pyramidp2p_channels-all-pseudoplex_seed',
            'pix2pix-MP': 'tupro_pix2pix_channels-all_seed-',
            'pix2pix-SP': 'tupro_pix2pix_channels-all-pseudoplex_seed',
            'cyclegan-MP': 'tupro_cyclegan_channels-all_seed', 
            'cyclegan-SP': 'tupro_cyclegan_channels-all-pseudoplex_seed'
            }

list_csv = []

for exp_name in keywords.keys(): 
    print(exp_name)
    
    exps = glob.glob(os.path.join(args.base_path + '/*/*' + keywords[exp_name] + '*'))
    print(exps)
    print(len(exps))
 
    msssim = []
    psnr = []
    rmse = []

    for exp in exps:
        print(exp.split('/')[-1])
        
        if len(sorted(glob.glob(exp + '/test_eval' + '/*/' + 'all_metrics_per_sample.csv'))) == 0:
            continue
        
        metrics_path = sorted(glob.glob(exp + '/test_eval' + '/*/' + 'all_metrics_per_sample.csv'))[0]
        print(sorted(glob.glob(exp + '/test_eval' + '/*/' + 'all_metrics_per_sample.csv')))
        print(metrics_path)
        df_exp = pd.read_csv(metrics_path)
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
    # round to 3 decimal places
    
    if len(msssim) > 1: 
        msssim_mean = round(statistics.mean(msssim), 3)
        msssim_stdev = round(statistics.stdev(msssim), 3)
        psnr_mean = round(statistics.mean(psnr), 3)
        psnr_stdev = round(statistics.stdev(psnr), 3)
        rmse_mean = round(statistics.mean(rmse), 3)
        rmse_stdev = round(statistics.stdev(rmse), 3)
        
        print('msssim_:', msssim_mean, msssim_stdev)
        print('psnr_:', psnr_mean, psnr_stdev)
        print('rmse_:', rmse_mean, rmse_stdev)
        list_csv.append([exp_name, msssim_mean, msssim_stdev, psnr_mean, psnr_stdev, rmse_mean, rmse_stdev])
        print('\n')

# write to csv
df = pd.DataFrame(list_csv, columns=['exp_name', 'msssim_mean', 'msssim_stdev', 'psnr_mean', 'psnr_stdev', 'rmse_mean', 'rmse_stdev'])
df.to_csv(os.path.join(args.save_path,'results.csv'), index=False)
print(df)
print('done')