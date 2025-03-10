import argparse
import json
from pathlib import Path
import os 

from src.dataset.dataset import TuProDataset
from src.inference.histoplexer_evaluator import HistoplexerEval


def main(args):
    """Run training or testing.

    Args:
        args: train/test configuration
        device: gpu or cpu device
    """
    # initialize trainer
    HistoplexerEval(args=args)

# # if running inference and eval for tupro/deepliif data: config from exp is used 
# python -m bin.inference --checkpoint_path=/raid/sonali/project_mvs/nmi_results/tupro_ours_channels-all_seed-0/checkpoint-step_495000.pt \
#                         --device=cuda:4 --get_predictions

# # if running eval only for cyclegan exps, inference is done using cyclegan script and exps doesn't save configs in the same format, therefore done in two steps                    
# python -m bin.inference --tgt_folder=/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x \
#                         --device=cuda:6 \
#                         --markers CD16 CD20 CD3 CD31 CD8a gp100 HLA-ABC HLA-DR MelanA S100 SOX10 \
#                         --save_path=/raid/sonali/project_mvs/nmi_results/cycleGAN/tupro_cyclegan_channels-all_seed-0/results \
#                         --split='/raid/sonali/project_mvs/meta/tupro/split3_train-test.csv'

# For FM 
# python -m bin.inference --checkpoint_path=/raid/sonali/project_mvs/nmi_results/ours-FM/tupro-patches_ours-FM_channels-all_seed-0/checkpoint-step_70000.pt \
#                         --get_predictions \
#                         --device=cuda:2 \
#                         --src_folder=/raid/sonali/project_mvs/data/tupro/he_rois_test/binary_he_rois_test \
#                         --test_embeddings_path='/raid/sonali/project_mvs/data/tupro/he_rois_test/embeddings-uni_v1.h5'\
#                         --tgt_folder=/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x 
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for HistoPlexer inference")
    parser.add_argument("--checkpoint_path", type=str, required=False, default=None, help="Path to checkpoint file")
    parser.add_argument("--src_folder", type=str, required=False, default=None, help="Path to source folder HE")
    parser.add_argument("--tgt_folder", type=str, required=False, default=None, help="Path to target folder GT IMC")
    parser.add_argument("--mode", type=str, required=False, default='test', help="which data split to use")
    parser.add_argument("--device", type=str, required=False, default='cuda:4', help="device to use")
    
    parser.add_argument("--measure_metrics", type=str, required=False, default=True, help="If perform evaluation")
    parser.add_argument("--save_path", type=str, required=False, default=None, help="Path to saved predictions")
    parser.add_argument("--markers", nargs="+", default=[], help="List of markers (optional), useful if running only evaluation")   
    parser.add_argument("--get_predictions", action="store_true", help="Enable prediction mode")
    parser.add_argument("--split", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/split3_train-test.csv', help="path to csv data split file")
    
    parser.add_argument("--test_embeddings_path", type=str, required=False, default=None, help="Path to h5 files with embeddings from foundation model")

    args = parser.parse_args()
    
    print(args.get_predictions)
    print(args.markers)
    print(args.save_path)

    # load config file
    main(args)
    print("Inference!")