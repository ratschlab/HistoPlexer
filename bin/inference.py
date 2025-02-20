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

# python -m bin.inference --checkpoint_path=/raid/sonali/project_mvs/nmi_results/tupro_ours_channels-all_seed-0/checkpoint-step_495000.pt --device=cuda:4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for HistoPlexer inference")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--src_folder", type=str, required=False, default=None, help="Path to source folder")
    parser.add_argument("--tgt_folder", type=str, required=False, default=None, help="Path to target folder")
    parser.add_argument("--mode", type=str, required=False, default='test', help="which data split to use")
    parser.add_argument("--device", type=str, required=False, default='cuda:4', help="device to use")
    parser.add_argument("--measure_metrics", type=str,required=False, default=True, help="If just inference or also evaluation using metrics")
    args = parser.parse_args()

    # load config file
    main(args)
    print("Inference!")