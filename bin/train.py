import argparse
import json
from pathlib import Path
import os 
import torch

from src.dataset.dataset import TuProDataset
from src.trainers.histoplexer_trainer import HistoplexerTrainer
from src.config.config import Config
from src.utils.misc import seed_everything


def main(args, device):
    """Run training or testing.

    Args:
        args: train/test configuration
        device: gpu or cpu device
    """
    seed_everything(seed=args.seed, device=device) # checked

    train_dataset = TuProDataset(
        split=args.split,
        mode='train',
        src_folder=args.src_folder,
        tgt_folder=args.tgt_folder,
        use_high_res=args.use_high_res,
        p_flip_jitter_hed_affine=args.p_flip_jitter_hed_affine,
        patch_size=args.patch_size,
        channels=args.channels, 
        cohort=args.cohort, 
        use_fm_features=args.use_fm_features, 
        fm_features_path=args.fm_features_path
        )

    datasets = [train_dataset]

    if args.val: 
        val_dataset = TuProDataset(
            split=args.split,
            mode='valid',
            src_folder=args.src_folder,
            tgt_folder=args.tgt_folder,
            use_high_res=args.use_high_res,
            p_flip_jitter_hed_affine=args.p_flip_jitter_hed_affine,
            patch_size=args.patch_size,
            channels=args.channels
        )
        datasets.append(val_dataset)
    
    print(f"Number of training images: {len(train_dataset)}")

    # initialize trainer
    trainer = HistoplexerTrainer(args=args, datasets=datasets)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for HistoPlexer")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    args = parser.parse_args()

    # load config file
    with open(args.config_path, "r") as ifile:
        config = Config(json.load(ifile))

    # experiment name 
    if config.resume_path == None: 
        channel_str = 'all' if config.channels is None else str(config.markers[config.channels[0]])
        config.experiment_name  = config.cohort + '_' + config.method + '_channels-' + channel_str  +'_seed-' + str(config.seed)
    else: 
        config.experiment_name = config.resume_path.split('/')[-1]
    
    print(config.experiment_name)
    
    config.save_path = os.path.join(config.base_save_path, config.experiment_name)
    print(f"save path: {config.save_path}")
    # create output folder if it doesn't exist yet
    Path(config.save_path).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in {config.save_path}")
    print(f"Using device: {config.device}")

    # save config file
    with open(os.path.join(config.save_path, "config.json"), "w") as ofile:
        json.dump(config.__dict__, ofile, indent=4)
    
    # # load config file
    # with open(args.config_path, "r") as ifile:
    #     config = Config(json.load(ifile))

    # run training or testing
    main(config, device=torch.device(config.device))
    print("Done!")
