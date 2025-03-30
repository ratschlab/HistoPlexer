# code modified public repo https://github.com/AIRMEC/im4MEC/blob/main/preprocess.py
# Used for extracting features from HE tiles at 10x (MIA paper showed 10x better than other resolutions) from pre-trained models 
import time
import cv2
import h5py
import numpy as np
import openslide
import torch
from PIL import ImageDraw
import matplotlib.pyplot as plt

from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.he_feats_utils import *

# -- resnet18 model
# python -m bin.he_feats --output_dir=$output_dir --input_slide="$svs_file" --checkpoint="/raid/sonali/project_mvs/downstream_tasks/tcga_prognosis/resnet18-f37072fd.pth" --tile_size=256 --out_size=224 --imagenet 
# -- MIA paper model Immune subtyping of melanoma whole slide images using multiple instance learning
# python -m bin.he_feats --output_dir=$output_dir --input_slide="$svs_file" --checkpoint="/raid/sonali/project_mvs/downstream_tasks/tcga_prognosis/tenpercent_resnet18.ckpt" --tile_size=256 --out_size=224 

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preprocessing script")
    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Feature extractor weights checkpoint",
    )
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Use imagenet pretrained weights instead of a custom feature extractor weights checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_size",
        help="Resize the square tile to this output size (in pixels).",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        help="which GPU device to use",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--downsample",
        help="Desired downsample for wsi.",
        type=float,
        default=4.01
    )

    args = parser.parse_args()

    # Open the slide for reading
    wsi = openslide.open_slide(args.input_slide)
    tile_level = wsi.get_best_level_for_downsample(args.downsample)
    resolution = round(float(wsi.properties[openslide.PROPERTY_NAME_MPP_X]), 2)
    if resolution < 0.27: 
        print(wsi.level_dimensions)
        print(wsi.level_downsamples)
        print(resolution)

        # Derive the slide ID from its name
        slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
        wip_file_path = os.path.join(args.output_dir, slide_id + "_wip.h5")
        output_file_path = os.path.join(args.output_dir, slide_id + "_features.h5")

        os.makedirs(args.output_dir, exist_ok=True)

        # Check if the _features output file already exist. If so, we terminate to avoid
        # overwriting it by accident. This also simplifies resuming bulk batch jobs.
        if os.path.exists(output_file_path):
            raise Exception(f"{output_file_path} already exists")
        
        # Decide on which slide level we want to base the segmentation
        seg_level = wsi.get_best_level_for_downsample(64)

        # Run the segmentation and  tiling procedure
        start_time = time.time()
        tissue_mask_scaled = create_tissue_mask(wsi, seg_level, tile_level) # at tile level
        filtered_tiles, tile_size_pix = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size, tile_level) # at level 0

        # Build a figure for quality control purposes, to check if the tiles are where we expect them.
        print('seg_level: ', seg_level)
        qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2)
        qc_img_target_width = 1920
        qc_img = qc_img.resize(
            (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
        )
        print(
            f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s"
        )

        # Save QC figure while keeping track of number of features/tiles used since RBG filtering is within DataLoader.
        qc_img_file_path = os.path.join(
            args.output_dir, f"{slide_id}_features_QC.png"
        )
        qc_img.save(qc_img_file_path)
        # save using matplotlib
        qc_img_file_path = os.path.join(
            args.output_dir, f"{slide_id}_QC.svg"
        )
        plt.imshow(qc_img)
        plt.axis('off')  # Hide axes
        plt.savefig(qc_img_file_path, bbox_inches='tight', dpi=300,  pad_inches = 0)
        plt.close()

        print(f"Saved QC image to {qc_img_file_path}")

        # Extract the rectangles, and compute the feature vectors
        model = load_encoder(
            checkpoint_file=args.checkpoint,
            use_imagenet_weights=args.imagenet,
            device=torch.device(args.device),
        )

        generator = extract_features(
            model,
            args.device,
            wsi,
            filtered_tiles,
            tile_level,
            tile_size_pix,
            args.workers,
            args.out_size,
            args.batch_size,
        )
        start_time = time.time()
        count_features = 0
        with h5py.File(wip_file_path, "w") as file:
            for i, (features, coords) in enumerate(generator):
                count_features += features.shape[0]
                write_to_h5(file, {"features": features, "coords": coords})
                print(
                    f"Processed batch {i}. Extracted features from {count_features}/{len(filtered_tiles)} tiles in {(time.time() - start_time):.2f}s."
                )

        # Rename the file containing the patches to ensure we can easily
        # distinguish incomplete bags of patches (due to e.g. errors) from complete ones in case a job fails.
        os.rename(wip_file_path, output_file_path)
        print(
            f"Finished extracting {count_features} features in {(time.time() - start_time):.2f}s"
        )
    else: 
        print('Features not extracted as image of lower resolution of:  ', resolution)