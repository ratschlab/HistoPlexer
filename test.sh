#!/bin/bash

# Define the base paths
EXPERIMENTS_PATH="/raid/sonali/project_mvs/nmi_results/cycleGAN"
CHECKPOINTS_DIR="/home/sonali/github_code/Boqi/mvs_project/nmi_results/cycleGAN"
DATAROOT="/raid/sonali/project_mvs/data/tupro/binary_he_rois_test"
SCRIPT_PATH="test.py" 
GPU_ID=3
NETG="unet_256"
MODEL="test"
NGF=32

# Loop through experiment folders
for EXPERIMENT_DIR in "$EXPERIMENTS_PATH"/*; do
    if [ -d "$EXPERIMENT_DIR" ]; then
        # Extract folder name
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")

        # Determine output_nc
        if [[ "$EXPERIMENT_NAME" == *"all"* ]]; then
            OUTPUT_NC=11
        else
            OUTPUT_NC=1
        fi
	echo $OUTPUT_NC
	echo $EXPERIMENT_NAME

        # Set results directory
        RESULTS_DIR="$EXPERIMENT_DIR/results"
	echo $RESULTS_DIR

        # Run the Python script
        python "$SCRIPT_PATH" \
            --dataroot "$DATAROOT" \
            --name "$EXPERIMENT_NAME" \
            --model "$MODEL" \
            --netG "$NETG" \
            --checkpoints_dir "$CHECKPOINTS_DIR" \
            --results_dir "$RESULTS_DIR" \
            --gpu_ids "$GPU_ID" \
            --output_nc "$OUTPUT_NC" \
            --ngf "$NGF"

        echo "Completed: $EXPERIMENT_NAME"
    fi
done
