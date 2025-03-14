#!/bin/bash

# Define base paths and models
BASE_PATH="/raid/sonali/project_mvs/nmi_results"
EXPERIMENT_TYPES=("ours-FM" "ours-FM-virchow2" "ours-FM-uni2")  # List of experiment types
# EXPERIMENT_TYPES=("ours" "pix2pix" "pyramidp2p" "cycleGAN")  # List of experiment types

# Loop through each experiment type (pix2pix, ours, pyramidp2p, etc.)
for EXPERIMENT_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    EXPERIMENTS_PATH="$BASE_PATH/$EXPERIMENT_TYPE"

    # Loop through experiment folders in the specific type
    for EXPERIMENT_DIR in "$EXPERIMENTS_PATH"/*; do
        # Extract experiment name
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        echo "Running for: $EXPERIMENT_NAME"

        # Condition for "channels-all_seed-" → Run with checkpoint and --get_predictions
        if [[ "$EXPERIMENT_NAME" == *"channels-all"* ]]; then
            taskset -c 160-191 python -m downstream_task.cell_level.bin.pseudo_cell --experiment_type="$EXPERIMENT_TYPE" --experiment_name="$EXPERIMENT_NAME" --overwrite &

        fi 
    done 
done  
