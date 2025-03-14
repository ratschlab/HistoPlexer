#!/bin/bash

# Define base paths and models
BASE_PATH="/raid/sonali/project_mvs/nmi_results"

# List of experiment types
# EXPERIMENT_TYPES=("ours-FM")  
# EXPERIMENT_TYPES=("ours-FM" "ours-FM-virchow2" "ours-FM-uni2")  # List of experiment types
EXPERIMENT_TYPES=("ours-FM-virchow2" "ours-FM-uni2")
DEVICE="cuda:1"

# Loop through each experiment type (pix2pix, ours, pyramidp2p, etc.)
for EXPERIMENT_TYPE in "${EXPERIMENT_TYPES[@]}"; do

    EXPERIMENTS_PATH="$BASE_PATH/$EXPERIMENT_TYPE"

    # Loop through experiment folders in the specific type
    for EXPERIMENT_DIR in "$EXPERIMENTS_PATH"/*; do

        # Extract experiment name
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        echo "Running for: $EXPERIMENT_NAME"

        EMBEDDING_PART=$(echo "$EXPERIMENT_TYPE" | cut -d'-' -f3)  # Extract part after first hyphen
        TEST_EMBEDDINGS_PATH="/raid/sonali/project_mvs/data/tupro/he_rois_test/embeddings-${EMBEDDING_PART}.h5"
        echo "$TEST_EMBEDDINGS_PATH"

        if [[ "$EXPERIMENT_NAME" == *"pseudo"* ]]; then # pseudo multiplex
            SAVE_PATH="$EXPERIMENT_DIR/test_images/step_150000"
            echo "$SAVE_PATH"
            
            python -m bin.inference --device="$DEVICE" \
                                    --src_folder=/raid/sonali/project_mvs/data/tupro/he_rois_test/binary_he_rois_test \
                                    --tgt_folder=/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x \
                                    --save_path="$SAVE_PATH" \
                                    --markers CD16 CD20 CD3 CD31 CD8a gp100 HLA-ABC HLA-DR MelanA S100 SOX10

        elif [[ "$EXPERIMENT_NAME" != *"all"* ]]; then # singleplex
            echo "Running for singlplex: $EXPERIMENT_NAME"
            LATEST_CHECKPOINT=$(ls "$EXPERIMENT_DIR"/checkpoint-step_*.pt 2>/dev/null | \
                sed 's/.*checkpoint-step_\([0-9]\+\)\.pt/\1 \0/' | \
                sort -nr | head -n1 | cut -d' ' -f2)

            echo "$LATEST_CHECKPOINT"
            python -m bin.inference --checkpoint_path="$LATEST_CHECKPOINT"  \
                    --get_predictions \
                    --device="$DEVICE" \
                    --src_folder=/raid/sonali/project_mvs/data/tupro/he_rois_test/binary_he_rois_test \
                    --tgt_folder=/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x \
                    --measure_metrics=False  \
                    --test_embeddings_path="$TEST_EMBEDDINGS_PATH" 

        elif [[ "$EXPERIMENT_NAME" == *"all_"* ]]; then # multiplex
            LATEST_CHECKPOINT=$(ls "$EXPERIMENT_DIR"/checkpoint-step_*.pt 2>/dev/null | \
                sed 's/.*checkpoint-step_\([0-9]\+\)\.pt/\1 \0/' | \
                sort -nr | head -n1 | cut -d' ' -f2)

            # LATEST_CHECKPOINT="$EXPERIMENT_DIR/checkpoint-step_495000.pt"
            echo "$LATEST_CHECKPOINT"

            python -m bin.inference --checkpoint_path="$LATEST_CHECKPOINT"  \
                            --get_predictions \
                            --device="$DEVICE" \
                            --src_folder=/raid/sonali/project_mvs/data/tupro/he_rois_test/binary_he_rois_test \
                            --tgt_folder=/raid/sonali/project_mvs/data/tupro/binary_imc_processed_11x \
                            --test_embeddings_path="$TEST_EMBEDDINGS_PATH" 

        fi
    done  
done  
