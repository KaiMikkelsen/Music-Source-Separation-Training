#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=0-00:120:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro


cd ../data
DATASET_NAME="MUSDB18HQ"
DATASET_ZIP="$DATASET_NAME.zip" # Specify the dataset ZIP name
SCRATCH_DIR="/home/kaim/scratch"


    # Move and unzip dataset to scratch directory
    echo "Moving $DATASET_ZIP to $SCRATCH_DIR for faster access"
    cp "$DATASET_ZIP" "$SCRATCH_DIR"

    DATASET_ZIP_BASENAME=$(basename "$DATASET_ZIP")
    SCRATCH_ZIP="$SCRATCH_DIR/$DATASET_ZIP_BASENAME"

    mkdir -p "$SCRATCH_DIR/$DATASET_NAME"
    echo "created directory $SCRATCH_DIR/$DATASET_NAME"
    echo "Unzipping dataset in $SCRATCH_DIR/$DATASET_NAME"
    unzip -q "$SCRATCH_ZIP"

    # if ! unzip -q "$SCRATCH_ZIP" -d "$SCRATCH_DIR/$DATASET_NAME"; then
    #     echo "zip failed"
    # fi