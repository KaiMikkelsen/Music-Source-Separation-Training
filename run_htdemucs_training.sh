#!/bin/bash

#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=3-00:00:00        # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

module load python/3.10 cuda/12.2 cudnn/8.9.5.29

# Load Java (ensure it's available)
echo "1" | module load java/21.0.1  # Automatically select Java 21.0.1

# Define directories
CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
CHECKPOINTS_PATH="checkpoints/${MODEL_TYPE}_${CURRENT_DATE}"
SLURM_LOGS_PATH="slurm_logs/${MODEL_TYPE}_${CURRENT_DATE}"
DATA_ZIP="moisesdb.zip"
SCRATCH_DIR="/scratch/kaim/data"

# Move zip file to scratch if not already there
echo "Moving data zip to scratch..."
if [ ! -f "$SCRATCH_DIR/$DATA_ZIP" ]; then
  cp "$HOME/projects/def-ichiro/kaim/data/$DATA_ZIP" "$SCRATCH_DIR/"
fi

# Extract data in scratch using jar
echo "Extracting data in scratch using jar..."
cd "$SCRATCH_DIR"
if [ ! -d "moisesdb" ]; then
  jar xvf "$DATA_ZIP"
fi

# Now proceed to run the training
echo "Running training script for model: $MODEL_TYPE"

source separation_env/bin/activate

python train_optuna.py \
  --model_type "$MODEL_TYPE" \
  --config_path "$CONFIG_PATH" \
  --results_path "$CHECKPOINTS_PATH" \
  --data_path "$SCRATCH_DIR/moisesdb/train" \
  --valid_path "$SCRATCH_DIR/moisesdb/validation" \
  --num_workers 4 \
  --start_check_point "" \
  --device_ids 0 \
  --metrics sdr l1_freq si_sdr neg_log_wmse aura_stft aura_mrstft bleedless fullness \
  --wandb_key 689bb384f0f7e0a9dbe275c4ba6458d13265990d