#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=3-00:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
SCRATCH_DIR=$SLURM_TMPDIR


# Variables
MODEL_TYPE="htdemucs"
CONFIG_PATH="configs/config_musdb18_htdemucs.yaml"
DATASET_NAME="MUSDB18HQ"
DATASET_ZIP="../data/$DATASET_NAME.zip"  # Specify the dataset ZIP name
SLURM_LOGS_PATH="slurm_logs/${MODEL_TYPE}_${CURRENT_DATE}"
CHECKPOINTS_PATH="checkpoints/${MODEL_TYPE}_${CURRENT_DATE}"

# Create necessary directories
#mkdir -p "$SCRATCH_DIR"
mkdir -p "$SLURM_LOGS_PATH"

# Redirect SLURM output dynamically
exec > >(tee -a "$SLURM_LOGS_PATH/slurm-${SLURM_JOB_ID}.out") 2>&1


# Activate the environment
source separation_env/bin/activate

echo "Running training script for model: $MODEL_TYPE with dataset at $DATA_PATH"

python train_optuna.py \
  --model_type "$MODEL_TYPE" \
  --config_path "$CONFIG_PATH" \
  --results_path "$CHECKPOINTS_PATH" \
  --data_path "$DATA_PATH/train" \
  --valid_path "$DATA_PATH/validation" \
  --metrics sdr l1_freq si_sdr neg_log_wmse aura_stft aura_mrstft bleedless fullness \
  --num_workers 4 \
  --start_check_point "" \
  --device_ids 0 \
  --wandb_key 689bb384f0f7e0a9dbe275c4ba6458d13265990d

# # Cleanup scratch directory
# echo "Cleaning up $SCRATCH_DIR"
# rm -rf "$SCRATCH_DIR"