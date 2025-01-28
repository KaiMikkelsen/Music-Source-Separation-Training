#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=3-00:00:00
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out

### Changes made to align with storage best practices ###
# 1. Use SLURM_TMPDIR for temporary data processing
# 2. Store permanent files in project space
# 3. Clean up scratch after job completion
# 4. Better path organization

module load python/3.10 cuda/12.2 cudnn/8.9.5.29

# Load Java - changed to more standard approach
module load java/21.0.1

# Define paths using environment variables
PROJECT_DIR="$HOME/projects/def-ichiro/kaim"
SCRATCH_DIR="/scratch/kaim"
DATA_ZIP="moisesdb.zip"
MODEL_TYPE="htdemucs"

# Create unique job ID-based directories
JOB_ID=${SLURM_JOB_ID:-$(date +%s)}
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints/${MODEL_TYPE}_${JOB_ID}"
LOGS_DIR="$PROJECT_DIR/slurm_logs"
DATA_SOURCE="$PROJECT_DIR/data/$DATA_ZIP"

# Set up directories
mkdir -p "$CHECKPOINTS_DIR" "$LOGS_DIR"

# Stage data in SLURM_TMPDIR (local node storage)
echo "Staging data in SLURM_TMPDIR..."
cp "$DATA_SOURCE" "$SLURM_TMPDIR/"

# Extract data in local storage
echo "Extracting dataset..."
unzip -q "$SLURM_TMPDIR/$DATA_ZIP" -d "$SLURM_TMPDIR"

# Run training from local storage
echo "Starting training..."
source "$PROJECT_DIR/separation_env/bin/activate"

python train_optuna.py \
  --model_type "$MODEL_TYPE" \
  --config_path "$PROJECT_DIR/$CONFIG_PATH" \
  --results_path "$CHECKPOINTS_DIR" \
  --data_path "$SLURM_TMPDIR/moisesdb/train" \
  --valid_path "$SLURM_TMPDIR/moisesdb/validation" \
  --num_workers 5 \  # Using 5 workers for 6 CPUs (1 for main process)
  --device_ids 0 \
  --metrics sdr l1_freq si_sdr neg_log_wmse aura_stft aura_mrstft bleedless fullness \
  --wandb_key 689bb384f0f7e0a9dbe275c4ba6458d13265990d

# Cleanup and final move of results
echo "Moving final results to project storage..."
cp -r "$CHECKPOINTS_DIR" "$PROJECT_DIR/checkpoints/"

# Optional: Clean scratch if used for any intermediate storage
# rm -rf "$SCRATCH_DIR/temp_${JOB_ID}"