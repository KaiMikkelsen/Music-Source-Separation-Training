#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=3-00:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=create_4_stemmoisesdb.out  # Use Job ID for unique output files

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

source separation_env/bin/activate

python scripts/moises_to_musdb.py --src_dir /home/kaim/scratch/moisesdb/moisesdb_v0.1 --dest_dir /home/kaim/scratch/moises_4_stem --stems bass drums vocals --num_workers 4 --max_folders 240 
#python create_4_stem_moisesdb.py