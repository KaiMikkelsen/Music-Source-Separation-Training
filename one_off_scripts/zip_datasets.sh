#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=4-00:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

zip -r /home/kaim/nearline/MUSDB18HQ_SPLITS.zip /home/kaim/projects/def-ichiro/kaim/data/MUSDB18HQ_SPLITS

zip -r /home/kaim/nearline/bleeding_splits.zip /home/kaim/scratch/bleeding_splits

zip -r /home/kaim/nearline/labelnoise_splits.zip /home/kaim/scratch/labelnoise_splits

zip -r /home/kaim/nearline/MOISESDB_SPLITS.zip /home/kaim/scratch/MOISESDB_SPLITS

zip -r /home/kaim/nearline/MOISESDB+MUSDB18HQ.zip /home/kaim/scratch/MOISESDB+MUSDB18HQ
