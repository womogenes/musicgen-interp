#!/bin/bash

#SBATCH --job-name=diffrhythm_dataset
#SBATCH -p mit_normal_gpu
#SBATCH --gres gpu:h200:1
#SBATCH --mincpus 8
#SBATCH --mem 32000
#SBATCH --time 6:00:00
#SBATCH --output=logs/dataset_creation_%j.out
#SBATCH --error=logs/dataset_creation_%j.err

# Run from diffrhythm_interp where pyproject.toml lives
module load ffmpeg
cd /home/wyf/musicgen-interp/diffrhythm_interp
uv run scripts/make_dataset.py "$@"
