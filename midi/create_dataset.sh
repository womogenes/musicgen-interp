#!/bin/bash

#SBATCH --job-name=musicgen_dataset
#SBATCH -p mit_normal_gpu
#SBATCH --gres gpu:h200:1
#SBATCH --mincpus 8
#SBATCH --mem 32000
#SBATCH --time 6:00:00
#SBATCH --output=logs/dataset_creation_%j.out
#SBATCH --error=logs/dataset_creation_%j.err

cd /home/harinit9/musicgen-interp/midi
uv run make_verified_dataset_missing.py
