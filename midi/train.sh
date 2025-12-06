#!/bin/bash

#SBATCH --job-name=musicgen_dataset
#SBATCH -p mit_normal_gpu
#SBATCH --gres gpu:l40s:1
#SBATCH --mincpus 4
#SBATCH --mem 32000
#SBATCH --time 12:00:00
#SBATCH --output=logs/dataset_creation_%j.out
#SBATCH --error=logs/dataset_creation_%j.err

uv run make_dataset_with_key.py
