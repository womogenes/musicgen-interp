#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 32
#SBATCH --mem 32000
#SBATCH --time 360

uv run save_data.py
