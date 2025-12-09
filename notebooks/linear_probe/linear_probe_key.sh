#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 32
#SBATCH --mem 32000
#SBATCH --time 360

uv run linear_probe_key.py
