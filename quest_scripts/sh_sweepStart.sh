#!/bin/bash

#SBATCH -A p53239
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH --mem=10g

module purge all
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate spikesort

MY_ARG="$1"

python --version
python .../src/pipeline.py "$MY_ARG"
