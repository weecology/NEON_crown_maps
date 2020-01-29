#!/usr/bin/env bash

#SBATCH -J dask-worker
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=19G
#SBATCH -t 24:00:00
#SBATCH --error=/home/b.weinstein/logs/dask-worker-%j.err
#SBATCH --account=ewhite
#SBATCH --output=/home/b.weinstein/logs/dask-worker-%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1

module load tensorflow/1.14.0

export PATH=${PATH}:/home/b.weinstein/miniconda/envs/crowns/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda/envs/crowns/lib/python3.7/site-packages/
cd /home/b.weinstein/NEON_crown_maps/

python available.py