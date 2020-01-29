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

python /home/b.weinstein/NEON_crown_maps/available.py