#!/bin/bash
#SBATCH --job-name=allometry   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/allometry_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/allometry_%j.err

export SLURM_TMPDIR=/orange/idtrees-collab/tmp/
export TMPDIR=/orange/idtrees-collab/tmp/

module load tensorflow/1.14.0

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/crowns/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda3/envs/crowns/lib/python3.7/site-packages/
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/crowns/lib/:${LD_LIBRARY_PATH}

cd /home/b.weinstein/NEON_crown_maps/

#comet debug
python allometry.py
