#!/bin/bash
#SBATCH --job-name=Sampling   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/sampling_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/sampling_%j.err

source activate crowns
cd /home/b.weinstein/NEON_crown_maps/

#comet debug
python sampling.py