#!/bin/bash
#SBATCH --job-name=Figures   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/figures_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/figures_%j.err

source activate crowns
cd /home/b.weinstein/NEON_crown_maps/

#comet debug
python figures.py