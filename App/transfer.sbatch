#!/bin/bash
#SBATCH --job-name=Convert   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=30:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/convert_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/convert_%j.err

scp -r /orange/idtrees-collab/OpenVisus/ bweinstein@tree.westus.cloudapp.azure.com:/datadrive/OpenVisus/