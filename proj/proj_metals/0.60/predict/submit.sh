#!/bin/bash
#SBATCH -N 1                   # Number of nodes
#SBATCH -C gpu                  # Use GPU nodes
#SBATCH -q regular  #debug              # Use regular queue
##SBATCH -t 00:30:00            # Time limit
#SBATCH -t 04:30:00            # Time limit
#SBATCH -A m2997               # Your account
#SBATCH --gpus-per-node=1       # GPUs per node

module purge; module load PrgEnv-nvidia; module load openmpi; 
export PYTHONPATH=/global/homes/c/changzhi/changzhi/softwares/AdsNet:$PYTHONPATH; 
module load cudatoolkit/11.7

python predict.py
