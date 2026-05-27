#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --output=result.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
####SBATCH --mem=480G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --account=suncat:normal
#SBATCH --gpus=1
###SBATCH --qos preemptable

#set >varmilan.txt

#source ~/.bashrc

export PYTHONPATH="$PYTHONPATH:/sdf/data/suncat/suncatlustre/changzhi/dataset/AdsNet"
conda activate /sdf/home/c/changzhi/softwares/anoconda3/envs/painn
conda info
python3 train.py 

#mpirun -n $SLURM_NPROCS /sdf/group/suncat/sw/electronicstructure/vasp-6.4.2/bin/vasp_std >run.log 2>run.err
#python dft.py > run.log 2>run.err
