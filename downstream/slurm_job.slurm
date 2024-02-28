#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -J mustard
#SBATCH --ntasks-per-node=4
#SBATCH  -n 4
#SBATCH --gres=gpu:4
#SBATCH -o ./output/%J.out   
#SBATCH -e ./output/%J.err         


module load cuda/11.8
# module load cuda

bash run.sh
