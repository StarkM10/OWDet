#!/bin/bash
#SBATCH --gpus=1

cd /home/starkmar/Desktop/code/cc

module load anaconda/anaconda3-2022.10
module load cuda/11.1.0
# module load gcc-9
module load g++-9

source activate cc
# conda activate cc
nvidia-smi

# sbatch --gpus=1 -p gpu1 ./run.sh


# python cluster.py
python train.py
