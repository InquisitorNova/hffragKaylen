#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --time=24:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/physics/phujdj/anaconda3/lib/

source /home/physics/phujdj/anaconda3/envs/SpocFit/bin/activate

python3 Trainer.py