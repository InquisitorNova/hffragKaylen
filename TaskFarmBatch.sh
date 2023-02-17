#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=4000
#SBATCH --time=48:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/physics/phujdj/anaconda3/lib/

source /home/physics/phujdj/anaconda3/envs/SpocFit/bin/activate

python3 /home/physics/phujdj/DeepLearningParticle/physics/SVParticleTransformerTrainer.py