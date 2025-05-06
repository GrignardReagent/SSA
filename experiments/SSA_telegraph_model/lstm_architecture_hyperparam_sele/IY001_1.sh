#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY001_1

# Use the current working dir
#$ -cwd

# Runtime limit:
#$ -l h_rt=48:00:00

# Set working directory to the directory where the job is submitted from:
#$ -cwd

# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -l gpu=2

# Request system RAM 
# the total system RAM available to the job is the value specified here multiplied by the number of requested GPUs (above)
#$ -l h_vmem=32G

# Email me when job is done
#$ -m bea -M s1732775@ed.ac.uk 

# Initialise the environment modules
. /etc/profile.d/modules.sh

# You have to load anaconda so that you can use conda
module load anaconda

# ! You can only use conda ! 
# ! You can't use mamba on Eddie !
conda activate stochastic_sim

# Export python path so that the shell knows where the package is

python IY001_1.py > IY001A_1.out

# Deactivate after job is done
conda deactivate
