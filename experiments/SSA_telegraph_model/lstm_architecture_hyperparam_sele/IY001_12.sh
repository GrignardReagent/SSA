#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY001_12

# Use the current working dir
#$ -cwd

# Max runtime limit (48h)
#$ -l h_rt=48:00:00

# Request 1 GPUs in the gpu queue
#$ -q gpu 
#$ -l gpu=1

# Request CPU cores
#$ -pe sharedmem 4

# Request 32G per core (32G × 4 cores virtual memory)
#$ -l h_vmem=32G

# Email notifications on job begin/end/abort
#$ -m bea -M s1732775@ed.ac.uk 

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda
module load anaconda

# Activate environment
conda activate stochastic_sim

# Run Python script and log output
python IY001_12.py > IY001A_12.out

# Deactivate after job is done
conda deactivate
