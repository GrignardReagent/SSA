#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY001_2

# Use the current working dir
#$ -cwd

# Max runtime limit (48h)
#$ -l h_rt=48:00:00

# Request 4 GPUs in the gpu queue (maximum allowed)
#$ -q gpu 
#$ -l gpu=2

# Request 16 CPU cores
#$ -pe sharedmem 16

# Request 128G per core (128G × 32 cores = 4TB virtual memory)
#$ -l h_vmem=128G

# Email notifications on job begin/end/abort
#$ -m bea -M s1732775@ed.ac.uk 

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda
module load anaconda

# Activate environment
conda activate stochastic_sim

# Run Python script and log output
python IY001_2.py > IY001A_2.out

# Deactivate after job is done
conda deactivate