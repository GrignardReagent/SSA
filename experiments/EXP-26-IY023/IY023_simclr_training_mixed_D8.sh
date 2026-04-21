#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY023_simclr_training_mixed_D8
#$ -o IY023_simclr_training_mixed_D8.o$JOB_ID
#$ -e IY023_simclr_training_mixed_D8.e$JOB_ID

# Use the current working dir
#$ -cwd

# Max runtime limit (48h)
#$ -l h_rt=47:59:59

# Request 1 GPUs in the gpu queue
#$ -q gpu 
#$ -l gpu=1

# Request 32G per core (32G × 4 cores virtual memory)
#$ -l h_rss=32G

# Email notifications on job begin/end/abort
#$ -m bea -M s1732775@ed.ac.uk 

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load miniforge
module load cuda

# Activate env
conda activate stochastic_sim

# Run Python script and log output
python IY023_simclr_training_mixed_D8.py > IY023_simclr_training_mixed_D8.out 2>&1

# Deactivate after job is done
conda deactivate
