#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY017_simclr_training_2
#$ -o IY017_simclr_training_2.o$JOB_ID
#$ -e IY017_simclr_training_2.e$JOB_ID

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
module load cuda
module load anaconda

# Activate environment
conda activate stochastic_sim

# Run Python script and log output
python IY017_simclr_training_2.py > IY017_simclr_training_14.out 


# Deactivate after job is done
conda deactivate
