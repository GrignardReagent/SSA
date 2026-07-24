#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY036_supcon_allclass_training_eddie
#$ -o IY036_supcon_allclass_training_eddie.o$JOB_ID
#$ -e IY036_supcon_allclass_training_eddie.e$JOB_ID

# Use the current working dir
#$ -cwd

# Max runtime limit (48h) - this must be listed, or the job won't run, max is 48h on Eddie.
#$ -l h_rt=47:59:59

# Request 1 GPU in the gpu queue
#$ -q gpu
#$ -l gpu-mig=1

# Request 32G per core (32G × 4 cores virtual memory)
#$ -l h_rss=32G

# Email notifications on job begin/end/abort
#$ -m bea -M s1732775@ed.ac.uk

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load miniforge
module load cuda

# Activate env (use conda instead of micromamba on Eddie)
conda activate stochastic_sim

# Run Python script and log output
python IY036_supcon_allclass_training_eddie.py > IY036_supcon_allclass_training_eddie.out 2>&1

# Deactivate after job is done
conda deactivate
