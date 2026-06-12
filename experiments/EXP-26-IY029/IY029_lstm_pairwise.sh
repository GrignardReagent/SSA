#!/bin/bash

# Grid Engine options
#$ -N IY029_lstm_pairwise
#$ -o IY029_lstm_pairwise.o$JOB_ID
#$ -e IY029_lstm_pairwise.e$JOB_ID
#$ -cwd
#$ -l h_rt=47:59:59
#$ -q gpu
#$ -l gpu-mig=1
#$ -l h_rss=32G
#$ -m bea -M s1732775@ed.ac.uk

# Initialise environment modules
. /etc/profile.d/modules.sh
module load miniforge
module load cuda

# Activate env
conda activate stochastic_sim

# Run Python script and log output
python IY029_lstm_pairwise.py > IY029_lstm_pairwise.out 2>&1

# Deactivate after job is done
conda deactivate
