#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY029_simclr_svm_pairwise
#$ -o IY029_simclr_svm_pairwise.o$JOB_ID
#$ -e IY029_simclr_svm_pairwise.e$JOB_ID

# Use the current working directory
#$ -cwd

# Max runtime (48h)
#$ -l h_rt=47:59:59

# Request 1 GPU in the gpu queue
#$ -q gpu
#$ -l gpu-mig=1

# 32G per core (sufficient for loading all 4 datasets + SVM fitting)
#$ -l h_rss=32G

# Email notifications on begin / end / abort
#$ -m bea -M s1732775@ed.ac.uk

# Initialise environment modules
. /etc/profile.d/modules.sh
module load miniforge
module load cuda

# Activate conda environment
conda activate stochastic_sim

# Run the evaluation script; all output goes to .out file for clean log
python IY029_simclr_svm_pairwise.py > IY029_simclr_svm_pairwise.out 2>&1

# Deactivate after job is done
conda deactivate
