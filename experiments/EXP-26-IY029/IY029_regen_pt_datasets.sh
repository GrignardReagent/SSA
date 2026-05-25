#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY029_regen_pt_datasets
#$ -o IY029_regen_pt_datasets.o$JOB_ID
#$ -e IY029_regen_pt_datasets.e$JOB_ID

# Use the current working dir
#$ -cwd

# Max runtime limit (48h)
#$ -l h_rt=47:59:59

# Request 1 GPU
#$ -q gpu
#$ -l gpu=1

# Request 32G RAM per core (32G × 4 cores virtual memory)
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
python IY029_regen_pt_datasets.py > IY029_regen_pt_datasets.out 2>&1

# Deactivate after job is done
conda deactivate
