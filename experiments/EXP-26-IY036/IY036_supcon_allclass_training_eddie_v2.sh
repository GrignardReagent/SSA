#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY036_supcon_allclass_training_eddie_v2
#$ -o IY036_supcon_allclass_training_eddie_v2.o$JOB_ID
#$ -e IY036_supcon_allclass_training_eddie_v2.e$JOB_ID

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

# Timestamped so repeated resubmissions with different hyperparameters each
# keep their own log, rather than overwriting a single static .out file
# (the -o/-e files above are already unique per $JOB_ID, but they end up
# ~empty since this redirect diverts all of the script's actual output here).
# Exported so the Python script stamps its artifacts (checkpoint, history CSV,
# W&B run name) with this SAME timestamp instead of taking its own reading a
# few seconds later -- otherwise a log cannot be matched to its artifacts.
export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run Python script and log output
python IY036_supcon_allclass_training_eddie_v2.py > "IY036_supcon_allclass_training_eddie_v2_${RUN_TIMESTAMP}.out" 2>&1

# Deactivate after job is done
conda deactivate
