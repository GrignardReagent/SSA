#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IY039_simclr_hparam_sweep_worker
#$ -o IY039_simclr_hparam_sweep_worker.o$JOB_ID
#$ -e IY039_simclr_hparam_sweep_worker.e$JOB_ID

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

# SWEEP_TIMESTAMP/WORKER_ID/N_TRIALS are exported by IY039_submit_sweep.sh via
# `qsub -v` -- SWEEP_TIMESTAMP is the SAME across all parallel workers so they
# resolve to the SAME Optuna storage file/study_name and coordinate as one
# shared sweep (see the JournalStorage note in IY039_simclr_hparam_sweep.py).
# This is a one-per-launch, job-level id -- distinct from the per-trial
# timestamp each trial stamps its own checkpoint with, since one worker job
# runs many (N_TRIALS) trials, so a single id here can't identify any one of
# them the way RUN_TIMESTAMP does for this repo's single-run job scripts.
: "${SWEEP_TIMESTAMP:?SWEEP_TIMESTAMP must be set -- launch via IY039_submit_sweep.sh, not qsub directly}"
: "${WORKER_ID:?WORKER_ID must be set -- launch via IY039_submit_sweep.sh, not qsub directly}"
: "${N_TRIALS:=25}"

STORAGE_PATH="IY039_optuna_study_${SWEEP_TIMESTAMP}.journal"
STUDY_NAME="IY039_sweep_${SWEEP_TIMESTAMP}"

export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run Python worker and log output (one .out per worker, per Grid Engine job)
python IY039_simclr_hparam_sweep.py \
    --storage-path "${STORAGE_PATH}" \
    --study-name "${STUDY_NAME}" \
    --n-trials "${N_TRIALS}" \
    > "IY039_simclr_hparam_sweep_worker${WORKER_ID}_${RUN_TIMESTAMP}.out" 2>&1

# Deactivate after job is done
conda deactivate
