#!/bin/bash
# Launches N_WORKERS parallel Eddie jobs that share ONE Optuna study (via a
# shared JournalStorage file), each running N_TRIALS_PER_WORKER trials -- the
# whole sweep totals N_WORKERS * N_TRIALS_PER_WORKER trials (~100 by default:
# 4 workers x 25 trials). All workers must resolve to the SAME storage
# file/study_name, so SWEEP_TIMESTAMP is generated ONCE here, then passed to
# every job via `qsub -v` -- it is NOT each job's own RUN_TIMESTAMP.
set -euo pipefail

N_WORKERS="${N_WORKERS:-4}"
N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER:-25}"
SWEEP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Submitting ${N_WORKERS} workers x ${N_TRIALS_PER_WORKER} trials (sweep ${SWEEP_TIMESTAMP})"

for WORKER_ID in $(seq 1 "${N_WORKERS}"); do
    qsub -v SWEEP_TIMESTAMP="${SWEEP_TIMESTAMP}",WORKER_ID="${WORKER_ID}",N_TRIALS="${N_TRIALS_PER_WORKER}" \
        IY039_simclr_hparam_sweep_worker.sh
done

echo "Submitted."
echo "  Shared study:   IY039_sweep_${SWEEP_TIMESTAMP}"
echo "  Shared storage: IY039_optuna_study_${SWEEP_TIMESTAMP}.journal"
echo "Once all ${N_WORKERS} jobs finish, run:"
echo "  python IY039_sweep_analysis.py --sweep-timestamp ${SWEEP_TIMESTAMP}"
