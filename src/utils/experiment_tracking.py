"""Experiment-run bookkeeping helpers."""

from __future__ import annotations

import csv
import os
from datetime import datetime

import pandas as pd

# Env var a job script sets so its .out log and the run's artifacts share a stamp
RUN_TIMESTAMP_ENV = "RUN_TIMESTAMP"


def run_timestamp() -> str:
    """Return the ``YYYYmmdd_HHMMSS`` stamp identifying this run.

    Job scripts export ``RUN_TIMESTAMP`` before launching Python, so the shell's
    ``.out`` log and every artifact written by the script (checkpoints, history
    CSVs, W&B run names) carry the *same* stamp and can be matched up later.
    When the script is run directly (no job script), fall back to the current
    time so standalone runs still get a unique stamp.
    
    In the .sh shell script:
    
    ```bash
    export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    python IY0XX.py > "IY0XX_${RUN_TIMESTAMP}.out" 2>&1
    ```
    
    In the Python script, to recover the stamp for naming artifacts:
    
    ```python
    from utils.experiment_tracking import run_timestamp
    
    timestamp = run_timestamp()
    artifact_save_path = f"IY0XX_analysis_{timestamp}.csv"
    ```    
    """
    return os.environ.get(RUN_TIMESTAMP_ENV) or datetime.now().strftime("%Y%m%d_%H%M%S")


def claim_config(config_key, results_file, csv_columns) -> bool:
    """Claim a sweep configuration by writing an ``IN_PROGRESS`` placeholder.

    The placeholder prevents concurrent jobs from duplicating the same
    architecture/hyperparameter row in simple CSV-backed sweeps.
    """
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        keys = set(
            (
                row["architecture"],
                row["hidden_size"],
                row["num_layers"],
                row["dropout_rate"],
                row["learning_rate"],
                row["batch_size"],
                row["epochs"],
            )
            for _, row in df.iterrows()
        )

        if config_key in keys:
            return False

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(
            {
                "architecture": config_key[0],
                "hidden_size": config_key[1],
                "num_layers": config_key[2],
                "dropout_rate": config_key[3],
                "learning_rate": config_key[4],
                "batch_size": config_key[5],
                "epochs": config_key[6],
                "train_acc": "IN_PROGRESS",
                "val_acc": None,
                "test_acc": None,
                "test_acc_std": None,
                "time": None,
            }
        )

    return True
