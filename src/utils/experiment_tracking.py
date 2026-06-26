"""Experiment-run bookkeeping helpers."""

from __future__ import annotations

import csv
import os

import pandas as pd


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
