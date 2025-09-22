#!/usr/bin/env python3
"""Benchmark LSTM vs Transformer on a few t_ac ratios with compact grid searches.

The script mirrors the original benchmarking flow but adds a light, systematic
hyperparameter sweep for both models on four stress/normal autocorrelation time
ratios (>1).  For each ratio we load the steady-state trajectories, create a
train/validation/test split, evaluate all combinations from small grids that
cover the most influential hyperparameters, and record the best validation model
along with its test accuracy.  A CSV and a simple trend plot are saved for later
inspection.  The script does not execute unless called directly.
"""

from __future__ import annotations

import itertools
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure classifier wrappers are importable when running from the project root.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from classifiers.lstm_classifier import lstm_classifier
from classifiers.transformer_classifier import transformer_classifier

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------
BASE_TAC_NORMAL = 50.0
RATIOS = [1.20, 1.40, 1.60, 1.80]
DATA_ROOT = Path(__file__).resolve().parent / "data_t_ac"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_simple_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

# Grid definitions keep the search compact while covering high-impact knobs.
LSTM_GRID = {
    "hidden_size": [64, 96, 128],
    "num_layers": [1, 2, 4],
    "dropout_rate": [0.20, 0.30],
    "learning_rate": [0.01, 0.001],
    "use_attention": [False, True],
    "use_conv1d": [False, True],
}
LSTM_FIXED = {
    "batch_size": 64,
    "epochs": 45,
    "bidirectional": True,
}

TRANSFORMER_GRID = {
    "d_model": [32, 64, 128],
    "num_layers": [2, 3],
    "dropout_rate": [0.10, 0.20, 0.30],
    "learning_rate": [0.01, 0.001, 8.0e-4, 9.0e-4],
    "use_conv1d": [False, True],
    "pooling_strategy": ["mean", "last"],
}
TRANSFORMER_FIXED = {
    "nhead": 4,
    "batch_size": 64,
    "epochs": 50,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ratio_to_directory(ratio: float) -> Path:
    stress_tac = (Decimal(str(ratio)) * Decimal(str(BASE_TAC_NORMAL))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    folder = f"mRNA_trajectories_tac_{float(stress_tac):.2f}_{BASE_TAC_NORMAL:.2f}"
    return DATA_ROOT / folder / "steady_state_trajectories"


def load_ratio_dataframe(ratio: float) -> pd.DataFrame:
    data_dir = ratio_to_directory(ratio)
    csv_files = sorted(data_dir.glob("*_SS.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No steady-state CSV files found for ratio {ratio}")

    frames = [pd.read_csv(path) for path in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def split_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_lstm_candidates() -> List[Dict]:
    combos: List[Dict] = []
    grid_keys = list(LSTM_GRID.keys())
    for values in itertools.product(*(LSTM_GRID[key] for key in grid_keys)):
        params = dict(zip(grid_keys, values))
        candidate = {**LSTM_FIXED, **params}
        if candidate["use_attention"]:
            candidate["use_conv1d"] = True
            candidate["num_attention_heads"] = 2 if candidate["hidden_size"] == 64 else 4
        combos.append(candidate)
    return combos


def build_transformer_candidates() -> List[Dict]:
    combos: List[Dict] = []
    grid_keys = list(TRANSFORMER_GRID.keys())
    for values in itertools.product(*(TRANSFORMER_GRID[key] for key in grid_keys)):
        params = dict(zip(grid_keys, values))
        candidate = {**TRANSFORMER_FIXED, **params}
        if candidate["use_conv1d"] and candidate["d_model"] >= 192:
            candidate["nhead"] = 6
        combos.append(candidate)
    return combos


def evaluate_candidates(
    candidates: Iterable[Dict],
    train_val_test: Tuple[np.ndarray, ...],
    model_type: str,
) -> Tuple[float, Dict]:
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test

    best_acc = -np.inf
    best_params: Dict | None = None

    for params in candidates:
        if model_type == "lstm":
            acc = lstm_classifier(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                input_size=1,
                output_size=len(np.unique(y_train)),
                **params,
            )
        elif model_type == "transformer":
            acc = transformer_classifier(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                input_size=1,
                output_size=len(np.unique(y_train)),
                **params,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if acc > best_acc:
            best_acc = acc
            best_params = params

    assert best_params is not None
    return best_acc, best_params


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_SEED)

    lstm_candidates = build_lstm_candidates()
    transformer_candidates = build_transformer_candidates()

    summary_rows: List[Dict] = []

    for ratio in RATIOS:
        print(f"\n=== Processing t_ac ratio {ratio:.2f} ===")
        df_ratio = load_ratio_dataframe(ratio)
        datasets = split_dataset(df_ratio)

        lstm_acc, lstm_best = evaluate_candidates(lstm_candidates, datasets, "lstm")
        trans_acc, trans_best = evaluate_candidates(transformer_candidates, datasets, "transformer")

        print(f"Best LSTM test accuracy: {lstm_acc:.3f} with {lstm_best}")
        print(f"Best Transformer test accuracy: {trans_acc:.3f} with {trans_best}")

        summary_rows.append(
            {
                "t_ac_ratio": ratio,
                "model": "LSTM",
                "best_test_accuracy": lstm_acc,
                "best_params": lstm_best,
            }
        )
        summary_rows.append(
            {
                "t_ac_ratio": ratio,
                "model": "Transformer",
                "best_test_accuracy": trans_acc,
                "best_params": trans_best,
            }
        )

    results_df = pd.DataFrame(summary_rows)
    csv_path = OUTPUT_DIR / "lstm_vs_transformer_simple_results_2.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")

    pivot_df = results_df.pivot_table(index="t_ac_ratio", columns="model", values="best_test_accuracy").sort_index()
    ax = pivot_df.plot(marker="o", figsize=(6, 4))
    ax.set_ylabel("Best test accuracy")
    ax.set_xlabel("t_ac ratio")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.4)
    fig = ax.get_figure()
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "lstm_vs_transformer_accuracy_trend_2.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved accuracy trend plot to {plot_path}")


if __name__ == "__main__":
    main()
