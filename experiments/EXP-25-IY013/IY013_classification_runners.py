"""
Classifier-runner helpers shared by the IY013 pairwise sweep scripts
(IY013_tf_condition_pairwise_old.py / IY013_tf_condition_pairwise_new.py).

make_val_split         — carve a val split out of a prepare_dataset() train portion
run_raw_svm            — RBF-SVM baseline on flat un-normalised series
run_lstm_transformer   — train + evaluate LSTM and Transformer classifiers
run_pairwise_sweep     — resumable one-vs-one sweep over every class pair
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classifiers.lstm_classifier import lstm_classifier
from classifiers.transformer_classifier import transformer_classifier
from utils.processing.pipeline import prepare_dataset

RANDOM_STATE = 42


def make_val_split(d: dict, random_state: int = RANDOM_STATE):
    """Further split prepare_dataset train portion 80/20 (stratified) into train/val."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        d["X_train_raw"], d["y_train"], test_size=0.2,
        random_state=random_state, stratify=d["y_train"],
    )
    return X_tr, X_val, d["X_test_raw"], y_tr, y_val, d["y_test"]


def run_raw_svm(d: dict, dataset_tag: str, random_state: int = RANDOM_STATE) -> float:
    """RBF-SVM on flat truncated (un-normalised) time series."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)),
    ])
    pipe.fit(d["X_train_raw"], d["y_train"])
    y_pred = pipe.predict(d["X_test_raw"])
    acc = accuracy_score(d["y_test"], y_pred)
    print(f"--- Raw SVM -- {dataset_tag}: Accuracy = {acc:.4f} ---")
    return acc


def run_lstm_transformer(d: dict, dataset_tag: str) -> dict:
    """Train + evaluate LSTM and Transformer classifiers on one prepared dataset dict."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = make_val_split(d)

    print(f"--- LSTM -- {dataset_tag} ---")
    lstm_acc = lstm_classifier(X_tr, X_val, X_te, y_tr, y_val, y_te)

    print(f"--- Transformer -- {dataset_tag} ---")
    transformer_acc = transformer_classifier(X_tr, X_val, X_te, y_tr, y_val, y_te)
    return {"LSTM": lstm_acc, "Transformer": transformer_acc}


def run_pairwise_sweep(
    variants: dict[str, tuple[list, list[str]]],
    classes: list[str],
    results_csv: Path,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """One-vs-one sweep: Raw SVM / LSTM / Transformer on every class pair x every variant.

    Resumable: if ``results_csv`` already exists, rows for (Class_A, Class_B, Dataset,
    Classifier) combos already present are skipped, and each newly computed pair+variant
    is appended to disk immediately so an interrupted run only loses the in-flight pair.

    Parameters
    ----------
    variants:
        Mapping of variant name (e.g. "Steady-state", "Full") to (ts_raw, label_strs),
        the raw per-file arrays and flat string labels as returned by
        ``load_labelled_time_series_csvs``.
    classes:
        All class names to draw pairs from (``itertools.combinations`` of the sorted list).
    results_csv:
        Path to write/append results to.
    """
    pairs = list(itertools.combinations(sorted(classes), 2))
    n_pairs = len(pairs)

    if results_csv.exists():
        done_df = pd.read_csv(results_csv)
        done_keys = set(
            zip(done_df["Class_A"], done_df["Class_B"], done_df["Dataset"], done_df["Classifier"])
        )
        print(f"Resuming from existing {results_csv.name}: {len(done_df)} rows already done")
    else:
        done_df = pd.DataFrame()
        done_keys = set()

    total = n_pairs * len(variants)
    step = 0
    for pair_idx, (cls_a, cls_b) in enumerate(pairs, start=1):
        for variant_name, (ts_raw, label_strs) in variants.items():
            step += 1
            tag = f"[{step}/{total}] {cls_a} vs {cls_b} / {variant_name}"

            needed = {"Raw SVM", "LSTM", "Transformer"} - {
                clf for (a, b, ds, clf) in done_keys if a == cls_a and b == cls_b and ds == variant_name
            }
            if not needed:
                print(f"{tag}: already complete, skipping")
                continue

            print(f"\n=== {tag} ===")
            d = prepare_dataset(
                ts_raw, label_strs, [cls_a, cls_b],
                dataset_name=tag, random_state=random_state,
            )
            accs = {"Raw SVM": run_raw_svm(d, tag, random_state=random_state)}
            accs.update(run_lstm_transformer(d, tag))

            rows = [
                {
                    "Class_A": cls_a, "Class_B": cls_b, "Dataset": variant_name,
                    "Classifier": clf_name, "Accuracy": acc,
                    "N_train": len(d["y_train"]), "N_test": len(d["y_test"]),
                }
                for clf_name, acc in accs.items() if clf_name in needed
            ]
            new_rows_df = pd.DataFrame(rows)
            write_header = not results_csv.exists()
            new_rows_df.to_csv(results_csv, mode="a", header=write_header, index=False)
            done_keys.update((r["Class_A"], r["Class_B"], r["Dataset"], r["Classifier"]) for r in rows)

    return pd.read_csv(results_csv)
