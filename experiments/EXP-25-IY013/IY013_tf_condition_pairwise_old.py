#!/usr/bin/env python
"""
IY013 -- Pairwise (one-vs-one) classification, 6-class old dataset.

Old dataset: Nrg1/Rtg1 TFs x 3 glucose conditions (IY021 sanity-check panel).
Same data-loading pipeline as IY013_tf_condition_lstm_transformer.ipynb Section 1,
but instead of one 6-class task, every pair of the 6 classes (15 pairs) is scored
with Raw SVM / LSTM / Transformer on both Steady-state and Full variants.

Resumable: safe to re-run (e.g. after a kill / interruption) -- already-computed
(pair, variant, classifier) rows in the results CSV are skipped.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

IY013_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY013")
IY008_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY008")

sys.path.insert(0, str(Path("/home/ianyang/stochastic_simulations/src").resolve()))
sys.path.insert(0, str(IY013_DIR))
from utils.experimental_time_series import load_labelled_time_series_csvs
from IY013_classification_runners import run_pairwise_sweep

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# --- Data loading -- identical to IY013_tf_condition_lstm_transformer.ipynb Section 1 ---
OLD_SS_DATA_DIR   = IY008_DIR / "4_transformed_exp_time_series"
OLD_FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series"
OLD_META_PATH     = IY008_DIR / "old_data_metadata.csv"
META_COLS         = ["id", "group", "experiment"]

OLD_EXCLUDED_EXPS = {"18446"}
OLD_FILE_RE       = re.compile(r"^(\d+)_.*_group_(.+?)_(GFP|mCherry)_time_series$")

# Fixed 6-class selection: Nrg1 and Rtg1 at the 3 glucose conditions used in IY021
OLD_FIXED_CLASSES = [
    "Nrg1 @ 0.01% glucose",
    "Nrg1 @ 0.1% glucose",
    "Nrg1 @ 2% glucose (mock/steady)",
    "Rtg1 @ 0.01% glucose",
    "Rtg1 @ 0.1% glucose",
    "Rtg1 @ 2% glucose (mock/steady)",
]

old_metadata = pd.read_csv(OLD_META_PATH)
old_metadata["exp_id"]   = old_metadata["exp_id"].astype(str)
old_metadata["group_id"] = old_metadata["group_id"].astype(str)

OLD_LABEL_LOOKUP = {
    (row.exp_id, row.group_id, row.channel): (row.tf, row.condition)
    for _, row in old_metadata.iterrows()
}
print(f"Old metadata entries: {len(OLD_LABEL_LOOKUP)}")

print("=" * 70)
print(f"STEADY-STATE  ({OLD_SS_DATA_DIR.name})")
print("=" * 70)
old_ss_ts_raw, old_ss_label_strs = load_labelled_time_series_csvs(
    data_dir=OLD_SS_DATA_DIR, file_re=OLD_FILE_RE, label_lookup=OLD_LABEL_LOOKUP,
    meta_cols=META_COLS, excluded_exps=OLD_EXCLUDED_EXPS, verbose=False,
)
print(f"Loaded {len(old_ss_ts_raw)} files, {len(old_ss_label_strs)} total cells")

print("=" * 70)
print(f"FULL  ({OLD_FULL_DATA_DIR.name})")
print("=" * 70)
old_full_ts_raw, old_full_label_strs = load_labelled_time_series_csvs(
    data_dir=OLD_FULL_DATA_DIR, file_re=OLD_FILE_RE, label_lookup=OLD_LABEL_LOOKUP,
    meta_cols=META_COLS, excluded_exps=OLD_EXCLUDED_EXPS, verbose=False,
)
print(f"Loaded {len(old_full_ts_raw)} files, {len(old_full_label_strs)} total cells")

# --- Pairwise sweep over all 15 pairs of the 6 fixed classes ---
RESULTS_CSV = IY013_DIR / "IY013_tf_condition_pairwise_old_results.csv"

variants = {
    "Steady-state": (old_ss_ts_raw, old_ss_label_strs),
    "Full": (old_full_ts_raw, old_full_label_strs),
}

results_df = run_pairwise_sweep(
    variants=variants, classes=OLD_FIXED_CLASSES,
    results_csv=RESULTS_CSV, random_state=RANDOM_STATE,
)
print(f"\nDone. {len(results_df)} rows written to {RESULTS_CSV}")
