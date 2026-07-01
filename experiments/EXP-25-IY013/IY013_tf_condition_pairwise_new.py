#!/usr/bin/env python
"""
IY013 -- Pairwise (one-vs-one) classification, 12-class new dataset.

New dataset: TF identity only, conditions pooled (IY021 sanity-check panel).
Same data-loading pipeline as IY013_tf_condition_lstm_transformer.ipynb Section 2,
but instead of one 12-class task, every pair of the 12 TFs (66 pairs) is scored
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

# --- Data loading -- identical to IY013_tf_condition_lstm_transformer.ipynb Section 2 ---
NEW_SS_DATA_DIR   = IY008_DIR / "4_transformed_exp_time_series_NEW"
NEW_FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series_NEW"
NEW_META_PATH     = IY008_DIR / "NEW_data_metadata.csv"
META_COLS         = ["id", "group", "experiment"]

NEW_EXCLUDED_TFS = {"Empty", "Gcd6_Poss_Mix", "Sse2_Poss_Mix"}
NEW_FILE_RE      = re.compile(r"^(\d+)_group_(.+?)_(GFP|mCherry)_time_series$")

# Fixed 12-class selection: TF identity only (conditions pooled), IY021 sanity-check panel
NEW_FIXED_TFS = {"Opi1", "Msn2", "Yox1", "Tea1", "Rox1", "Sok2",
                  "Cup9", "Spt15", "Cbf1", "Stb5", "Cin5", "Rsc3"}

new_metadata = pd.read_csv(NEW_META_PATH)
new_metadata["exp_id"]   = new_metadata["exp_id"].astype(str)
new_metadata["group_id"] = new_metadata["group_id"].astype(str)

NEW_LABEL_LOOKUP = {
    (row.exp_id, row.group_id, row.channel): (row.tf, row.condition)
    for _, row in new_metadata.iterrows()
    if row.tf not in NEW_EXCLUDED_TFS
}
print(f"New metadata entries: {len(new_metadata)}  (after excluding noisy TFs: {len(NEW_LABEL_LOOKUP)})")

print("=" * 70)
print(f"STEADY-STATE  ({NEW_SS_DATA_DIR.name})")
print("=" * 70)
new_ss_ts_raw, new_ss_label_strs = load_labelled_time_series_csvs(
    data_dir=NEW_SS_DATA_DIR, file_re=NEW_FILE_RE, label_lookup=NEW_LABEL_LOOKUP,
    meta_cols=META_COLS, label_fn=lambda tf, condition: tf, verbose=False,
)
print(f"Loaded {len(new_ss_ts_raw)} files, {len(new_ss_label_strs)} total cells")

print("=" * 70)
print(f"FULL  ({NEW_FULL_DATA_DIR.name})")
print("=" * 70)
new_full_ts_raw, new_full_label_strs = load_labelled_time_series_csvs(
    data_dir=NEW_FULL_DATA_DIR, file_re=NEW_FILE_RE, label_lookup=NEW_LABEL_LOOKUP,
    meta_cols=META_COLS, label_fn=lambda tf, condition: tf, verbose=False,
)
print(f"Loaded {len(new_full_ts_raw)} files, {len(new_full_label_strs)} total cells")

# --- Pairwise sweep over all 66 pairs of the 12 fixed TFs ---
RESULTS_CSV = IY013_DIR / "IY013_tf_condition_pairwise_new_results.csv"

variants = {
    "Steady-state": (new_ss_ts_raw, new_ss_label_strs),
    "Full": (new_full_ts_raw, new_full_label_strs),
}

results_df = run_pairwise_sweep(
    variants=variants, classes=sorted(NEW_FIXED_TFS),
    results_csv=RESULTS_CSV, random_state=RANDOM_STATE,
)
print(f"\nDone. {len(results_df)} rows written to {RESULTS_CSV}")
