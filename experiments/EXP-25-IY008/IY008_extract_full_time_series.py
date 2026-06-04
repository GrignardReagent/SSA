"""
Extract FULL (unfiltered) time-series CSVs for both the old and NEW experimental datasets.

Unlike the step-3/4 pipeline (IY008_extract_post_media_switch.py +
IY008_transform_exp_time_series.py), this script:
  - Does NOT discard pre-media-switch timepoints
  - Does NOT apply steady-state extraction
  - Pivots the raw long-format TSV directly to a per-cell matrix and imputes NaNs

Old dataset output : 5_FULL_transformed_exp_time_series/
NEW dataset output : 5_FULL_transformed_exp_time_series_NEW/

NOTE: experiment 18446 is excluded — not properly recorded (no usable TF metadata).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "/home/ianyang/wela/src")
from wela.calculate_stats import load_tsv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
WELA_DIR  = BASE_DIR / "2_wela_data_analysis"

OUT_OLD = BASE_DIR / "5_FULL_transformed_exp_time_series"
OUT_NEW = BASE_DIR / "5_FULL_transformed_exp_time_series_NEW"

# Experiments explicitly excluded from processing
EXCLUDED_EXPS = {
    "18446",  # not properly recorded — no usable TF metadata
}

# Old-dataset experiment names (long-form, matching directory names in 2_wela_data_analysis/)
OLD_EXP_NAMES = [
    "18360_2020_01_04_steadystate_glucose_2min_01",
    "18464_2020_01_20_steadystate_glucose_756S_2min_mock_00",
    "18589_2020_02_10_steadystate_glucose_898S_2w2_01",
    "19316_2020_10_26_steadystate_glucose_144m_2w2_00",
    "19330_2020_11_02_steadystate_glucose_1345m_2w2_00",
    "19391_2020_11_12_steadystate_glucose_1345m_2w0p01_00",
    "19392_2020_11_12_steadystate_glucose_898m_2w0p01_00",
    "19394_2020_11_12_steadystate_glucose_1345m_2w0p1_00",
    "19554_2020_12_06_steadystate_glucose_2w0p01_900m_00",
    "19566_2020_12_07_steadystate_glucose_1344m_2w0p1_01",
    "20213_2021_09_07_steady_0p01glc_1344_1346_1347_00",
]

# New-dataset numeric OMID list (matches IY008_transform_exp_time_series_NEW.py)
NEW_OMIDS = [
    4053, 4052, 4051, 4054,
    4102, 4103, 4105, 4104,
    4106, 4107, 4108, 4109, 4110,
    3903, 3902, 4251,
    2858, 2854, 2853, 2852,
    2841, 2842, 2843, 2844, 2849,
    2801,
]

# ---------------------------------------------------------------------------
# Core transformation functions (shared between old and new)
# ---------------------------------------------------------------------------

CHANNELS = {
    "CV_mCherry": "mCherry",
    "CV_GFP":     "GFP",
}


def ensure_cv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CV columns from mean/std if they are absent (old-format TSVs)."""
    df = df.copy()
    for ch in ["GFP", "mCherry"]:
        cv_col   = f"CV_{ch}"
        mean_col = f"mother_mean_{ch}"
        std_col  = f"mother_std_{ch}"
        if cv_col not in df.columns and mean_col in df.columns and std_col in df.columns:
            df[cv_col] = df[std_col] / df[mean_col]
    return df


def create_time_series_matrix(df: pd.DataFrame, experiment_name: str, channel: str) -> dict:
    """Pivot long-format data to per-group (cell × time) matrices for one channel."""
    result = {}
    for group in df["group"].unique():
        gdf   = df[df["group"] == group]
        pivot = gdf.pivot(index="id", columns="time", values=channel).reset_index()
        pivot["group"]      = group
        pivot["experiment"] = experiment_name
        meta      = ["id", "group", "experiment"]
        time_cols = sorted(c for c in pivot.columns if c not in meta)
        result[f"group_{group}"] = pivot[meta + time_cols]
    return result


def handle_missing_values(ts_dict: dict) -> dict:
    """Impute NaNs in each group matrix using IterativeImputer (LinearRegression)."""
    processed = {}
    for group_name, df in ts_dict.items():
        meta      = ["id", "group", "experiment"]
        time_cols = [c for c in df.columns if c not in meta]
        if not time_cols:
            processed[group_name] = df.copy()
            continue

        matrix      = df[time_cols].values
        missing_pct = np.isnan(matrix).sum() / matrix.size * 100

        if missing_pct == 0:
            out = matrix
        elif missing_pct > 90:
            # Too sparse for IterativeImputer — fall back to per-row interpolation
            out = matrix.copy()
            for i in range(out.shape[0]):
                s = pd.Series(out[i]).interpolate().ffill().bfill()
                out[i] = s.values
        else:
            try:
                imputer = IterativeImputer(
                    estimator=LinearRegression(), random_state=42, verbose=0
                )
                out = imputer.fit_transform(matrix)
                if out.shape != matrix.shape:
                    raise ValueError(f"Shape mismatch: {out.shape} vs {matrix.shape}")
            except Exception:
                out = matrix.copy()
                for i in range(out.shape[0]):
                    s = pd.Series(out[i]).interpolate().ffill().bfill()
                    out[i] = s.values

        new_df            = df.copy()
        new_df[time_cols] = out
        processed[group_name] = new_df
    return processed


def process_tsv(omid: str, tsv_path: Path, out_dir: Path) -> list[Path]:
    """Pivot, impute, and write per-channel CSVs for one experiment. No time filtering."""
    print(f"[{omid}] Loading {tsv_path.name}")
    raw = load_tsv(str(tsv_path))
    raw = ensure_cv_columns(raw)

    needed = ["group", "time", "id", "CV_mCherry", "CV_GFP"]
    available = [c for c in needed if c in raw.columns]
    df_full = raw[available]

    written = []
    for col, channel_label in CHANNELS.items():
        if col not in df_full.columns:
            print(f"  [{channel_label}] column '{col}' not found — skipping")
            continue

        df = df_full[["group", "time", "id", col]].copy()
        ts = create_time_series_matrix(df, omid, channel=col)
        ts = handle_missing_values(ts)

        for group_key, gdf in ts.items():
            group_val = gdf["group"].iloc[0]
            fname     = f"{omid}_group_{group_val}_{channel_label}_time_series.csv"
            out_path  = out_dir / fname
            gdf.to_csv(out_path, index=False)
            n_cells = gdf.shape[0]
            n_tp    = gdf.shape[1] - 3  # subtract id, group, experiment columns
            print(f"  [{channel_label}] {fname}  ({n_cells} cells × {n_tp} timepoints)")
            written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Old dataset processing
# ---------------------------------------------------------------------------

def process_old_dataset():
    """Process the old (long-name) experimental dataset to OUT_OLD."""
    OUT_OLD.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"OLD DATASET → {OUT_OLD}")
    print(f"{'='*70}")

    all_written = []
    missing     = []

    for exp_name in OLD_EXP_NAMES:
        exp_id = exp_name.split("_")[0]
        if exp_id in EXCLUDED_EXPS:
            print(f"\n[{exp_name}] EXCLUDED — skipping")
            continue

        # Full TSV lives in 2_wela_data_analysis/{exp_id}/{exp_name}.tsv
        tsv_path = WELA_DIR / exp_id / f"{exp_name}.tsv"
        if not tsv_path.exists():
            print(f"\n[{exp_name}] TSV not found at {tsv_path} — skipping")
            missing.append(exp_name)
            continue

        written = process_tsv(exp_name, tsv_path, OUT_OLD)
        all_written.extend(written)

    if missing:
        print(f"\nWARNING: TSV not found for: {missing}")
    print(f"\nOld dataset done: {len(all_written)} CSV(s) written to {OUT_OLD}")
    return all_written


# ---------------------------------------------------------------------------
# NEW dataset processing
# ---------------------------------------------------------------------------

def process_new_dataset():
    """Process the NEW (numeric-OMID) experimental dataset to OUT_NEW."""
    OUT_NEW.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"NEW DATASET → {OUT_NEW}")
    print(f"{'='*70}")

    all_written = []
    missing     = []

    for omid in NEW_OMIDS:
        omid_str  = str(omid)
        omid_dir  = WELA_DIR / omid_str
        # Find the primary TSV (exclude any post_media_switch files)
        candidates = [
            p for p in omid_dir.glob(f"{omid_str}*.tsv")
            if "_post_media_switch" not in p.name
        ] if omid_dir.exists() else []

        if not candidates:
            print(f"\n[{omid_str}] TSV not found in {omid_dir} — skipping")
            missing.append(omid_str)
            continue

        tsv_path = candidates[0]
        written  = process_tsv(omid_str, tsv_path, OUT_NEW)
        all_written.extend(written)

    if missing:
        print(f"\nWARNING: TSV not found for omids: {missing}")
    print(f"\nNEW dataset done: {len(all_written)} CSV(s) written to {OUT_NEW}")
    return all_written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    old_written = process_old_dataset()
    new_written = process_new_dataset()
    total = len(old_written) + len(new_written)
    print(f"\n{'='*70}")
    print(f"All done. {total} CSV file(s) written total.")
    print(f"  Old: {len(old_written)} → {OUT_OLD}")
    print(f"  New: {len(new_written)} → {OUT_NEW}")
