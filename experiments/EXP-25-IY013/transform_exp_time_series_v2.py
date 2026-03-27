"""
Transform experimental time series data into steady-state CSV files for both
fluorescence channels (mCherry and GFP).

Discovers all *_post_media_switch.tsv files in ./exp_data, runs the same
steady-state extraction pipeline independently for each channel, and writes
one CSV per channel per group.

Output filename format:
    {omid}_group_{group}_mCherry_time_series.csv
    {omid}_group_{group}_GFP_time_series.csv

Usage:
    python transform_exp_time_series_v2.py
    python transform_exp_time_series_v2.py --out-dir /path/to/output
"""

import argparse
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
# Core transformation functions
# ---------------------------------------------------------------------------

def create_time_series_matrix(df: pd.DataFrame, experiment_name: str, channel: str) -> dict:
    """Pivot long-format data to per-group time-series matrices for one channel."""
    result = {}
    for group in df["group"].unique():
        gdf = df[df["group"] == group]
        pivot = gdf.pivot(index="id", columns="time", values=channel).reset_index()
        pivot["group"] = group
        pivot["experiment"] = experiment_name
        meta = ["id", "group", "experiment"]
        time_cols = sorted(c for c in pivot.columns if c not in meta)
        result[f"group_{group}"] = pivot[meta + time_cols]
    return result


def handle_missing_values(ts_dict: dict) -> dict:
    """Impute missing values in each group's matrix."""
    processed = {}
    for group_name, df in ts_dict.items():
        meta = ["id", "group", "experiment"]
        time_cols = [c for c in df.columns if c not in meta]
        if not time_cols:
            processed[group_name] = df.copy()
            continue

        matrix = df[time_cols].values
        missing_pct = np.isnan(matrix).sum() / matrix.size * 100

        if missing_pct == 0:
            out = matrix
        elif missing_pct > 90:
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
                    raise ValueError(f"Imputer returned shape {out.shape}, expected {matrix.shape}")
            except Exception:
                out = matrix.copy()
                for i in range(out.shape[0]):
                    s = pd.Series(out[i]).interpolate().ffill().bfill()
                    out[i] = s.values

        new_df = df.copy()
        new_df[time_cols] = out
        processed[group_name] = new_df
    return processed


def _detect_initial_burst(
    values: np.ndarray,
    burst_window: int = 5,
    burst_threshold: float = 0.2,
    min_burst_duration: int = 3,
) -> int:
    """Return index where initial burst ends (first stable point)."""
    burst_end_idx = 0
    for i in range(burst_window, len(values) - min_burst_duration):
        if pd.isna(values[i]):
            continue
        window = values[max(0, i - burst_window) : i]
        valid = window[~pd.isna(window)]
        if len(valid) < burst_window * 0.8:
            continue
        cv = np.std(valid) / abs(np.mean(valid)) if np.mean(valid) != 0 else 0
        if cv > burst_threshold:
            continue
        stable = True
        for j in range(i, min(i + min_burst_duration, len(values))):
            if pd.isna(values[j]):
                continue
            local = values[max(0, j - 2) : min(len(values), j + 3)]
            valid_local = local[~pd.isna(local)]
            if len(valid_local) >= 3:
                lcv = (
                    np.std(valid_local) / abs(np.mean(valid_local))
                    if np.mean(valid_local) != 0
                    else 0
                )
                if lcv > burst_threshold:
                    stable = False
                    break
        if stable:
            burst_end_idx = i
            break
    return burst_end_idx


def extract_steady_state(
    ts_dict: dict,
    window_size: int = 30,
    tolerance: float = 0.10,
    min_steady_duration: int = 5,
    burst_threshold: float = 0.10,
) -> dict:
    """Crop each group matrix to the median steady-state time window."""
    steady = {}
    for group, df in ts_dict.items():
        time_cols = sorted(c for c in df.columns if isinstance(c, (int, float)))
        starts, ends = [], []

        for _, row in df.iterrows():
            vals = row[time_cols].values
            if np.sum(~pd.isna(vals)) < window_size + min_steady_duration:
                continue

            burst_end = _detect_initial_burst(vals, burst_threshold=burst_threshold)
            analysis_start = max(burst_end, window_size)
            if analysis_start >= len(vals) - min_steady_duration:
                continue

            win_means, win_times = [], []
            for i in range(analysis_start, len(time_cols) - window_size + 1):
                w = vals[i : i + window_size]
                if np.sum(~pd.isna(w)) >= window_size * 0.8:
                    win_means.append(np.nanmean(w))
                else:
                    win_means.append(np.nan)
                win_times.append(time_cols[i + window_size - 1])

            ss_start = None
            for i in range(len(win_means) - min_steady_duration):
                if pd.isna(win_means[i]):
                    continue
                ref = win_means[i] if win_means[i] != 0 else 1e-10
                ok = True
                for j in range(i + 1, min(i + 1 + min_steady_duration, len(win_means))):
                    if pd.isna(win_means[j]) or abs(win_means[j] - ref) / abs(ref) > tolerance:
                        ok = False
                        break
                if ok:
                    ss_start = i
                    break

            ss_end = None
            if ss_start is not None:
                ref = win_means[ss_start] if win_means[ss_start] != 0 else 1e-10
                for i in range(ss_start + min_steady_duration, len(win_means) - min_steady_duration):
                    if pd.isna(win_means[i]):
                        continue
                    if abs(win_means[i] - ref) / abs(ref) > tolerance:
                        done = True
                        for j in range(i + 1, min(i + 1 + min_steady_duration, len(win_means))):
                            if pd.isna(win_means[j]):
                                continue
                            if abs(win_means[j] - ref) / abs(ref) <= tolerance:
                                done = False
                                break
                        if done:
                            ss_end = i
                            break

            if ss_start is not None:
                starts.append(win_times[ss_start])
                ends.append(win_times[ss_end] if ss_end is not None else np.nan)

        if not starts:
            continue

        t_start = np.median([t for t in starts if not pd.isna(t)])
        valid_ends = [t for t in ends if not pd.isna(t)]
        t_end = np.median(valid_ends) if valid_ends else None

        if t_end is not None:
            keep = [c for c in time_cols if t_start <= c <= t_end]
        else:
            keep = [c for c in time_cols if c >= t_start]

        if keep:
            out = df[["id", "group", "experiment"] + keep].copy()
            out = out.dropna(subset=keep, how="all")
            steady[group] = out

    return steady


# ---------------------------------------------------------------------------
# Per-channel pipeline
# ---------------------------------------------------------------------------

CHANNELS = {
    "CV_mCherry": "mCherry",
    "CV_GFP":     "GFP",
}


def process_omid(
    omid: str,
    tsv_path: Path,
    out_dir: Path,
    window_size: int = 30,
    tolerance: float = 0.10,
    min_steady_duration: int = 5,
    burst_threshold: float = 0.10,
) -> list[Path]:
    """Process one experiment for both channels; return list of written CSV paths."""
    print(f"[{omid}] Loading {tsv_path}")
    needed_cols = ["group", "time", "id", "CV_mCherry", "CV_GFP"]
    raw = load_tsv(str(tsv_path))
    available = [c for c in needed_cols if c in raw.columns]
    df_full = raw[available]

    written = []
    for col, channel_label in CHANNELS.items():
        if col not in df_full.columns:
            print(f"  [{channel_label}] column '{col}' not found — skipping")
            continue

        df = df_full[["group", "time", "id", col]].copy()

        ts = create_time_series_matrix(df, omid, channel=col)
        ts = handle_missing_values(ts)
        ts = extract_steady_state(
            ts,
            window_size=window_size,
            tolerance=tolerance,
            min_steady_duration=min_steady_duration,
            burst_threshold=burst_threshold,
        )

        for group_key, gdf in ts.items():
            group_val = gdf["group"].iloc[0]
            fname = f"{omid}_group_{group_val}_{channel_label}_time_series.csv"
            out_path = out_dir / fname
            gdf.to_csv(out_path, index=False)
            print(f"  [{channel_label}] Wrote {out_path}  "
                  f"({gdf.shape[0]} cells × {gdf.shape[1] - 3} timepoints)")
            written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "transformed_exp_time_series_data_v2"),
        help="Output directory for CSVs (default: ./transformed_exp_time_series_data_v2)",
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--min-steady-duration", type=int, default=5)
    parser.add_argument("--burst-threshold", type=float, default=0.10)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_dir = Path(__file__).parent / "exp_data"
    tsv_files = sorted(tsv_dir.glob("*_post_media_switch.tsv"))
    if not tsv_files:
        print(f"ERROR: no *_post_media_switch.tsv files found in {tsv_dir}", file=sys.stderr)
        sys.exit(1)

    tasks = [(p.name.replace("_post_media_switch.tsv", ""), p) for p in tsv_files]
    print(f"Found {len(tasks)} TSV file(s) in {tsv_dir}")

    all_written = []
    for omid, tsv_path in tasks:
        written = process_omid(
            omid,
            tsv_path,
            out_dir,
            window_size=args.window_size,
            tolerance=args.tolerance,
            min_steady_duration=args.min_steady_duration,
            burst_threshold=args.burst_threshold,
        )
        all_written.extend(written)

    print(f"\nDone. {len(all_written)} CSV file(s) written to {out_dir}")


if __name__ == "__main__":
    main()
