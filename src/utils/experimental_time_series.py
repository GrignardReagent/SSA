"""Utilities for loading labelled experimental time-series CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Pattern

import pandas as pd


def load_labelled_time_series_csvs(
    data_dir: Path,
    file_re: Pattern[str],
    label_lookup: dict[tuple[str, str, str], tuple[str, str]],
    meta_cols: Iterable[str],
    label_fn: Callable[[str, str], str] | None = None,
    excluded_exps: set[str] | None = None,
    verbose: bool = True,
    missing_metadata_message: bool = True,
) -> tuple[list, list[str]]:
    """Load experiment CSV matrices and assign labels from metadata lookup.

    Parameters
    ----------
    data_dir:
        Directory containing per-channel time-series CSV files.
    file_re:
        Regex matching file stems and capturing exp_id, group_id and channel.
    label_lookup:
        Mapping from ``(exp_id, group_id, channel)`` to ``(tf, condition)``.
    meta_cols:
        Metadata columns to exclude from the returned time-series matrix.
    label_fn:
        Converts ``(tf, condition)`` into the class label. Defaults to
        ``"TF @ condition"``.
    excluded_exps:
        Experiment ids to skip.
    verbose:
        Print one load summary line per accepted file.
    missing_metadata_message:
        Print skipped-file messages when metadata is absent.

    Returns
    -------
    tuple[list, list[str]]
        Raw matrices and one label string per matrix row.
    """
    label_fn = label_fn or (lambda tf, condition: f"{tf} @ {condition}")
    excluded_exps = excluded_exps or set()
    meta_cols = set(meta_cols)
    all_ts_raw, label_strs = [], []

    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        match = file_re.match(csv_path.stem)
        if match is None:
            continue

        exp_id, group_id, channel = match.group(1), match.group(2), match.group(3)
        if exp_id in excluded_exps:
            continue

        key = (exp_id, group_id, channel)
        if key not in label_lookup:
            if verbose and missing_metadata_message:
                print(f"  Skipping (no metadata): {csv_path.name[:70]}")
            continue

        tf, condition = label_lookup[key]
        label = label_fn(tf, condition)
        df = pd.read_csv(csv_path)
        time_cols = [col for col in df.columns if col not in meta_cols]
        ts = df[time_cols].values.astype(float)
        all_ts_raw.append(ts)
        label_strs.extend([label] * len(ts))

        if verbose:
            print(
                f"  {exp_id}/{group_id}/{channel:8s}  -> "
                f"{label:45s}  ({len(ts)} cells, {ts.shape[1]} tp)"
            )

    return all_ts_raw, label_strs
