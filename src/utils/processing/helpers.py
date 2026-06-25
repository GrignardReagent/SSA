"""
Internal type-conversion helpers shared by stats and visualisation modules.

These are intentionally private (underscore-prefixed) to signal they are
implementation details, not part of the public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_numpy(data) -> np.ndarray:
    """Convert a pandas DataFrame to a numpy array; pass arrays through unchanged."""
    if isinstance(data, pd.DataFrame):
        return data.values
    return data


def _safe_slice(data, start_idx=None, end_idx=None, axis: int = 1) -> np.ndarray:
    """Slice a DataFrame or numpy array along rows (axis=0) or columns (axis=1)."""
    if isinstance(data, pd.DataFrame):
        if axis == 1:
            return data.iloc[:, start_idx:end_idx].values
        return data.iloc[start_idx:end_idx, :].values
    if axis == 1:
        return data[:, start_idx:end_idx]
    return data[start_idx:end_idx, :]


def _return_no_label_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 'label' column from a DataFrame, warning if it is present."""
    if "label" in df.columns:
        print("Warning: 'label' column exists in DataFrame.")
        df = df.drop(columns=["label"], errors="ignore")
    return df
