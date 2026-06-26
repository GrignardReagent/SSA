"""Feature-frame preprocessing helpers."""

from __future__ import annotations

import pandas as pd


def fit_fill_feature_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill non-finite feature values using medians estimated on the train split."""
    train_df = train_df.replace([float("inf"), float("-inf")], pd.NA).copy()
    test_df = test_df.replace([float("inf"), float("-inf")], pd.NA).copy()
    medians = train_df.median()
    return train_df.fillna(medians).fillna(0.0), test_df.fillna(medians).fillna(0.0)
