"""
Class-balancing utilities for experimental time-series datasets.

balance_classes    — single-channel: subsample (X, y) to minority class size
balance_by_label   — dual-channel: subsample (X_m, X_g, metadata) together
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample every class to the minority class size (single-channel).

    Parameters
    ----------
    X:
        Feature array of shape (n_samples, ...).
    y:
        Integer class labels (n_samples,).
    random_state:
        RNG seed for reproducible subsampling.

    Returns
    -------
    X_bal : np.ndarray
        Balanced feature array of shape (n_balanced, ...).
    y_bal : np.ndarray
        Balanced labels (n_balanced,).

    Examples
    --------
    >>> from utils.processing.balancing import balance_classes
    >>> X_bal, y_bal = balance_classes(X, y)
    >>> np.bincount(y_bal)   # equal counts per class
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    # randomly subsample each class to the minority class size
    min_count = min(int(np.sum(y == c)) for c in classes)
    keep_idx = np.concatenate([
        rng.choice(np.where(y == c)[0], size=min_count, replace=False)
        for c in sorted(classes)
    ])
    keep_idx = np.sort(keep_idx)
    return X[keep_idx], y[keep_idx]


def balance_by_label(
    X_m: np.ndarray,
    X_g: np.ndarray,
    metadata: pd.DataFrame,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Subsample every class to the minority class size for dual-channel data.

    Used for paired mCherry / GFP fluorescence matrices where ``metadata``
    carries the ``"label"`` column. All three arrays are subsampled together
    so their row correspondence is preserved.

    Parameters
    ----------
    X_m:
        mCherry channel array (n_samples, n_timepoints).
    X_g:
        GFP channel array (n_samples, n_timepoints).
    metadata:
        DataFrame with a ``"label"`` column, aligned row-wise with X_m / X_g.
    random_state:
        RNG seed.

    Returns
    -------
    X_m_bal, X_g_bal : np.ndarray
        Balanced channel matrices.
    metadata_bal : pd.DataFrame
        Balanced metadata with index reset.

    Examples
    --------
    >>> from utils.processing.balancing import balance_by_label
    >>> X_m_bal, X_g_bal, meta_bal = balance_by_label(X_m, X_g, metadata)
    """
    rng = np.random.default_rng(random_state)
    labels = metadata["label"].to_numpy()
    classes = np.unique(labels)
    # randomly subsample each class to the minority class size
    min_count = min(int(np.sum(labels == c)) for c in classes)
    keep_idx = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=min_count, replace=False)
        for c in sorted(classes)
    ])
    keep_idx = np.sort(keep_idx)
    return (
        X_m[keep_idx],
        X_g[keep_idx],
        metadata.iloc[keep_idx].reset_index(drop=True),
    )
