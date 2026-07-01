"""
Time-series normalisation utilities.

For per-timepoint z-scoring across the sample axis, use
sklearn.preprocessing.StandardScaler directly (fit on train only when a
train/test split is involved, to avoid test-set leakage).
"""

from __future__ import annotations

import numpy as np


def instance_norm_np(x: np.ndarray) -> np.ndarray:
    """Z-score one trajectory using its own mean and standard deviation."""
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)
