"""
Time-series normalisation utilities.

batch_wise_normalise  — z-score per timepoint across the sample axis (SimCLR convention)

Note: for train/test normalisation where you need to apply training statistics to
a held-out set, use sklearn.preprocessing.StandardScaler directly.
"""

from __future__ import annotations

import numpy as np


def batch_wise_normalise(X: np.ndarray) -> np.ndarray:
    """Z-score each timepoint across the sample axis.

    Matches the normalisation applied during SimCLR pre-training. Each column
    (timepoint) is centred and scaled using statistics from the entire batch.

    Parameters
    ----------
    X:
        Array of shape (n_samples, n_timepoints).

    Returns
    -------
    np.ndarray
        Normalised array with the same shape.

    Examples
    --------
    >>> from utils.processing.normalisation import batch_wise_normalise
    >>> Xn = batch_wise_normalise(np.random.randn(100, 50))
    >>> abs(Xn.mean(axis=0)).max() < 1e-6
    True
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std
