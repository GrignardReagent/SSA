"""
NaN imputation utilities for both numpy arrays and pandas DataFrames.

fill_nans            — numpy-level; used in ML pipelines
handle_missing_values — DataFrame-level; used in simulation/stats pipelines
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def fill_nans(
    X: np.ndarray,
    imputation_method: str = "IterativeImputer",
    random_state: int = 42,
) -> np.ndarray:
    """Impute NaN values in a 2-D numpy array using IterativeImputer.

    All-NaN columns are pre-filled with the global mean before fitting so that
    IterativeImputer can handle them. Residual NaNs after imputation are filled
    with the per-column median (global median as last resort).

    Parameters
    ----------
    X:
        Array of shape (n_samples, n_timepoints). May contain NaN.
    imputation_method:
        Currently only ``"IterativeImputer"`` is supported.
    random_state:
        Seed for IterativeImputer reproducibility.

    Returns
    -------
    np.ndarray
        Imputed array of the same shape, guaranteed NaN-free.

    Examples
    --------
    >>> import numpy as np
    >>> from utils.processing.imputation import fill_nans
    >>> X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    >>> np.isnan(fill_nans(X)).any()
    False
    """
    X_out = np.asarray(X, dtype=float).copy()
    original_shape = X_out.shape
    X_out = np.atleast_2d(X_out)

    if not np.isnan(X_out).any():
        return X_out.reshape(original_shape)

    # Pre-fill all-NaN columns so IterativeImputer can work
    all_nan_cols = np.isnan(X_out).all(axis=0)
    if all_nan_cols.any():
        finite_vals = X_out[np.isfinite(X_out)]
        fill_val = float(np.nanmean(finite_vals)) if finite_vals.size else 0.0
        X_out[:, all_nan_cols] = fill_val

    if imputation_method == "IterativeImputer":
        imputer = IterativeImputer(
            max_iter=10,
            tol=1e-3,
            initial_strategy="mean",
            random_state=random_state,
        )
    else:
        # TODO: support other imputation methods (e.g., KNN, MICE, etc.)
        raise ValueError(f"Unknown imputation method: {imputation_method}")

    X_out = imputer.fit_transform(X_out)

    # Per-column median fallback for residual NaNs
    if np.isnan(X_out).any():
        col_med = np.nanmedian(X_out, axis=0)
        g_med = float(np.nanmedian(X_out)) if np.isfinite(X_out).any() else 0.0
        col_med = np.where(np.isfinite(col_med), col_med, g_med)
        nan_mask = np.isnan(X_out)
        X_out[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

    return X_out.reshape(original_shape)


def handle_missing_values(time_series_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a time-series DataFrame using IterativeImputer.

    Each row is one time series; each column is a time step. Returns a DataFrame
    with the same index and columns, NaN-free.

    Parameters
    ----------
    time_series_df:
        DataFrame (n_samples × n_timepoints). Must be a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame with identical shape, index, and columns.

    Raises
    ------
    TypeError
        If ``time_series_df`` is not a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from utils.processing.imputation import handle_missing_values
    >>> df = pd.DataFrame({"t0": [1.0, np.nan], "t1": [np.nan, 2.5]})
    >>> handle_missing_values(df).isna().any().any()
    False
    """
    if not isinstance(time_series_df, pd.DataFrame):
        raise TypeError(
            "time_series_df must be a pandas DataFrame with rows as samples "
            "and columns as time steps"
        )

    index, columns = time_series_df.index, time_series_df.columns
    data_matrix = time_series_df.to_numpy(dtype=float, copy=False)

    print(f"  Data matrix shape before imputation: {data_matrix.shape}")
    print(
        f"  Missing values: {np.isnan(data_matrix).sum()} "
        f"({np.isnan(data_matrix).mean() * 100:.1f}%)"
    )

    if np.isnan(data_matrix).any():
        print("  Applying IterativeImputer…")
        imputer = IterativeImputer(
            max_iter=10,
            tol=1e-3,
            initial_strategy="mean",
            imputation_order="ascending",
            random_state=42,
        )
        imputed = imputer.fit_transform(data_matrix)
        print(f"  Remaining NaN values: {np.isnan(imputed).sum()}")
    else:
        print("  No missing values detected, using original data")
        imputed = data_matrix.astype(float, copy=True)

    df_out = pd.DataFrame(imputed, index=index, columns=columns)

    # Row-wise interpolation as a last resort
    if df_out.isna().any().any():
        print("  Warning: residual NaNs — applying row-wise interpolation…")
        df_out = (
            df_out.interpolate(method="linear", axis=1, limit_direction="both")
            .ffill(axis=1)
            .bfill(axis=1)
        )
        if df_out.isna().any().any():
            df_out = df_out.fillna(float(np.nanmean(data_matrix)))

    return df_out
