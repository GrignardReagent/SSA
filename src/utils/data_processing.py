"""
Data processing utilities for labelling and processing time series data.
"""
import re
from collections import defaultdict
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pathlib import Path


def add_binary_labels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of ``df`` with a new ``label`` column.

    Parameters
    ----------
    df:
        DataFrame containing the simulation results.
    column:
        Column on which to base the 50/50 split.  Values greater than the
        halfway point of the sorted column are labelled ``1``; the rest are
        labelled ``0``.
    Returns
    -------
    pd.DataFrame
        DataFrame with the new ``label`` column added.
    """
    labelled = df.copy()
    
    # Check if all values are the same
    if labelled[column].nunique() == 1:
        print(f"Warning: All values in column '{column}' are identical.")
        # Return random 50/50 split
        labels = np.random.choice([0, 1], size=len(labelled))
        labelled["label"] = labels
        return labelled
    
    # Sort the values in the specified column to find the median split point
    order = np.argsort(labelled[column].values)
    
    # Calculate the midpoint index for a 50/50 binary classification split
    midpoint = len(labelled) // 2
    
    # Initialize all labels as 0 (lower half of sorted values)
    labels = np.zeros(len(labelled), dtype=int)
    
    # Assign label 1 to the upper half (values above median)
    labels[order[midpoint:]] = 1
    labelled["label"] = labels
    return labelled


def add_nearest_neighbour_labels(
    df: pd.DataFrame,
    mu_col: str = "mu_target",
    cv_col: str = "cv_target",
    tac_col: str = "t_ac_target",
    *,
    positive_on: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` labelled by nearest neighbour pairing.

    This function groups rows into disjoint pairs based on proximity in the
    three‑dimensional parameter space spanned by ``mu_col``, ``cv_col`` and
    ``tac_col``.  Pairing is performed greedily: the lowest‑index unused row is
    matched with the unused row to which it has the smallest Euclidean distance.
    Within each pair, whichever row has the larger ``positive_on`` value is
    labelled ``1`` and its partner ``0``.  If the number of rows is odd, the
    leftover row is labelled by comparing its ``positive_on`` value to the
    median of that column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results with parameter columns.
    mu_col, cv_col, tac_col : str, optional
        Names of the columns representing the parameter space.  The defaults
        correspond to ``"mu_target"``, ``"cv_target"`` and
        ``"t_ac_target"``.
    positive_on : str, optional
        Column used to decide which element of a pair gets the positive label.
        If ``None`` (default), ``mu_col`` is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``label`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'mu_target': [1.0, 1.2, 0.8, 1.1],
    ...     'cv_target': [0.10, 0.15, 0.12, 0.14],
    ...     't_ac_target': [5, 5, 5, 5],
    ... })
    >>> add_nearest_neighbour_labels(df)
       mu_target  cv_target  t_ac_target  label
    0        1.0       0.10               5      0
    1        1.2       0.15               5      1
    2        0.8       0.12               5      0
    3        1.1       0.14               5      1

    In this example the rows are paired by their nearest neighbours in
    ``(mu_target, cv_target, t_ac_target)`` space.  Within each pair the
    higher ``mu_target`` value receives the positive label ``1``.
    """

    if positive_on is None:
        # Default to using the first coordinate as the label comparator
        positive_on = mu_col

    labelled = df.copy()  # operate on a copy to avoid mutating caller's DataFrame
    coords = labelled[[mu_col, cv_col, tac_col]].to_numpy(dtype=float)
    n = len(coords)

    if n == 0:
        # Nothing to label; return empty column for consistency
        labelled["label"] = np.array([], dtype=int)
        return labelled

    # Precompute pairwise Euclidean distances in parameter space
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)  # ignore self-distances when searching for neighbours

    unused = set(range(n))  # indices not yet paired
    labels = np.zeros(n, dtype=int)
    pos_vals = labelled[positive_on].to_numpy()

    # Greedily build pairs: choose the smallest unused index and match to its nearest neighbour
    while len(unused) >= 2:
        i = min(unused)
        remaining = list(unused - {i})
        j = remaining[np.argmin(dist[i, remaining])]

        # Assign labels within the pair based on the positive_on column
        if pos_vals[i] > pos_vals[j]:
            labels[i], labels[j] = 1, 0
        elif pos_vals[i] < pos_vals[j]:
            labels[i], labels[j] = 0, 1
        else:  # deterministic tie‑break keeps function stable across runs
            labels[i], labels[j] = (0, 1) if i < j else (1, 0)

        unused.remove(i)
        unused.remove(j)

    # If there's an unpaired item, label it relative to the column median
    if unused:
        k = unused.pop()
        median_val = float(np.median(pos_vals))
        labels[k] = 1 if pos_vals[k] >= median_val else 0

    labelled["label"] = labels
    return labelled