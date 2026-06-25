"""
Label-assignment utilities for simulation parameter DataFrames.

add_binary_labels             — median-split binary label on a single parameter column
add_nearest_neighbour_labels  — greedy nearest-neighbour pairing in 3-D parameter space
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_binary_labels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of ``df`` with a new ``label`` column (binary median split).

    Values in ``column`` that fall in the upper half of the sorted order receive
    label ``1``; the lower half receive ``0``.

    Parameters
    ----------
    df:
        DataFrame containing simulation parameters and results.
    column:
        Column on which to perform the 50/50 split.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``label`` column added.

    Examples
    --------
    >>> from utils.processing.labelling import add_binary_labels
    >>> df = pd.DataFrame({"mu_target": [1.0, 1.2, 0.8, 1.1]})
    >>> add_binary_labels(df, "mu_target")["label"].tolist()
    [0, 1, 0, 1]
    """
    labelled = df.copy()

    if labelled[column].nunique() == 1:
        print(f"Warning: all values in '{column}' are identical — using random 50/50 split.")
        labelled["label"] = np.random.choice([0, 1], size=len(labelled))
        return labelled

    order = np.argsort(labelled[column].values)
    midpoint = len(labelled) // 2
    labels = np.zeros(len(labelled), dtype=int)
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
    """Return a copy of ``df`` labelled by nearest-neighbour pairing.

    Rows are paired greedily in the 3-D parameter space (mu, cv, t_ac). Within
    each pair, the row with the larger ``positive_on`` value gets label ``1``.
    An odd leftover row is labelled relative to the column median.

    Parameters
    ----------
    df:
        DataFrame with parameter columns.
    mu_col, cv_col, tac_col:
        Names of the three parameter columns used as coordinates.
    positive_on:
        Column used to assign the positive label within each pair.
        Defaults to ``mu_col``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``label`` column.

    Examples
    --------
    >>> from utils.processing.labelling import add_nearest_neighbour_labels
    >>> df = pd.DataFrame({
    ...     "mu_target": [1.0, 1.2, 0.8, 1.1],
    ...     "cv_target": [0.10, 0.15, 0.12, 0.14],
    ...     "t_ac_target": [5, 5, 5, 5],
    ... })
    >>> add_nearest_neighbour_labels(df)["label"].tolist()
    [0, 1, 0, 1]
    """
    if positive_on is None:
        positive_on = mu_col

    labelled = df.copy()
    coords = labelled[[mu_col, cv_col, tac_col]].to_numpy(dtype=float)
    n = len(coords)

    if n == 0:
        labelled["label"] = np.array([], dtype=int)
        return labelled

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    unused = set(range(n))
    labels = np.zeros(n, dtype=int)
    pos_vals = labelled[positive_on].to_numpy()

    while len(unused) >= 2:
        i = min(unused)
        remaining = list(unused - {i})
        j = remaining[np.argmin(dist[i, remaining])]

        if pos_vals[i] > pos_vals[j]:
            labels[i], labels[j] = 1, 0
        elif pos_vals[i] < pos_vals[j]:
            labels[i], labels[j] = 0, 1
        else:
            labels[i], labels[j] = (0, 1) if i < j else (1, 0)

        unused.remove(i)
        unused.remove(j)

    if unused:
        k = unused.pop()
        labels[k] = 1 if pos_vals[k] >= float(np.median(pos_vals)) else 0

    labelled["label"] = labels
    return labelled
