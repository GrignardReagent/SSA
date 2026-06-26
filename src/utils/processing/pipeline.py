"""
End-to-end data preparation pipeline for experimental time-series datasets.

prepare_dataset — impute → filter → truncate → balance → normalise → split
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from utils.processing.imputation import fill_nans
from utils.processing.normalisation import batch_wise_normalise
from utils.processing.balancing import balance_classes


def prepare_dataset(
    ts_raw: list,
    label_strs: list[str],
    kept_classes: list[str],
    dataset_name: str,
    random_state: int = 42,
) -> dict:
    """Filter, impute, balance, normalise, and split time-series data.

    Standard experiment setup in one call:

    1. Impute NaNs per file with :func:`~utils.processing.imputation.fill_nans`.
    2. Keep only rows whose string label is in ``kept_classes``.
    3. Truncate all series to the shortest common length.
    4. Balance classes to the minority count.
    5. Apply :func:`~utils.processing.normalisation.batch_wise_normalise`.
    6. Stratified 80 / 20 train / test split.

    Parameters
    ----------
    ts_raw:
        Iterable of 2-D arrays of shape (n_cells, n_timepoints), one per CSV file.
    label_strs:
        Flat list of string labels aligned row-wise with all arrays in ``ts_raw``.
    kept_classes:
        Class names to retain; all other rows are discarded.
    dataset_name:
        Display name used in progress messages.
    random_state:
        Seed for class balancing and the train / test split.

    Returns
    -------
    dict
        Keys:

        - ``X_bal``, ``y_bal`` — full balanced data before splitting
        - ``X_train``, ``X_test`` — batch-normalised splits
        - ``X_train_raw``, ``X_test_raw`` — un-normalised splits
        - ``y_train``, ``y_test``
        - ``class_names`` — sorted list of retained class names
        - ``min_T`` — common time-series length after truncation
        - ``min_count`` — cells per class after balancing

    Examples
    --------
    >>> from utils.processing.pipeline import prepare_dataset
    >>> ds = prepare_dataset(ts_raw, label_strs, kept_classes=["WT", "mutA"], dataset_name="my_exp")
    >>> ds["X_train"].shape
    (n_train, min_T)
    """
    kept_set = set(kept_classes)
    class_names = sorted(kept_classes)
    label_to_int = {cls: i for i, cls in enumerate(class_names)}

    flat_series: list[np.ndarray] = []
    flat_lbls: list[str] = []
    lbl_iter = iter(label_strs)

    for ts in ts_raw:
        # imputation
        ts_imp = fill_nans(np.asarray(ts, dtype=float))
        for row in ts_imp:
            lbl = next(lbl_iter)
            # filtering, keep only rows whose label is in kept_classes
            if lbl in kept_set:
                flat_series.append(row)
                flat_lbls.append(lbl)
    
    # truncate to shortest common length
    min_T = min(len(s) for s in flat_series)
    X_all = np.vstack([s[:min_T] for s in flat_series])
    y_all = np.array([label_to_int[l] for l in flat_lbls])
    n_cls = len(class_names)

    print(
        f"{dataset_name}: {X_all.shape[0]} cells × {min_T} tp, "
        f"{n_cls} classes, NaN remaining: {np.isnan(X_all).sum()}"
    )
    
    # balance classes to the minority count
    X_bal, y_bal = balance_classes(X_all, y_all, random_state=random_state)
    min_count = int(np.bincount(y_bal).min())
    print(f"  Balancing to {min_count} cells/class")

    X_norm = batch_wise_normalise(X_bal)

    idx_all = np.arange(len(y_bal))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=0.2, random_state=random_state, stratify=y_bal
    )
    print(f"  Train: {len(idx_tr)}  |  Test: {len(idx_te)}")

    return dict(
        X_bal=X_bal,            y_bal=y_bal,
        X_train_raw=X_bal[idx_tr], X_test_raw=X_bal[idx_te],
        X_train=X_norm[idx_tr], X_test=X_norm[idx_te],
        y_train=y_bal[idx_tr],  y_test=y_bal[idx_te],
        class_names=class_names,
        min_T=min_T,
        min_count=min_count,
    )
