"""
End-to-end data preparation pipeline for experimental time-series datasets.

prepare_dataset       — impute → filter → truncate → balance → split → normalise
prepare_split_dataset — as above, but a 3-way train/val/test split, optional
                        balancing, and the non-kept cells returned alongside
resample_to_length    — truncate / linspace-upsample one trace to a fixed length
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.processing.imputation import fill_nans
from utils.processing.balancing import balance_classes


def resample_to_length(row: np.ndarray, target: int) -> np.ndarray:
    """Truncate (or uniformly linspace-upsample) one trace to ``target`` points.

    Used when pooling files of differing lengths into one matrix, instead of
    truncating everything to the shortest file and discarding the tail of the
    longer ones.
    """
    if len(row) == target:
        return row
    if len(row) > target:
        return row[:target]
    # nearest-neighbour upsample: index the short trace at `target` evenly
    # spaced positions, so no values are interpolated into existence
    idx = np.round(np.linspace(0, len(row) - 1, target)).astype(int)
    return row[idx]


def prepare_split_dataset(
    ts_raw: list,
    label_strs: list[str],
    kept_classes: list[str],
    dataset_name: str,
    random_state: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    balance: bool = False,
    seq_len: int | None = None,
    return_rest: bool = False,
) -> dict:
    """Prepare a 3-way train / val / test split of a time-series dataset.

    Differs from :func:`prepare_dataset` in three ways, each motivated by
    contrastive-pretraining experiments (EXP-26-IY036 onwards):

    1. **A genuine val split**, carved alongside test rather than out of train
       by the caller. Checkpoint selection needs a held-out split that is not
       the test set, and making it a first-class output keeps every caller's
       val split identical.
    2. **Balancing is optional and off by default.** Subsampling to the
       minority class discards most of the data (57% of the IY036 benchmark);
       preferring all cells + ``balanced_accuracy_score`` keeps the metric
       comparable while roughly doubling the val and test splits.
    3. **The non-kept cells can be returned** (``return_rest``), so a caller
       that pretrains on every class but evaluates on a few can derive both
       pools from ONE pass over the data. Because the evaluation splits and the
       pretraining pool then come from disjoint index sets, test cells cannot
       leak into pretraining by construction — no post-hoc trace matching.

    Parameters
    ----------
    ts_raw:
        Iterable of 2-D arrays of shape (n_cells, n_timepoints), one per file.
    label_strs:
        Flat list of string labels aligned row-wise with all arrays in ``ts_raw``.
    kept_classes:
        Class names forming the evaluation benchmark. Other rows are discarded,
        or returned separately when ``return_rest`` is set.
    dataset_name:
        Display name used in progress messages.
    random_state:
        Seed for balancing and both splits.
    val_fraction, test_fraction:
        Fractions **of the whole kept set** (not of the remainder), so
        ``0.2 / 0.2`` gives a 60/20/20 split. Must sum to less than 1.
    balance:
        Subsample every kept class to the minority count before splitting.
        Off by default; set True to reproduce :func:`prepare_dataset`.
    seq_len:
        Resample every trace to this length via :func:`resample_to_length`.
        When None, truncate to the shortest common length among kept classes
        (matching :func:`prepare_dataset`).
    return_rest:
        Also return the cells whose label is not in ``kept_classes``.

    Returns
    -------
    dict
        Keys:

        - ``X_train``, ``X_val``, ``X_test`` — splits normalised with the
          train-fitted ``StandardScaler``
        - ``X_train_raw``, ``X_val_raw``, ``X_test_raw`` — un-normalised splits
        - ``y_train``, ``y_val``, ``y_test`` — integer labels
        - ``scaler`` — the fitted ``StandardScaler``, so callers can apply the
          identical normalisation to other data without refitting it
        - ``class_names`` — sorted list of kept class names
        - ``min_T`` — trace length after truncation / resampling
        - ``X_rest_raw``, ``y_rest_str`` — present only when ``return_rest``;
          un-normalised, same trace length, string labels

    Examples
    --------
    >>> from utils.processing.pipeline import prepare_split_dataset
    >>> d = prepare_split_dataset(ts_raw, label_strs, ["WT", "mutA"], "my_exp",
    ...                           val_fraction=0.2, test_fraction=0.2)
    >>> d["X_train"].shape, d["X_val"].shape, d["X_test"].shape
    """
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError(
            f"val_fraction + test_fraction must lie in (0, 1), "
            f"got {val_fraction} + {test_fraction} = {val_fraction + test_fraction}"
        )

    kept_set = set(kept_classes)
    class_names = sorted(kept_classes)
    label_to_int = {cls: i for i, cls in enumerate(class_names)}

    kept_series: list[np.ndarray] = []
    kept_lbls: list[str] = []
    rest_series: list[np.ndarray] = []
    rest_lbls: list[str] = []

    # ONE imputation pass over every file; kept and rest cells are separated
    # here so both pools get byte-identical preprocessing
    lbl_iter = iter(label_strs)
    for ts in ts_raw:
        ts_imp = fill_nans(np.asarray(ts, dtype=float))
        for row in ts_imp:
            lbl = next(lbl_iter)
            if lbl in kept_set:
                kept_series.append(row)
                kept_lbls.append(lbl)
            elif return_rest:
                rest_series.append(row)
                rest_lbls.append(lbl)

    if not kept_series:
        raise ValueError(f"no cells matched kept_classes={kept_classes}")

    # target length: caller-specified, else the shortest kept trace
    min_T = seq_len if seq_len is not None else min(len(s) for s in kept_series)
    X_all = np.vstack([resample_to_length(s, min_T) for s in kept_series])
    y_all = np.array([label_to_int[l] for l in kept_lbls])

    print(
        f"{dataset_name}: {X_all.shape[0]} cells × {min_T} tp, "
        f"{len(class_names)} classes, NaN remaining: {np.isnan(X_all).sum()}"
    )

    if balance:
        X_all, y_all = balance_classes(X_all, y_all, random_state=random_state)
        print(f"  Balancing to {int(np.bincount(y_all).min())} cells/class")

    # Two stratified carves. test_fraction is a fraction of the WHOLE set, so
    # the second carve is rescaled by the fraction that survived the first --
    # otherwise a 0.2/0.2 request would yield a 64/16/20 split, not 60/20/20.
    idx_all = np.arange(len(y_all))
    idx_fit, idx_test = train_test_split(
        idx_all, test_size=test_fraction, random_state=random_state, stratify=y_all
    )
    idx_train, idx_val = train_test_split(
        idx_fit,
        test_size=val_fraction / (1.0 - test_fraction), # to get the requested fraction of the whole set
        random_state=random_state,
        stratify=y_all[idx_fit],
    )
    print(f"  Train: {len(idx_train)}  |  Val: {len(idx_val)}  |  Test: {len(idx_test)}")

    # fit normalisation stats on train only, avoiding val/test leakage
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_all[idx_train])
    X_val_norm = scaler.transform(X_all[idx_val])
    X_test_norm = scaler.transform(X_all[idx_test])

    out = dict(
        X_train_raw=X_all[idx_train], X_val_raw=X_all[idx_val], X_test_raw=X_all[idx_test],
        X_train=X_train_norm,         X_val=X_val_norm,         X_test=X_test_norm,
        y_train=y_all[idx_train],     y_val=y_all[idx_val],     y_test=y_all[idx_test],
        scaler=scaler,
        class_names=class_names,
        min_T=min_T,
    )

    if return_rest:
        out["X_rest_raw"] = (
            np.vstack([resample_to_length(s, min_T) for s in rest_series])
            if rest_series else np.empty((0, min_T))
        )
        out["y_rest_str"] = np.array(rest_lbls)
        print(f"  Rest (non-benchmark): {out['X_rest_raw'].shape[0]} cells, "
              f"{len(set(rest_lbls))} classes")

    return out


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
    5. Stratified 80 / 20 train / test split.
    6. Fit ``StandardScaler`` on the train split only, transform both splits
       (avoids test-set statistics leaking into the training normalisation).

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
        - ``X_train``, ``X_test`` — splits normalised with train-fitted ``StandardScaler``
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
    # ! this could lead to data loss if the classes are very imbalanced, but it's a common practice to avoid bias in training
    X_bal, y_bal = balance_classes(X_all, y_all, random_state=random_state)
    min_count = int(np.bincount(y_bal).min())
    print(f"  Balancing to {min_count} cells/class")

    idx_all = np.arange(len(y_bal))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=0.2, random_state=random_state, stratify=y_bal
    )
    print(f"  Train: {len(idx_tr)}  |  Test: {len(idx_te)}")

    # fit normalisation stats on train only, avoiding test-set leakage
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_bal[idx_tr])
    X_test_norm = scaler.transform(X_bal[idx_te])

    return dict(
        X_bal=X_bal,            y_bal=y_bal,
        X_train_raw=X_bal[idx_tr], X_test_raw=X_bal[idx_te],
        X_train=X_train_norm,  X_test=X_test_norm,
        y_train=y_bal[idx_tr],  y_test=y_bal[idx_te],
        class_names=class_names,
        min_T=min_T,
        min_count=min_count,
    )
