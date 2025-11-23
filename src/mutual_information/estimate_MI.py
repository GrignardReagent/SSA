import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import clone

def _mi_from_confusion(cm: np.ndarray) -> float:
    """
    Mutual information I(Y; Ŷ) computed from a row-normalised confusion matrix.
    Rows: true class y, columns: predicted class ŷ.
    """
    # p(ŷ | y) — assume cm is row-normalised
    p_yhat_given_y = cm
    # class prior p(y) — assume balanced unless sample counts provided;
    # with confusion built from the test split, we use empirical class prior:
    n_per_class = cm.sum(axis=1)
    if np.any(n_per_class == 0):
        # fall back to uniform prior if a class is missing in this bootstrap sample
        p_y = np.ones(cm.shape[0]) / cm.shape[0]
    else:
        p_y = n_per_class / n_per_class.sum()

    # p(ŷ) = sum_y p(y) p(ŷ|y)
    p_yhat = np.sum(p_y[:, None] * p_yhat_given_y, axis=0)

    # H(Ŷ)
    nz = p_yhat > 0
    h_yhat = -np.sum(p_yhat[nz] * np.log2(p_yhat[nz]))

    # H(Ŷ|Y)
    # handle log(0) safely by masking zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_p = np.zeros_like(p_yhat_given_y)
        mask = p_yhat_given_y > 0
        log2_p[mask] = np.log2(p_yhat_given_y[mask])
    h_yhat_given_y = -np.sum(p_y * np.sum(p_yhat_given_y * log2_p, axis=1))

    return h_yhat - h_yhat_given_y


def _mi_from_probabilities(y_true: np.ndarray, P_hat: np.ndarray) -> float:
    """
    Estimate I(Y; Ŷ) using predicted class-probability vectors instead of hard labels.
    P_hat is shape (n_samples, n_classes) with rows summing to 1.
    We estimate p(ŷ|y) by averaging predicted distributions within each true class.
    """
    classes = np.unique(y_true)
    n_classes = P_hat.shape[1]
    p_y = np.array([(y_true == c).mean() for c in classes])

    # p(ŷ|y=c) = average of predicted distributions for samples with true class c
    p_yhat_given_y = np.zeros((len(classes), n_classes))
    for i, c in enumerate(classes):
        idx = (y_true == c)
        if idx.sum() == 0:
            # if class absent, use a tiny uniform smoothing to avoid zeros
            p_yhat_given_y[i] = np.ones(n_classes) / n_classes
        else:
            avg = P_hat[idx].mean(axis=0)
            # normalise for numerical safety
            s = avg.sum()
            p_yhat_given_y[i] = avg / s if s > 0 else np.ones(n_classes) / n_classes

    # p(ŷ)
    p_yhat = np.sum(p_y[:, None] * p_yhat_given_y, axis=0)

    # H(Ŷ)
    nz = p_yhat > 0
    h_yhat = -np.sum(p_yhat[nz] * np.log2(p_yhat[nz]))

    # H(Ŷ|Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_p = np.zeros_like(p_yhat_given_y)
        mask = p_yhat_given_y > 0
        log2_p[mask] = np.log2(p_yhat_given_y[mask])
    h_yhat_given_y = -np.sum(p_y * np.sum(p_yhat_given_y * log2_p, axis=1))

    return h_yhat - h_yhat_given_y


def estimate_mi(
    data,
    classifier_factory,
    *,
    overtime: bool = True,
    n_bootstraps: int = 100,
    ci=(0.25, 0.75),
    test_size: float = 0.25,
    stratify: bool = True,
    random_state: int | None = None,
    param_grid: dict | None = None,
    cv: int = 5,
    n_jobs: int = -1,
    use_probabilities: bool = False,
    verbose: bool = False,
):
    """
    Estimate the mutual information (MI) between time-series and labels using
    a user-supplied classifier. This function itself performs *no* model
    selection beyond an optional GridSearchCV that you configure.

    Parameters
    ----------
    data : list[np.ndarray]
        List of arrays, one per class. Each array is shape (n_cells, T),
        with rows = single-cell time series, columns = time points.
    classifier_factory : Callable[[], Estimator]
        A zero-argument callable that returns a *fresh, unfitted* estimator or
        Pipeline implementing `fit`, `predict`, and (optionally) `predict_proba`.
        This design supports a wide range of ML and DL models, including
        scikit-learn, sklearn-compatible wrappers, and custom pipelines.
    overtime : bool, default True
        If True, compute MI for increasing durations (t = 1..T). If False, use
        full length only.
    n_bootstraps : int, default 100
        Number of bootstrap resamples for uncertainty.
    ci : tuple(float, float), default (0.25, 0.75)
        Quantiles to report as confidence limits.
    test_size : float, default 0.25
        Proportion of data reserved for testing within each bootstrap.
    stratify : bool, default True
        Stratify train/test splits by class.
    random_state : int or None
        Random seed for reproducibility of the splitting.
    param_grid : dict or None
        If provided, perform GridSearchCV on the *entire* dataset for each
        duration to determine best hyper-parameters, then use that in bootstraps.
        Keys must match estimator parameter names.
    cv : int, default 5
        Cross-validation folds for GridSearchCV (ignored if param_grid is None).
    n_jobs : int, default -1
        Parallelism for GridSearchCV.
    use_probabilities : bool, default False
        If True and the estimator provides `predict_proba`, compute MI from the
        predicted probability distributions (usually less noisy than hard labels).
        Otherwise, MI is computed from the confusion matrix of hard predictions.
    verbose : bool, default False
        Print progress and selected hyper-parameters.

    Returns
    -------
    res : np.ndarray, shape (n_durations, 3)
        For each duration, returns [median_MI, lower_CI, upper_CI].

    Notes
    -----
    - MI is computed as I(Y; Ŷ) using either:
        (a) a row-normalised confusion matrix p(ŷ|y), or
        (b) averaged predicted class-probability vectors to estimate p(ŷ|y).
    - Class priors are taken empirically from the test split in each bootstrap;
      if a class is absent in a split, a uniform prior is used for stability.
      
    USAGE EXAMPLE
    --------------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> # 1) Build a factory that returns a fresh estimator/pipeline
    >>> def clf_factory():
    >>>     return Pipeline([
    >>>         ("scale", StandardScaler(with_mean=True)),
    >>>         ("clf", LogisticRegression(max_iter=500))
    >>>     ])

    >>> # 2) (Optional) define a param grid (keys must match the estimator)
    >>> param_grid = {
    >>>     "clf__C": np.logspace(-3, 3, 7),
    >>>     "clf__penalty": ["l2"],
    >>>     "clf__solver": ["lbfgs"],
    >>> }
    >>> # 3) Call estimate_mi
    >>> res = estimate_mi(
    >>>     data=[data_class0, data_class1, data_class2],
    >>>     classifier_factory=clf_factory,
    >>>     overtime=True,
    >>>     n_bootstraps=200,
    >>>     ci=(0.10, 0.90),
    >>>     param_grid=param_grid,   # or None to skip tuning
    >>>     cv=5,
    >>>     use_probabilities=True,  # if your model supports predict_proba
    >>>     verbose=True,
    >>> )
    """
    # Stack data and build labels
    n_classes = len(data)
    Xo = np.vstack([ts for ts in data])
    y = np.hstack([i * np.ones(ts.shape[0], dtype=int) for i, ts in enumerate(data)])

    # Durations to evaluate
    durations = np.arange(1, Xo.shape[1] + 1) if overtime else np.array([Xo.shape[1]])

    # Results container
    res = np.zeros((len(durations), 3))

    # Precompute stratification labels if requested
    stratify_labels = y if stratify else None

    rng = np.random.RandomState(random_state)

    for j, duration in enumerate(durations):
        if verbose:
            print(f"→ Duration = {duration}")

        X = Xo[:, :duration]

        # Optional one-off hyper-parameter search per duration
        base_estimator = classifier_factory()
        best_params = None
        if param_grid is not None:
            if verbose:
                print("  Performing GridSearchCV...")
            gs = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
            )
            gs.fit(X, y)
            best_estimator = gs.best_estimator_
            best_params = gs.best_params_
            if verbose:
                print(f"  Best params: {best_params}")
        else:
            best_estimator = base_estimator

        # Bootstrap MI estimates
        mi_vals = np.empty(n_bootstraps)
        for b in range(n_bootstraps):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=rng.randint(0, 2**31 - 1),
            )

            # Fresh clone per bootstrap to avoid leakage/state carry-over
            est = clone(best_estimator) if hasattr(best_estimator, "get_params") else classifier_factory()
            est.fit(X_train, y_train)

            if use_probabilities and hasattr(est, "predict_proba"):
                P_hat = est.predict_proba(X_test)  # (n_samples, n_classes)
                mi_vals[b] = _mi_from_probabilities(y_test, P_hat)
            else:
                y_pred = est.predict(X_test)
                cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes), normalize="true")
                mi_vals[b] = _mi_from_confusion(cm)

        lo, hi = sorted(ci)
        mi_sorted = np.sort(mi_vals)
        res[j, 0] = np.median(mi_vals)
        res[j, 1] = mi_sorted[int(lo * n_bootstraps)]
        res[j, 2] = mi_sorted[int(hi * n_bootstraps)]

        if verbose:
            span = f"[{res[j,1]:.3f}, {res[j,2]:.3f}]"
            if best_params is not None:
                print(f"  MI (median, {lo:.2f}–{hi:.2f}): {res[j,0]:.3f} {span} | params={best_params}")
            else:
                print(f"  MI (median, {lo:.2f}–{hi:.2f}): {res[j,0]:.3f} {span}")

    return res
