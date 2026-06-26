"""catch22 feature extraction for time-series arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def extract_catch22(series_list, desc: str = "", leave: bool = False) -> pd.DataFrame:
    """Extract the 22 canonical catch22 features for each time series.

    Parameters
    ----------
    series_list:
        Iterable of 1-D time-series arrays.
    desc:
        Optional progress-bar label.
    leave:
        Whether the tqdm progress bar remains after completion.

    Returns
    -------
    pd.DataFrame
        One row per input series and one column per catch22 feature. Infinite
        values are converted to NaN so callers can impute or drop consistently.
    """
    import pycatch22

    rows = []
    for series in tqdm(series_list, desc=desc, leave=leave):
        values = np.asarray(series, dtype=float).ravel().tolist()
        out = pycatch22.catch22_all(values)
        rows.append(dict(zip(out["names"], out["values"])))
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def fill_feature_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill non-finite feature values using train-split medians.

    Residual NaNs, such as all-NaN feature columns, are filled with zero after
    the train medians are applied.
    """
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    medians = train_df.median()
    return train_df.fillna(medians).fillna(0.0), test_df.fillna(medians).fillna(0.0)


def extract_catch22_from_loader(loader, desc: str = "") -> tuple[pd.DataFrame, np.ndarray]:
    """Extract catch22 features and labels from a PyTorch DataLoader."""
    series = []
    labels = []

    for X_batch, y_batch in tqdm(loader, desc=desc, leave=False):
        X_np = X_batch.cpu().numpy().squeeze()
        y_np = y_batch.cpu().numpy().ravel()
        X_np = np.atleast_2d(X_np)

        for i in range(X_np.shape[0]):
            series.append(X_np[i])
            labels.append(y_np[i])

    return extract_catch22(series, desc=desc), np.asarray(labels)


def run_catch22_svm(train_loader, test_loader, exp_name: str = "Experiment") -> float:
    """Extract catch22 features from loaders, fit an RBF SVM, and return accuracy."""
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    print(f"\n=== Running Catch22 + SVM on {exp_name} ===")
    X_train, y_train = extract_catch22_from_loader(train_loader, f"{exp_name} (Train)")
    X_test, y_test = extract_catch22_from_loader(test_loader, f"{exp_name} (Test)")

    if len(X_train) == 0:
        return 0.0

    X_train, X_test = fill_feature_frame(X_train, X_test)
    print(f"   Extracted {X_train.shape[1]} C22 features.")
    print("   Training SVM...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])
    pipe.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"{exp_name} Catch22 Accuracy: {acc:.2%}")
    print("-" * 30)
    return acc


def run_catch22_series_svm(
    dataset: dict,
    dataset_tag: str,
    random_state: int = 42,
    report: bool = True,
) -> tuple[float, np.ndarray]:
    """Run catch22 + RBF-SVM on a prepared train/test series dictionary.

    The expected dictionary keys are ``series_train``, ``series_test``,
    ``y_train``, ``y_test`` and ``class_names``. This matches the IY031
    experimental-condition notebooks.
    """
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    print(f"\n=== Catch22 + SVM (RBF) -- {dataset_tag} ===")
    X_train = extract_catch22(dataset["series_train"], desc="Train")
    X_test = extract_catch22(dataset["series_test"], desc="Test")
    X_train, X_test = fill_feature_frame(X_train, X_test)
    print(f"Feature matrix: {X_train.shape[1]} features")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)),
    ])
    pipe.fit(X_train, dataset["y_train"])

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(dataset["y_test"], y_pred)
    chance = 1 / len(dataset["class_names"])
    print(f"Accuracy: {acc:.4f}  (chance={chance:.4f})")

    if report:
        print(
            classification_report(
                dataset["y_test"],
                y_pred,
                target_names=dataset["class_names"],
            )
        )

    return acc, y_pred


def _catch22_values(x: np.ndarray) -> list:
    """Compute catch22 values for one 1-D trajectory."""
    import pycatch22

    return pycatch22.catch22_all(np.asarray(x, dtype=float).ravel().tolist())["values"]


def extract_catch22_pair(X_np: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """Compute concatenated catch22 features for each pair in a sequence array."""
    from joblib import Parallel, delayed

    half = X_np.shape[1] // 2
    x1_list = [X_np[i, :half, 0] for i in range(len(X_np))]
    x2_list = [X_np[i, half:, 0] for i in range(len(X_np))]

    f1 = Parallel(n_jobs=n_jobs)(delayed(_catch22_values)(x) for x in x1_list)
    f2 = Parallel(n_jobs=n_jobs)(delayed(_catch22_values)(x) for x in x2_list)

    features = np.array([a + b for a, b in zip(f1, f2)], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0)


catch22_features = extract_catch22_pair
