import numpy as np

from utils.embeddings import fit_predict_knn


def _imbalanced_blobs(n_major=200, n_minor=20, seed=0):
    """Two overlapping Gaussian blobs, one class 10x rarer than the other.

    The overlap is what makes the prior matter: in the contested region a
    plain vote is decided by which class brought more neighbours.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, 1.0, size=(n_major, 2)),
                   rng.normal(1.5, 1.0, size=(n_minor, 2))])
    y = np.array([0] * n_major + [1] * n_minor)
    return X, y


def test_prior_correction_recovers_minority_recall():
    X_tr, y_tr = _imbalanced_blobs()
    X_te, y_te = _imbalanced_blobs(n_major=100, n_minor=100, seed=1)  # balanced test

    plain = fit_predict_knn(X_tr, y_tr, X_te, prior_correction=False)
    corrected = fit_predict_knn(X_tr, y_tr, X_te, prior_correction=True)

    minority = y_te == 1
    assert (corrected[minority] == 1).mean() > (plain[minority] == 1).mean()


def test_prior_correction_is_a_no_op_on_a_balanced_fit_set():
    # equal priors divide every column by the same constant, so argmax is unmoved
    rng = np.random.default_rng(2)
    X_tr = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(3, 1, (50, 2))])
    y_tr = np.array([0] * 50 + [1] * 50)
    X_te = rng.normal(1.5, 1, (40, 2))
    assert np.array_equal(fit_predict_knn(X_tr, y_tr, X_te, prior_correction=False),
                          fit_predict_knn(X_tr, y_tr, X_te, prior_correction=True))


def test_predictions_are_valid_labels_and_shaped_correctly():
    X_tr, y_tr = _imbalanced_blobs()
    X_te, _ = _imbalanced_blobs(seed=3)
    for pc in (False, True):
        pred = fit_predict_knn(X_tr, y_tr, X_te, prior_correction=pc)
        assert pred.shape == (len(X_te),)
        assert set(np.unique(pred)).issubset(set(np.unique(y_tr)))


def test_uncorrected_path_matches_sklearn_exactly():
    # the default must remain plain KNN, since IY035 and the v1 scripts rely on it
    from sklearn.neighbors import KNeighborsClassifier

    X_tr, y_tr = _imbalanced_blobs()
    X_te, _ = _imbalanced_blobs(seed=4)
    ref = KNeighborsClassifier(n_neighbors=10, metric="euclidean").fit(X_tr, y_tr)
    assert np.array_equal(fit_predict_knn(X_tr, y_tr, X_te), ref.predict(X_te))
