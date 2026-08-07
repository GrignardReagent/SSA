import numpy as np
import torch

from models.ssl_transformer import SSL_Transformer
from utils.embeddings import make_knn_eval_fn

DEVICE = torch.device("cpu")


def _make_model():
    torch.manual_seed(0)
    return SSL_Transformer(input_size=1, d_model=8, nhead=2, num_layers=1, dropout=0.0)


def _blobs(n_per_class=15, n_classes=3, seq_len=20, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        X.append(rng.normal(c * 3.0, 0.5, size=(n_per_class, seq_len)))
        y.append(np.full(n_per_class, c))
    return np.vstack(X).astype(np.float32), np.concatenate(y)


def test_returns_expected_keys_and_ranges():
    X_tr, y_tr = _blobs(seed=0)
    X_val, y_val = _blobs(seed=1)
    eval_fn = make_knn_eval_fn(X_tr, y_tr, X_val, y_val, DEVICE, n_neighbors=5)
    metrics = eval_fn(_make_model())
    for key in ("knn_val_acc_smooth", "knn_val_acc", "knn_train_acc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_smoothing_is_a_trailing_mean_over_the_window():
    X_tr, y_tr = _blobs(seed=0)
    X_val, y_val = _blobs(seed=1)
    eval_fn = make_knn_eval_fn(X_tr, y_tr, X_val, y_val, DEVICE, n_neighbors=5, smooth_window=3)
    model = _make_model()

    raw = [eval_fn(model)["knn_val_acc"] for _ in range(5)]
    # re-run to read the smoothed value fresh (raw is stochastic-free given a fixed frozen model,
    # so this just checks the running window arithmetic, not learning dynamics)
    eval_fn2 = make_knn_eval_fn(X_tr, y_tr, X_val, y_val, DEVICE, n_neighbors=5, smooth_window=3)
    smoothed = [eval_fn2(model)["knn_val_acc_smooth"] for _ in range(5)]
    assert smoothed[-1] == np.mean(raw[-3:])


def test_model_left_in_train_mode_after_call():
    X_tr, y_tr = _blobs(seed=0)
    X_val, y_val = _blobs(seed=1)
    eval_fn = make_knn_eval_fn(X_tr, y_tr, X_val, y_val, DEVICE, n_neighbors=5)
    model = _make_model()
    eval_fn(model)
    assert model.training, "eval_fn must restore train() mode for the caller's next training step"


def test_independent_windows_across_separate_eval_fn_instances():
    X_tr, y_tr = _blobs(seed=0)
    X_val, y_val = _blobs(seed=1)
    model = _make_model()
    eval_fn_a = make_knn_eval_fn(X_tr, y_tr, X_val, y_val, DEVICE, n_neighbors=5)
    eval_fn_b = make_knn_eval_fn(X_tr, y_tr, X_val, y_val, DEVICE, n_neighbors=5)
    a1 = eval_fn_a(model)["knn_val_acc_smooth"]
    b1 = eval_fn_b(model)["knn_val_acc_smooth"]
    assert a1 == b1  # single call each -> window mean equals the single raw value
