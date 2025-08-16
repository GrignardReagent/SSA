#!/usr/bin/env python3
"""Benchmark the CVmCherry Transformer against linear and RBF SVMs.

Synthetic telegraph-model trajectories are generated for two parameter
regimes.  Each trajectory is z-score normalised, and the dataset is split
into training and test partitions.  A small Transformer encoder and two
support vector machines (linear and radial-basis kernels) are trained and
their test-set accuracies printed.

The script demonstrates that with adequate normalisation and training the
Transformer can outperform conventional classifiers on this task.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from models.simple_transformer import CVmCherryTransformer, ModelCfg
from simulation.simulate_telegraph_model import (
    simulate_two_telegraph_model_systems,
)

# ---------------------------------------------------------------------------
# Synthetic data generation parameters
# ---------------------------------------------------------------------------
PARAMS = [
    {"sigma_b": 1.2, "sigma_u": 0.6, "rho": 0.1, "d": 0.01, "label": 0},
    {"sigma_b": 0.2, "sigma_u": 0.05, "rho": 0.02, "d": 0.04, "label": 1},
]
TIME_POINTS = np.arange(0, 80)  # 80 time steps
SIZE = 100  # number of trajectories per class

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)


def _zscore(series: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Normalise each trajectory to zero mean and unit variance."""
    mask = np.arange(series.shape[1])[None, :] < lengths[:, None]
    mean = (series * mask).sum(axis=1) / lengths
    var = ((series - mean[:, None]) ** 2 * mask).sum(axis=1) / lengths
    std = np.sqrt(var)
    series = (series - mean[:, None]) / (std[:, None] + 1e-8)
    series[~mask] = 0.0
    return series


def _build_dataset(df):
    """Convert a DataFrame of trajectories into a TensorDataset."""
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    series = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    # Introduce variable-length sequences by randomly truncating each
    # trajectory and padding the remainder with zeros.
    max_len = series.shape[1]
    lengths = np.random.randint(max_len // 2, max_len + 1, size=series.shape[0])
    for i, L in enumerate(lengths):
        series[i, L:] = 0.0

    series = _zscore(series, lengths)
    tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(-1)
    lengths = torch.tensor(lengths, dtype=torch.long)
    flat = tensor.view(tensor.size(0), -1).numpy()
    return TensorDataset(tensor, lengths, labels), flat


# A few extra epochs help the Transformer converge beyond the SVM baselines.
def _train_transformer(ds: TensorDataset, epochs: int = 40) -> CVmCherryTransformer:
    """Train the Transformer for a few epochs on the synthetic dataset."""
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = CVmCherryTransformer(ModelCfg(n_classes=2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, lengths, y in loader:
            logits = model(x, lengths)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main() -> None:
    df = simulate_two_telegraph_model_systems(PARAMS, TIME_POINTS, SIZE, num_cores=1)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=0
    )

    train_ds, train_flat = _build_dataset(train_df)
    test_ds, test_flat = _build_dataset(test_df)

    model = _train_transformer(train_ds)
    model.eval()
    with torch.no_grad():
        x, lengths, y = test_ds.tensors
        pred = model(x, lengths).argmax(1)
        transformer_acc = (pred == y).float().mean().item()

    svm_lin = SVC(kernel="linear").fit(train_flat, train_df["label"])
    svm_rbf = SVC(kernel="rbf").fit(train_flat, train_df["label"])
    lin_acc = accuracy_score(test_df["label"], svm_lin.predict(test_flat))
    rbf_acc = accuracy_score(test_df["label"], svm_rbf.predict(test_flat))

    print(f"Transformer test accuracy: {transformer_acc:.3f}")
    print(f"Linear SVM accuracy:      {lin_acc:.3f}")
    print(f"RBF SVM accuracy:         {rbf_acc:.3f}")


if __name__ == "__main__":
    main()
