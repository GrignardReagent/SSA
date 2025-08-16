#!/usr/bin/env python3
"""Benchmark CVmCherry Transformer on clean vs noise-corrupted synthetic data.

The script synthesises a modest set of telegraph-model trajectories, trains the
`CVmCherryTransformer` on the clean sequences and then evaluates the trained
classifier on both the original data and on versions corrupted with additive
Gaussian noise to mimic measurement artefacts.

Usage
-----
PYTHONPATH=src python experiments/EXP-25-IY010/IY010_noise_benchmark.py
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.simple_transformer import CVmCherryTransformer, ModelCfg
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems


# ---------------------------------------------------------------------------
# Data generation and normalisation utilities
# ---------------------------------------------------------------------------

def add_gaussian_noise(trajectories: np.ndarray, snr: float = 5.0) -> np.ndarray:
    """Return ``trajectories`` with zero-mean Gaussian noise added.

    The noise standard deviation is ``traj_std / snr`` per trajectory so higher
    ``snr`` values correspond to milder corruption.  Negative values are clipped
    to zero to respect count data.
    """
    traj_std = trajectories.std(axis=1, keepdims=True)
    noise = np.random.normal(0.0, traj_std / snr, size=trajectories.shape)
    noisy = trajectories + noise
    return np.clip(noisy, a_min=0.0, a_max=None)


def _prepare_dataset(series: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """Normalise ``series`` per trajectory and package as a dataset."""
    series_t = torch.tensor(series, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    lengths = (series_t != 0).sum(1)
    max_len = series_t.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    mean = (series_t * mask).sum(1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    var = ((series_t - mean).pow(2) * mask).sum(1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    std = var.sqrt()
    series_t = (series_t - mean) / (std + 1e-8)
    series_t[~mask] = 0.0
    series_t = series_t.unsqueeze(-1)
    return TensorDataset(series_t, lengths, labels_t)


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: CVmCherryTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """Run a single epoch and return ``(loss, accuracy)``."""
    loss_sum, correct = 0.0, 0
    for x, lengths, y in loader:
        logits = model(x, lengths)
        loss = criterion(logits, y)
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, correct / n


def main() -> None:
    # Two simple regimes with distinct mean expression levels
    parameter_sets = [
        {"sigma_u": 0.02, "sigma_b": 0.06, "rho": 0.5, "d": 0.1, "label": 0},
        {"sigma_u": 0.04, "sigma_b": 0.12, "rho": 0.5, "d": 0.1, "label": 1},
    ]
    time_points = np.arange(0.0, 60.0, 1.0)
    df = simulate_two_telegraph_model_systems(parameter_sets, time_points, size=50, num_cores=1)
    labels = df["label"].to_numpy()
    clean_series = df.drop(columns="label").to_numpy(dtype=float)

    clean_ds = _prepare_dataset(clean_series, labels)
    noisy_ds = _prepare_dataset(add_gaussian_noise(clean_series), labels)

    model = CVmCherryTransformer(ModelCfg(n_classes=2))
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(clean_ds, batch_size=32, shuffle=True)
    for epoch in range(1, 6):
        loss, acc = _run_epoch(model, train_loader, criterion, optimiser)
        print(f"epoch {epoch:02d} loss={loss:.4f} acc={acc:.3f}")

    model.eval()
    eval_loader = DataLoader(clean_ds, batch_size=32)
    _, acc_clean = _run_epoch(model, eval_loader, criterion)
    noisy_loader = DataLoader(noisy_ds, batch_size=32)
    _, acc_noisy = _run_epoch(model, noisy_loader, criterion)
    print(f"clean accuracy: {acc_clean:.3f}\nnoisy accuracy: {acc_noisy:.3f}")


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
