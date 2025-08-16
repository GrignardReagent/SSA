"""Exploratory utility to add measurement noise to synthetic trajectories.

This script generates a handful of telegraph-model mRNA trajectories using the
existing stochastic simulation utilities and then injects additive Gaussian noise
into the sequences. The noise mimics experimental measurement artifacts such as
camera readout or photon shot noise.  Data are not saved; the script merely
prints the first few noisy trajectories for quick inspection.

Usage
-----
PYTHONPATH=src python experiments/EXP-25-IY010/IY010_add_measurement_noise.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems


def add_gaussian_noise(trajectories: np.ndarray, snr: float = 5.0) -> np.ndarray:
    """Return trajectories with zero-mean Gaussian noise added.

    Parameters
    ----------
    trajectories:
        Array of shape ``[n_traj, T]`` containing mRNA counts.
    snr:
        Desired signal-to-noise ratio.  Noise standard deviation is computed as
        ``traj_std / snr`` so higher values give milder noise.
    """
    traj_std = trajectories.std(axis=1, keepdims=True)
    noise = np.random.normal(0.0, traj_std / snr, size=trajectories.shape)
    noisy = trajectories + noise
    return np.clip(noisy, a_min=0.0, a_max=None)


def main() -> None:
    """Generate a few example trajectories and corrupt them with noise."""
    # Two simple parameter sets to emulate different expression regimes
    parameter_sets = [
        {"sigma_u": 0.02, "sigma_b": 0.06, "rho": 0.5, "d": 0.1, "label": 0},
        {"sigma_u": 0.04, "sigma_b": 0.12, "rho": 0.5, "d": 0.1, "label": 1},
    ]

    # Simulate three trajectories per class without saving them to disk
    time_points = np.arange(0, 60.0, 1.0)
    df = simulate_two_telegraph_model_systems(parameter_sets, time_points, size=3, num_cores=1)

    # Drop the labels for noise injection then re-attach afterwards
    labels = df["label"].to_numpy()
    trajectories = df.drop(columns="label").to_numpy(dtype=float)

    noisy = add_gaussian_noise(trajectories)
    noisy_df = pd.DataFrame(noisy, columns=df.columns[1:])
    noisy_df.insert(0, "label", labels)

    print("Noisy trajectories (first five rows):")
    print(noisy_df.head())


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
