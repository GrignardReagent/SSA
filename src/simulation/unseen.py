"""Helpers for generating held-out synthetic telegraph-model classes."""

from __future__ import annotations

import numpy as np

from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
from simulation.mean_cv_t_ac import find_tilda_parameters


def generate_unseen_classes(
    n_classes: int = 5,
    n_trajs_per_class: int = 20,
    seq_len: int = 3000,
) -> list[dict]:
    """Generate synthetic mystery classes with random telegraph parameters."""
    print(f"Generating {n_classes} unseen classes...")
    unseen_data = []

    # Sample from the broad biological ranges used by baseline experiments.
    mus = np.random.uniform(1, 10000, n_classes)
    cvs = np.random.uniform(0.5, 2.0, n_classes)
    tacs = np.random.uniform(5, 120, n_classes)
    time_points = np.arange(0, seq_len, 1.0)

    for i in range(n_classes):
        try:
            rho, d, sigma_b, sigma_u = find_tilda_parameters(mus[i], tacs[i], cvs[i])
            params = [
                {
                    "sigma_b": sigma_b,
                    "sigma_u": sigma_u,
                    "rho": rho,
                    "d": d,
                    "label": 0,
                }
            ]
            df = simulate_telegraph_model(params, time_points, n_trajs_per_class)
            trajs = df.drop(columns=["label"], errors="ignore").values

            unseen_data.append(
                {
                    "class_id": f"Mystery_Class_{i}",
                    "parameters": {"mu": mus[i], "cv": cvs[i], "tac": tacs[i]},
                    "trajectories": trajs,
                }
            )
            print(f"  Generated Class {i}: Mu={mus[i]:.1f}, CV={cvs[i]:.2f}")
        except Exception as exc:
            print(f"  Skipped a class due to solver error: {exc}")

    return unseen_data
