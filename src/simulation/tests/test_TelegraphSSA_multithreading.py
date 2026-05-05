"""
Wrapper-level checks for the Julia telegraph SSA simulator.

These tests exercise ``simulation.julia_simulate_telegraph_model`` rather than
calling Julia directly, so they match the public simulation API used by
experiments.
"""

import os

import numpy as np

from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
from simulation.mean_cv_t_ac import find_tilda_parameters


os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")


def test_output_shape_correct_many_params_many_traj():
    """Wrapper output should preserve rows, labels, and non-negative counts."""
    n_params = 3
    traj = 20
    expected_rows = n_params * traj

    parameter_sets = [
        {
            "sigma_b": 1.0,
            "sigma_u": 1.0,
            "rho": 10.0,
            "d": 1.0,
            "label": i,
        }
        for i in range(n_params)
    ]
    time_points = np.arange(0, 10, 1.0)

    df = simulate_telegraph_model(parameter_sets, time_points, traj, num_cores=2)

    assert df.shape == (expected_rows, len(time_points) + 1)
    for label_id in range(n_params):
        count = int((df["label"] == label_id).sum())
        assert count == traj

    counts_matrix = df.drop(columns=["label"]).to_numpy()
    assert not np.isnan(counts_matrix).any()
    assert (counts_matrix >= 0).all()


def test_output_shape_correct_single_param_many_traj():
    """A single parameter set should produce one row per trajectory."""
    traj = 40
    rho, d, sigma_b, sigma_u = find_tilda_parameters(10.0, 5.0, 0.5, sigma_sum=5.0)
    parameter_sets = [
        {
            "sigma_b": sigma_b,
            "sigma_u": sigma_u,
            "rho": rho,
            "d": d,
            "label": 0,
        }
    ]
    time_points = np.arange(0.0, 20.0, 1.0)

    df = simulate_telegraph_model(parameter_sets, time_points, traj, num_cores=2)

    assert df.shape == (traj, len(time_points) + 1)
    assert (df["label"] == 0).all()
    counts = df.drop(columns=["label"]).to_numpy()
    assert not np.isnan(counts).any()
    assert (counts >= 0).all()


def test_realistic_simulation_completes():
    """A small multi-condition simulation should complete through the wrapper."""
    targets = [
        (10.0, 5.0, 0.5, "slow"),
        (30.0, 5.0, 0.7, "mid"),
        (50.0, 5.0, 1.0, "fast"),
    ]
    parameter_sets = []
    for mu, t_ac, cv, label in targets:
        rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv, sigma_sum=5.0)
        parameter_sets.append(
            {
                "sigma_b": sigma_b,
                "sigma_u": sigma_u,
                "rho": rho,
                "d": d,
                "label": label,
            }
        )

    time_points = np.arange(0.0, 50.0, 5.0)
    df = simulate_telegraph_model(parameter_sets, time_points, 10, num_cores=2)

    assert df.shape == (len(targets) * 10, len(time_points) + 1)
    assert set(df["label"]) == {"slow", "mid", "fast"}
