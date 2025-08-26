"""Tests for handling ``d_tilda`` roots near the bracket edges.

The suite exercises three scenarios:

* a standard interior root solved by Brent's method,
* a boundary root near ``d=1`` that triggers the linearised l'Hôpital/Taylor
  path, and
* error handling when no valid root exists or when the Taylor slope is nearly
  flat.
"""

import sys
import numpy as np
import pytest

sys.path.append("src")
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import (
    calculate_autocorrelation,
    calculate_ac_time_interp1d,
)


def test_root_inside():
    """The standard solver should succeed for roots away from boundaries."""
    rho, d, sigma_b, sigma_u = find_tilda_parameters(10, 2, 0.5)
    assert abs(d - 0.7835718033111415) < 1e-9


def test_boundary_solver_used(capsys):
    """Boundary roots near ``d=1`` should trigger the linearised solver."""

    mu = 1.0
    cv = np.sqrt(1.001)
    rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, 1.0, cv)
    captured = capsys.readouterr()
    # this message would be printed if the condition is triggered
    assert "Boundary hit" in captured.out
    assert 1.001 < d < 1.002

    # simulate the system to check that we get back the stats properties we asked for
    param_set = [
        {"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}
    ]
    time_points = np.arange(0, 100, 1.0)
    np.random.seed(0)
    df_results = simulate_one_telegraph_model_system(param_set, time_points, size=100, num_cores=1)

    trajectories = df_results[df_results["label"] == 0].drop("label", axis=1).values
    mean_obs = calculate_mean(trajectories, param_set, use_steady_state=False)
    var_obs = calculate_variance(trajectories, param_set, use_steady_state=False)
    cv_obs = calculate_cv(var_obs, mean_obs)

    ac_results = calculate_autocorrelation(df_results)
    ac_mean = ac_results["stress_ac"].mean(axis=0)
    lags = ac_results["stress_lags"]
    ac_time_obs = calculate_ac_time_interp1d(ac_mean, lags)

    assert abs(mean_obs - mu) / mu < 0.2
    assert abs(cv_obs - cv) / cv < 0.2
    assert abs(ac_time_obs - 1.0) / 1.0 < 0.2


def test_no_valid_root():
    """Invalid autocorrelation targets should raise an error."""
    with pytest.raises(ValueError):
        find_tilda_parameters(10, 1.0, 0.5, ac_target=2)
