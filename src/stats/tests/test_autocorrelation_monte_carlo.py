"""Regression test for the telegraph model autocorrelation.

This test compares the closed-form expression implemented in
``calculate_ac_from_params`` against Monte Carlo estimates obtained via
``gillespie_ssa``.  A discrepancy would indicate a bug either in the analytic
derivation or in the simulator parameters.
"""

import numpy as np
from itertools import product

from simulation.gillespie_algorithm import (
    gillespie_ssa,
    telegraph_model_propensity,
    update_matrix,
)
from stats.autocorrelation import calculate_ac_from_params


def estimate_ac_monte_carlo(rho, d, sigma_b, sigma_u, t_lag, *, seed, t_relax=150.0, n_reps=200):
    """Return a Monte Carlo estimate of lagged autocorrelation.

    Each replicate runs a minimal Gillespie simulation with three time points
    (``0``, ``t_relax`` and ``t_relax + t_lag``).  The relaxation time provides a
    burn-in so that the correlation is measured near steady state.  The Pearson
    correlation of the final two counts across ``n_reps`` replicates yields the
    sample autocorrelation at lag ``t_lag``.

    Parameters
    ----------
    rho, d, sigma_b, sigma_u : float
        Telegraph model parameters.
    t_lag : float
        Lag time at which autocorrelation is evaluated.
    seed : int
        Random seed for reproducibility.
    t_relax : float, optional
        Burn-in time to approximate steady state, by default 150.
    n_reps : int, optional
        Number of trajectories to sample, by default 200.

    Returns
    -------
    float
        Sample autocorrelation at lag ``t_lag``.
    """
    np.random.seed(seed)
    pop0 = np.array([1, 0, 0], dtype=int)
    time_points = np.array([0.0, t_relax, t_relax + t_lag])
    m0 = np.empty(n_reps)
    m1 = np.empty(n_reps)
    for i in range(n_reps):
        traj = gillespie_ssa(
            telegraph_model_propensity,
            update_matrix,
            pop0,
            time_points,
            args=(sigma_u, sigma_b, rho, d),
        )
        m0[i] = traj[1, 2]
        m1[i] = traj[2, 2]
    return np.corrcoef(m0, m1)[0, 1]


rho_vals = [5.0, 10.0]
d_vals = [1.0, 2.0]
param_grid = list(product(rho_vals, d_vals))


def test_calculate_ac_matches_gillespie():
    """Check analytic and Monte Carlo autocorrelation across a parameter grid.

    For several combinations of ``rho`` and ``d``, this test evaluates the
    analytical autocorrelation given by ``calculate_ac_from_params`` and
    compares it against the Monte Carlo estimate from ``estimate_ac_monte_carlo``.
    A loose tolerance accounts for stochastic sampling noise while still
    guarding against regression in the analytic expression.
    """
    sigma_b = 0.2
    sigma_u = 0.3
    t_ac = 1.0
    for idx, (rho, d) in enumerate(param_grid):
        ac_expected = calculate_ac_from_params(rho, d, sigma_b, sigma_u, t_ac)
        ac_mc = estimate_ac_monte_carlo(rho, d, sigma_b, sigma_u, t_ac, seed=idx)
        assert np.isclose(ac_mc, ac_expected, rtol=0.2, atol=0.05), (
            f"Mismatch for parameters rho={rho}, d={d}: "
            f"analytical={ac_expected:.3f}, monte_carlo={ac_mc:.3f}"
        )
