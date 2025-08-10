import numpy as np
from simulation.mean_cv_autocorr_v2 import find_tilda_parameters
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d


def test_find_tilda_parameters_matches_simulation():
    """Parameters returned by ``find_tilda_parameters`` should reproduce
    the requested statistics when used in stochastic simulations.
    The test uses a small system to keep execution time reasonable
    while still verifying mean, CV and autocorrelation time."""

    mu_target = 10.0
    t_ac_target = 1.5
    cv_target = 1.0

    # Derive kinetic parameters; the seed speeds up convergence for this regime
    rho, d, sigma_b, sigma_u = find_tilda_parameters(
        mu_target, t_ac_target, cv_target, sigma_sum_seed=20.0
    )

    param_set = [
        {
            "sigma_b": sigma_b,
            "sigma_u": sigma_u,
            "rho": rho,
            "d": d,
            "label": 0,
        }
    ]

    time_points = np.arange(0, 200, 1.0)
    np.random.seed(0)
    df_results = simulate_one_telegraph_model_system(
        param_set, time_points, size=100, num_cores=1
    )

    trajectories = df_results[df_results["label"] == 0].drop("label", axis=1).values
    mean_obs = calculate_mean(trajectories, param_set, use_steady_state=False)
    var_obs = calculate_variance(trajectories, param_set, use_steady_state=False)
    cv_obs = calculate_cv(var_obs, mean_obs)

    ac_results = calculate_autocorrelation(df_results)
    ac_mean = ac_results["stress_ac"].mean(axis=0)
    lags = ac_results["stress_lags"]
    ac_time_obs = calculate_ac_time_interp1d(ac_mean, lags)

    # Allow small deviations due to stochasticity in the simulation
    assert abs(mean_obs - mu_target) / mu_target < 0.2
    assert abs(cv_obs - cv_target) / cv_target < 0.2
    assert abs(ac_time_obs - t_ac_target) / t_ac_target < 0.2
