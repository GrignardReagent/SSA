import numpy as np
from simulation.mean_cv_autocorr_v2 import find_sigma_sum
from stats.mean import calculate_mean_from_params
from stats.cv import calculate_cv_from_params
from stats.autocorrelation import calculate_ac_from_params


def test_find_sigma_sum():
    # Choose a representative target system
    mu_target = 1.0
    t_ac_target = 0.5
    cv_target = 1.5

    # Search for the smallest sigma_sum that reproduces the targets
    sigma_sum, rho, d, sigma_b, sigma_u = find_sigma_sum(
        mu_target, t_ac_target, cv_target, tolerance=1e-3
    )

    # Recompute the metrics from the returned parameters to verify that the
    # helper indeed met the requested targets
    mu_est = calculate_mean_from_params(rho, d, sigma_b, sigma_u)
    cv_est = calculate_cv_from_params(rho, d, sigma_b, sigma_u)
    ac_est = calculate_ac_from_params(rho, d, sigma_b, sigma_u, t_ac_target)

    # Mean and CV are compared relative to the target to ensure high accuracy
    assert abs((mu_est - mu_target) / mu_target) < 1e-6
    assert abs((cv_est - cv_target) / cv_target) < 1e-6
    # Autocorrelation target is absolute as exp(-1) is a concrete value
    assert abs(ac_est - np.exp(-1)) < 1e-3
    # sigma_sum should be a positive scaling factor
    assert sigma_sum > 0
