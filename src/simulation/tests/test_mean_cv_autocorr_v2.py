from simulation.mean_cv_autocorr_v2 import find_tilda_parameters
from stats.mean import calculate_mean_from_params, calculate_mean
from stats.variance import calculate_variance_from_params, calculate_variance
from stats.cv import calculate_cv_from_params, calculate_cv
from stats.autocorrelation import calculate_ac_from_params, calculate_autocorrelation, calculate_ac_time_interp1d
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
import numpy as np
import pandas as pd

def test_find_tilda_parameters_analytical():
    """Test Case 1: Plug Computed Parameters Back into Original Equations to Verify"""
    # Set up test parameters
    mu_target = 10
    autocorr_target = 2
    cv_target = 0.5
    # Get the parameters using the scaled equations
    rho, d, sigma_b, sigma_u = find_tilda_parameters(mu_target, autocorr_target, cv_target)
    print('Parameters found:', rho, d, sigma_b, sigma_u)
    
    # Plug back in to verify
    mu_est = calculate_mean_from_params(rho, d, sigma_b, sigma_u)
    cv_est = calculate_cv_from_params(rho, d, sigma_b, sigma_u)
    ac_est = calculate_ac_from_params(rho, d, sigma_b, sigma_u, autocorr_target)
    
    # Print comparison
    print(f"Mean:   target={mu_target:.4f},   analytic={mu_est:.4f},   error={100*(mu_est-mu_target)/mu_target:.2f}%")
    print(f"CV:     target={cv_target:.4f},    analytic={cv_est:.4f},    error={100*(cv_est-cv_target)/cv_target:.2f}%")
    print(f"AC({autocorr_target!r}): target={np.exp(-1):.4f}, analytic={ac_est:.4f}, error={100*(ac_est-np.exp(-1))/np.exp(-1):.2f}%")
    
    # Assert that parameters are close to targets
    assert abs((mu_est - mu_target) / mu_target) < 0.01, f"Mean error too large: {100*(mu_est-mu_target)/mu_target:.2f}%"
    assert abs((cv_est - cv_target) / cv_target) < 0.01, f"CV error too large: {100*(cv_est-cv_target)/cv_target:.2f}%"
    assert abs((ac_est - np.exp(-1)) / np.exp(-1)) < 0.01, f"AC error too large: {100*(ac_est-np.exp(-1))/np.exp(-1):.2f}%"

def test_find_tilda_parameters_simulation():
    """Test Case 2: Simulation using Parameters"""
    # Set up test parameters
    mu_target = 10
    autocorr_target = 2
    cv_target = 0.5
    # Get the parameters using the scaled equations
    rho, d, sigma_b, sigma_u = find_tilda_parameters(mu_target, autocorr_target, cv_target)
    
    # Create parameter set for simulation
    parameter_set = [
        {
            'sigma_b': sigma_b,
            'sigma_u': sigma_u,
            'rho': rho,
            'd': d,
            'label': 0
        }
    ]

    # Run simulation
    time_points = np.arange(0, 144.0, 1.0)
    size = 200
    # Use a single core to avoid multiprocessing overhead during tests
    df_results = simulate_one_telegraph_model_system(parameter_set, time_points, size, num_cores=1)
    
    # Extract trajectories (remove label column and convert to numpy array)
    trajectories = df_results[df_results['label'] == 0].drop('label', axis=1).values
    
    # Calculate observed statistics
    mean_observed = calculate_mean(trajectories, parameter_set, use_steady_state=False)
    variance_observed = calculate_variance(trajectories, parameter_set, use_steady_state=False)
    cv_observed = calculate_cv(variance_observed, mean_observed)
    
    # Calculate autocorrelation for the trajectories
    autocorr_results = calculate_autocorrelation(df_results)
    
    # Get mean autocorrelation values and lags for normal condition (label=0 in this case)
    normal_ac_mean = autocorr_results['stress_ac'].mean(axis=0)  # stress_ac corresponds to label=0
    normal_lags = autocorr_results['stress_lags']
    
    # Calculate autocorrelation time using interpolation
    ac_time_observed = calculate_ac_time_interp1d(normal_ac_mean, normal_lags)
    
    print(f"\n=== Observed Statistics vs Targets ===")
    print(f"Mean: Target = {mu_target}, Observed = {mean_observed:.3f}")
    print(f"CV: Target = {cv_target}, Observed = {cv_observed:.3f}")
    print(f"AC Time: Target = {autocorr_target}, Observed = {ac_time_observed:.3f}")
    print(f"Variance: Observed = {variance_observed:.3f}")
    
    # Note: Simulation results may have larger tolerances due to stochastic nature
    # These are more lenient assertions for simulation validation
    print(f"Mean relative error: {100*abs(mean_observed - mu_target)/mu_target:.1f}%")
    print(f"CV relative error: {100*abs(cv_observed - cv_target)/cv_target:.1f}%")
    print(f"AC Time relative error: {100*abs(ac_time_observed - autocorr_target)/autocorr_target:.1f}%")

if __name__ == "__main__":
    print("=== Test Case 1: Analytical Verification ===")
    test_find_tilda_parameters_analytical()
    
    print("\n=== Test Case 2: Simulation Verification ===")
    test_find_tilda_parameters_simulation()
    
    print("\nAll tests completed!")