
import unittest
import numpy as np
import pandas as pd
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
from simulation.mean_cv_t_ac import find_tilda_parameters

class TestAutocorrelationEstimation(unittest.TestCase):
    
    def setUp(self):
        # Reproducibility
        self.rng = np.random.default_rng(42)
        np.random.seed(42)  # For legacy random usage if any
        
        # Simulation parameters
        # Use reasonable parameters for unit testing: enough for stability, small enough for speed
        self.time_points = np.arange(0, 2500, 1.0)
        self.n_reps = 40 
        self.mu_target = 100.0
        self.cv_target = 0.5  # Fixed CV for these tests

    def _run_test_case(self, target_t_ac, tolerance=0.3):
        """Helper to run a single estimation test case using the real simulator."""
        print(f"\n--- Testing Target T_ac={target_t_ac} ---")
        
        # 1. Solve for telegraph parameters
        try:
            rho, d, sigma_b, sigma_u = find_tilda_parameters(self.mu_target, target_t_ac, self.cv_target)
        except Exception as e:
            self.fail(f"Failed to solve parameters for mu={self.mu_target}, t_ac={target_t_ac}, cv={self.cv_target}: {e}")

        # Note: simulating label=0 (stress condition) usually
        parameter_set = [
            {
                "sigma_b": sigma_b, 
                "sigma_u": sigma_u, 
                "rho": rho, 
                "d": d, 
                "label": 0
            }
        ]
        
        # 2. Run Simulation
        try:
            # The simulator returns a DataFrame with columns like '0', '1', ... (time points) and 'label'
            # Or it might return named time columns depending on the implementation.
            # Usually: label, t0, t1, t2...
            df_results = simulate_telegraph_model(parameter_set, self.time_points, self.n_reps)
        except Exception as e:
             self.fail(f"Simulation failed: {e}")
        
        # 3. Calculate Stats
        # calculate_autocorrelation expects specific structure. 
        # If stationary=True, it subtracts the global mean.
        autocorr_results = calculate_autocorrelation(df_results, stationary=True)
        
        self.assertIn('stress_ac', autocorr_results, "Autocorrelation calculation failed to return 'stress_ac'")
        
        ac_mean = autocorr_results["stress_ac"].mean(axis=0)
        lags = autocorr_results["stress_lags"]
        
        # 4. Calculate T_ac
        ac_time_observed = calculate_ac_time_interp1d(ac_mean, lags)
        
        # 5. Check error
        error = abs(ac_time_observed - target_t_ac) / target_t_ac
        
        print(f"Target T_ac: {target_t_ac}, Observed: {ac_time_observed:.2f}, Error: {error:.2%}")
        
        self.assertLess(error, tolerance, 
            f"Observed T_ac ({ac_time_observed:.2f}) deviates from target ({target_t_ac}) by more than {tolerance:.0%} (Error: {error:.2%})")

    def test_short_autocorrelation(self):
        """Test estimation for short autocorrelation time (5.0)."""
        self._run_test_case(5.0, tolerance=0.25)

    def test_medium_autocorrelation(self):
        """Test estimation for medium autocorrelation time (20.0)."""
        self._run_test_case(20.0, tolerance=0.25)

    def test_long_autocorrelation(self):
        """Test estimation for long autocorrelation time (50.0)."""
        self._run_test_case(50.0, tolerance=0.25)
        
    def test_very_long_autocorrelation(self):
        """Test estimation for very long autocorrelation time (100.0)."""
        # Longer timescales might require more data or have larger variance in estimation
        self._run_test_case(100.0, tolerance=0.35)

    def test_robustness_to_tail_noise(self):
        """Test that the estimator ignores noise in the tail (after 1/e) using synthetic data."""
        # This test remains synthetic because we want to deliberately inject bad noise 
        # that a real simulator might not produce deterministically.
        lags = np.arange(0, 100, 1.0)
        target_t = 10.0
        ac_clean = np.exp(-lags / target_t)
        
        # Add a bump in the tail that would confuse a full-range interpolator
        ac_noisy = ac_clean.copy()
        ac_noisy[40:60] = 0.5 
        
        ac_time = calculate_ac_time_interp1d(ac_noisy, lags)
        error = abs(ac_time - target_t) / target_t
        
        print(f"\nRobustness Test: Target={target_t}, Observed={ac_time:.2f}, Error={error:.2%}")
        self.assertLess(error, 0.05, "Estimator was confused by tail noise")

if __name__ == '__main__':
    unittest.main()
