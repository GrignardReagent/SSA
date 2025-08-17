#!/usr/bin/env python3
"""
Test solve_tilda_parameters function across parameter space.

This script tests the _solve_tilda_parameters function by:
1. Finding parameters for combinations of mu, t_ac, and cv targets
2. Simulating mRNA trajectories using those parameters
3. Comparing observed vs target statistics
4. Saving trajectory data and results to CSV files

Author: Generated from IY010_test_solve_tilda.ipynb
Date: August 2025
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d
    
# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Define parameter ranges
mu_target = np.logspace(0, 4, 10)  # 1 to 10000
t_ac_target = np.logspace(-0.3, 2, 10)  # ~0.5 to 100
cv_target = np.logspace(-0.3, 0.7, 20)  # ~0.5 to 5
sigma_sum = 1

# Track success/failure
success_count = 0
failure_count = 0
total_combinations = len(mu_target) * len(t_ac_target) * len(cv_target)

print(f"Testing {total_combinations} parameter combinations...")
print(f"Started at: {datetime.now()}")

# Loop through every combination
combination_idx = 0
for mu in mu_target:
    for t_ac in t_ac_target:
        for cv in cv_target:
            combination_idx += 1
            
            # Progress update
            if combination_idx % 100 == 0:
                print(f"PROGRESS: {combination_idx}/{total_combinations} ({100*combination_idx/total_combinations:.1f}%)")
            
            # Initialize result record
            result_record = {
                'mu_target': mu,
                't_ac_target': t_ac,
                'cv_target': cv,
                'sigma_sum': sigma_sum,
                'success': False,
                'error_message': '',
                # System parameters
                'rho': np.nan,
                'd': np.nan,
                'sigma_b': np.nan,
                'sigma_u': np.nan,
                # Observed statistics
                'mu_observed': np.nan,
                'cv_observed': np.nan,
                't_ac_observed': np.nan,
                'variance_observed': np.nan,
                # Relative errors (in percentage)
                'mean_rel_error_pct': np.nan,
                'cv_rel_error_pct': np.nan,
                'ac_rel_error_pct': np.nan,
                # Trajectory filename
                'trajectory_filename': ''
            }
            
            try:
                # Get the parameters using the scaled equations
                rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
                
                # Store system parameters
                result_record.update({
                    'rho': rho,
                    'd': d,
                    'sigma_b': sigma_b,
                    'sigma_u': sigma_u
                })
                
                print(f"Testing mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f}")
                print(f"  Parameters: rho={rho:.4f}, d={d:.4f}, sigma_b={sigma_b:.4f}, sigma_u={sigma_u:.4f}")
                
                # Set up simulation parameters
                parameter_set = [{
                    'sigma_b': sigma_b,
                    'sigma_u': sigma_u,
                    'rho': rho,
                    'd': d,
                    'label': 0
                }]
                
                # Make sure time points are sufficient for autocorrelation calculation
                time_points = np.arange(0, max(144, int(t_ac * 20)), 1.0)
                size = 200
                
                # Run simulation
                df_results = simulate_one_telegraph_model_system(parameter_set, time_points, size)
                
                # Extract normal trajectories (remove label column and convert to numpy array)
                trajectories = df_results[df_results['label'] == 0].drop('label', axis=1).values
                
                # Calculate observed statistics
                mean_observed = calculate_mean(trajectories, parameter_set, use_steady_state=True)
                variance_observed = calculate_variance(trajectories, parameter_set, use_steady_state=True)
                cv_observed = calculate_cv(variance_observed, mean_observed)
                
                # Calculate autocorrelation
                autocorr_results = calculate_autocorrelation(df_results)
                ac_mean = autocorr_results['stress_ac'].mean(axis=0)  # stress_ac corresponds to label=0
                lags = autocorr_results['stress_lags']
                ac_time_observed = calculate_ac_time_interp1d(ac_mean, lags)
                
                # Calculate relative errors
                mean_rel_error = abs(mean_observed - mu) / mu
                cv_rel_error = abs(cv_observed - cv) / cv
                ac_rel_error = abs(ac_time_observed - t_ac) / t_ac
                
                # Update result record with observed values
                result_record.update({
                    'success': True,
                    'mu_observed': mean_observed,
                    'cv_observed': cv_observed,
                    't_ac_observed': ac_time_observed,
                    'variance_observed': variance_observed,
                    'mean_rel_error_pct': mean_rel_error * 100,
                    'cv_rel_error_pct': cv_rel_error * 100,
                    'ac_rel_error_pct': ac_rel_error * 100
                })
                
                # Save trajectory data
                trajectory_filename = f"mRNA_trajectories_{mu:.3f}_{cv:.3f}_{t_ac:.3f}.csv"
                trajectory_path = os.path.join(data_dir, trajectory_filename)
                
                # Create DataFrame for trajectories with time points as index
                trajectory_df = pd.DataFrame(trajectories, columns=[f't_{i}' for i in range(len(time_points))])
                trajectory_df.to_csv(trajectory_path, index=False)
                
                result_record['trajectory_filename'] = trajectory_filename
                
                print(f"  Target: mu={mu:.3f}, cv={cv:.3f}, t_ac={t_ac:.3f}")
                print(f"  Observed: mu={mean_observed:.3f}, cv={cv_observed:.3f}, t_ac={ac_time_observed:.3f}")
                print(f"  Errors: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={ac_rel_error:.1%}")
                
                # Check tolerances
                if mean_rel_error < 0.2 and cv_rel_error < 0.2 and ac_rel_error < 0.2:
                    print("  ✅ All assertions passed")
                    success_count += 1
                else:
                    print("  ❌ Tolerance check failed")
                    result_record['error_message'] = f"Tolerance exceeded: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={ac_rel_error:.1%}"
                    failure_count += 1
                    
            except Exception as e:
                failure_count += 1
                error_msg = str(e)
                result_record['error_message'] = error_msg
                print(f"  FAILED: mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f} - Error: {error_msg}")
            
            # save and append result_record to csv file
            results_df = pd.DataFrame([result_record])  # Single record
            results_path = os.path.join(data_dir, "IY010_simulation_parameters.csv")
            # Use header only for the first write
            write_header = not os.path.exists(results_path)
            results_df.to_csv(results_path, mode='a', header=write_header, index=False)

# Print final summary
print(f"\n=== Final Results ===")
print(f"Total combinations tested: {total_combinations}")
print(f"Successful runs: {success_count}")
print(f"Failed runs: {failure_count}")
print(f"Success rate: {100*success_count/total_combinations:.1f}%")
print(f"Results saved to: {results_path}")
print(f"Completed at: {datetime.now()}")

# Print some statistics on errors for successful runs
successful_results = results_df[results_df['success'] == True]
if len(successful_results) > 0:
    print(f"\n=== Error Statistics (Successful Runs Only) ===")
    print(f"Mean error - Mean: {successful_results['mean_rel_error_pct'].mean():.2f}%, Std: {successful_results['mean_rel_error_pct'].std():.2f}%")
    print(f"CV error - Mean: {successful_results['cv_rel_error_pct'].mean():.2f}%, Std: {successful_results['cv_rel_error_pct'].std():.2f}%")
    print(f"AC error - Mean: {successful_results['ac_rel_error_pct'].mean():.2f}%, Std: {successful_results['ac_rel_error_pct'].std():.2f}%")