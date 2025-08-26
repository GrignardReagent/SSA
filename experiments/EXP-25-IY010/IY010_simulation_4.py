#!/usr/bin/env python3
"""
Sample parameter space using Sobol sequences to test solve_tilda_parameters.

This script generates a quasi-random, evenly distributed set of parameter
combinations for targets ``mu``, ``t_ac`` and ``cv`` using a Sobol sequence.
Each sample is simulated and the resulting statistics are compared against the
targets.

This script uses the find_tilda_parameters() function by:
1. Finding parameters for combinations of mu, t_ac, and cv targets
2. Simulating mRNA trajectories using those parameters
3. Comparing observed vs target statistics
4. Saving trajectory data and results to CSV files

Timepoint is set to: int(t_ac * 20)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
from scipy.stats import qmc
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d
import traceback
from tqdm import tqdm
np.random.seed(42) # for reproducibility. 
    
# Create data directory if it doesn't exist
data_dir = "data_4" # different folder to the first attempt.
os.makedirs(data_dir, exist_ok=True)

# Define parameter ranges and other simulation parameters
# build Sobol seq (better space‑filling with fewer points)
N = 1000 
sobol = qmc.Sobol(d=3, scramble=True, seed=42)
U = sobol.random_base2(int(np.ceil(np.log2(N))))[:N]  # N points in [0,1)^3

mu_target  = qmc.scale(U[:,0:1], [1], [10_000])      # map to [1, 10000]
t_ac_target = qmc.scale(U[:,1:2], [0.5], [100])      # map to [0.5, 100]
cv_target  = qmc.scale(U[:,2:3], [0.5], [5.0])      # map to [0.5, 5.0]

# Flatten the arrays since qmc.scale returns 2D arrays
mu_target = mu_target.flatten()
t_ac_target = t_ac_target.flatten()
cv_target = cv_target.flatten()
sigma_sum = 1 # unused, but is a default for find_tilda_parameters
max_runtime = 15 * 60

############## START OF CODE ##############

# Track success/failure
success_count = 0
failure_count = 0
total_combinations = len(mu_target)  # since they're aligned Sobol triples
all_records = []

print(f"Testing {total_combinations} parameter combinations...")
print(f"Started at: {datetime.now()}")

# Loop through every combination
for combination_idx, (mu, t_ac, cv) in enumerate(
    tqdm(zip(mu_target, t_ac_target, cv_target), total=total_combinations, desc='Simulating Sobol samples'),
    start=1):

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
        't_ac_rel_error_pct': np.nan,
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
        time_points = np.arange(0, 
                                int(t_ac * 20), # need to set a minimum timespan so that for small t_ac, there are still enough timepoints
                                1.0)
        size = 200 # TODO: may need to reconsider the choice of this to be adaptive.
        
        # time and limit the simulation time
        start_time = time.time()
        # Run simulation
        df_results = simulate_one_telegraph_model_system(parameter_set, time_points, size)
        if time.time() - start_time > max_runtime:
            raise RuntimeError(f"simulate_one_telegraph_model_system exceeded the runtime limit of {max_runtime} s.")
        
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
        t_ac_rel_error = abs(ac_time_observed - t_ac) / t_ac
        
        # Update result record with observed values
        result_record.update({
            'success': True,
            'mu_observed': mean_observed,
            'cv_observed': cv_observed,
            't_ac_observed': ac_time_observed,
            'variance_observed': variance_observed,
            'mean_rel_error_pct': mean_rel_error * 100,
            'cv_rel_error_pct': cv_rel_error * 100,
            't_ac_rel_error_pct': t_ac_rel_error * 100
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
        print(f"  Errors: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}")
        
        # Check tolerances
        if mean_rel_error < 0.2 and cv_rel_error < 0.2 and t_ac_rel_error < 0.2:
            print("  ✅ All assertions passed")
            success_count += 1
        else:
            print("  ❌ Tolerance check failed")
            result_record['error_message'] = f"Tolerance exceeded: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
            failure_count += 1
            
    except Exception as e:
        failure_count += 1
        error_msg = str(e)
        result_record['error_message'] = error_msg
        print(f"  FAILED: mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f} - Error: {error_msg}")
    
    # save and append result_record to csv file
    results_df = pd.DataFrame([result_record])  # Single record
    all_records.append(result_record)
    results_path = os.path.join(data_dir, "IY010_simulation_parameters_4.csv")
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
all_results_df = pd.DataFrame(all_records)
successful_results = all_results_df[all_results_df['success'] == True]
if len(successful_results) > 0:
    print(f"\n=== Error Statistics (Successful Runs Only) ===")
    print(f"Mean error - Mean: {successful_results['mean_rel_error_pct'].mean():.2f}%, Std: {successful_results['mean_rel_error_pct'].std():.2f}%")
    print(f"CV error - Mean: {successful_results['cv_rel_error_pct'].mean():.2f}%, Std: {successful_results['cv_rel_error_pct'].std():.2f}%")
    print(f"AC error - Mean: {successful_results['t_ac_rel_error_pct'].mean():.2f}%, Std: {successful_results['t_ac_rel_error_pct'].std():.2f}%")
    