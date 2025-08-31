#!/usr/bin/env python3
"""
Simulate a mini batch of Sobol sequence samples with mean (mu) and CV fixed.

This script is adapted from ``IY010_simulation_7_mu.py``. The only difference
is that the target mean ``mu`` and ``cv`` are fixed to a constant value, while the other
statistics (autocorrelation time ``t_ac``) are sampled using a Sobol sequence.
The purpose of this is to create time series difficult to classify by a classifier that doesnt recognise temporal patterns e.g., SVM

- ``mu`` is fixed to ``MU_FIXED``
- ``cv`` is fixed to ``CV_FIXED``
- ``t_ac`` sampled from 0.5 to 100
- Uses 48 Sobol samples
- Each parameter set simulates 200 trajectories
- Constant time points: t_ac * 10 + 144, with 1.0 intervals; t_ac*10 is for making sure that we hit steady state.

Results (parameters and trajectories) are saved in a dedicated folder for this
experiment.
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

np.random.seed(42)  # for reproducibility

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MU_FIXED = 1000  # constant mean value for all runs
CV_FIXED = 1.0   # constant CV value for all runs

# Create data directory if it doesn't exist
# Each simulation script writes to its own folder to avoid overwriting results
# from other experiments.
data_dir = "data_7_mu_cv_fixed"
os.makedirs(data_dir, exist_ok=True)

# Define parameter ranges and other simulation parameters
N = 48  # number of Sobol samples
sobol = qmc.Sobol(d=2, scramble=True, seed=42)  # two varying parameters
U = sobol.random_base2(int(np.ceil(np.log2(N))))[:N]  # N points in [0,1)^2

# Map Sobol sequence to parameter ranges
# ``t_ac`` in [0.5, 100] 
t_ac_target = qmc.scale(U[:, 0:1], [0.5], [100]).flatten()
cv_target = np.full(N, CV_FIXED)
mu_target = np.full(N, MU_FIXED)  # same ``mu`` for all samples

sigma_sum = 1  # unused but required by ``find_tilda_parameters``
max_runtime = 15 * 60  # 15 minutes per simulation as a safety measure

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
success_count = 0
failure_count = 0
total_combinations = len(mu_target)
all_records = []

print(f"Testing {total_combinations} parameter combinations...")
print(f"Started at: {datetime.now()}")

# Iterate over each Sobol sample
for combination_idx, (mu, t_ac, cv) in enumerate(
    tqdm(zip(mu_target, t_ac_target, cv_target), total=total_combinations, desc="Simulating Sobol samples"),
    start=1,
):
    # Record structure to store results for this parameter set
    result_record = {
        "mu_target": mu,
        "t_ac_target": t_ac,
        "cv_target": cv,
        "sigma_sum": sigma_sum,
        "success": False,
        "error_message": "",
        # System parameters
        "rho": np.nan,
        "d": np.nan,
        "sigma_b": np.nan,
        "sigma_u": np.nan,
        # Observed statistics
        "mu_observed": np.nan,
        "cv_observed": np.nan,
        "t_ac_observed": np.nan,
        "variance_observed": np.nan,
        # Relative errors (percentage)
        "mean_rel_error_pct": np.nan,
        "cv_rel_error_pct": np.nan,
        "t_ac_rel_error_pct": np.nan,
        # Trajectory filename
        "trajectory_filename": "",
    }

    try:
        # Convert statistical targets to model parameters
        rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
        result_record.update({"rho": rho, "d": d, "sigma_b": sigma_b, "sigma_u": sigma_u})

        print(f"Testing mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f}")
        print(f"  Parameters: rho={rho:.4f}, d={d:.4f}, sigma_b={sigma_b:.4f}, sigma_u={sigma_u:.4f}")

        # Build simulation parameter set
        parameter_set = [
            {"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}
        ]

        # Ensure enough time points for autocorrelation estimation
        time_points = np.arange(0, 
                                (t_ac * 10) + 144,
                                1.0)
        size = 200  # number of trajectories

        start_time = time.time()
        df_results = simulate_one_telegraph_model_system(parameter_set, time_points, size)
        if time.time() - start_time > max_runtime:
            raise RuntimeError(
                f"simulate_one_telegraph_model_system exceeded the runtime limit of {max_runtime} s."
            )

        # Remove label column and convert to numpy array
        trajectories = df_results[df_results["label"] == 0].drop("label", axis=1).values

        # Compute observed statistics
        mean_observed = calculate_mean(trajectories, parameter_set, use_steady_state=True)
        variance_observed = calculate_variance(trajectories, parameter_set, use_steady_state=True)
        cv_observed = calculate_cv(variance_observed, mean_observed)

        autocorr_results = calculate_autocorrelation(df_results)
        ac_mean = autocorr_results["stress_ac"].mean(axis=0)  # stress_ac corresponds to label=0
        lags = autocorr_results["stress_lags"]
        ac_time_observed = calculate_ac_time_interp1d(ac_mean, lags)

        # Relative errors
        mean_rel_error = abs(mean_observed - mu) / mu
        cv_rel_error = abs(cv_observed - cv) / cv
        t_ac_rel_error = abs(ac_time_observed - t_ac) / t_ac

        result_record.update(
            {
                "success": True,
                "mu_observed": mean_observed,
                "cv_observed": cv_observed,
                "t_ac_observed": ac_time_observed,
                "variance_observed": variance_observed,
                "mean_rel_error_pct": mean_rel_error * 100,
                "cv_rel_error_pct": cv_rel_error * 100,
                "t_ac_rel_error_pct": t_ac_rel_error * 100,
            }
        )

        # Save trajectories to CSV
        trajectory_filename = f"mRNA_trajectories_{mu:.3f}_{cv:.3f}_{t_ac:.3f}.csv"
        trajectory_path = os.path.join(data_dir, trajectory_filename)
        trajectory_df = pd.DataFrame(trajectories, columns=[f"t_{i}" for i in range(len(time_points))])
        trajectory_df.to_csv(trajectory_path, index=False)
        result_record["trajectory_filename"] = trajectory_filename

        print(f"  Target: mu={mu:.3f}, cv={cv:.3f}, t_ac={t_ac:.3f}")
        print(
            f"  Observed: mu={mean_observed:.3f}, cv={cv_observed:.3f}, t_ac={ac_time_observed:.3f}"
        )
        print(
            f"  Errors: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
        )

        if mean_rel_error < 0.2 and cv_rel_error < 0.2 and t_ac_rel_error < 0.2:
            print("  ✅ All assertions passed")
            success_count += 1
        else:
            print("  ❌ Tolerance check failed")
            result_record["error_message"] = (
                f"Tolerance exceeded: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
            )
            failure_count += 1

    except Exception as e:
        failure_count += 1
        error_msg = str(e)
        result_record["error_message"] = error_msg
        print(f"  FAILED: mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f} - Error: {error_msg}")

    # Append record and persist to CSV after each run
    all_records.append(result_record)
    results_df = pd.DataFrame([result_record])
    results_path = os.path.join(data_dir, "IY010_simulation_parameters_7_mu_cv_fixed.csv")
    write_header = not os.path.exists(results_path)
    results_df.to_csv(results_path, mode="a", header=write_header, index=False)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Final Results ===")
print(f"Total combinations tested: {total_combinations}")
print(f"Successful runs: {success_count}")
print(f"Failed runs: {failure_count}")
