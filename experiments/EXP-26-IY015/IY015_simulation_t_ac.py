#!/usr/bin/env python3

"""
Simulate a temporal control dataset where Mean and CV are FIXED, 
but Autocorrelation Time (Tac) varies. This effectively isolates 
temporal dynamics as the only distinguishing feature.

This script also extracts steady state files as npz files.
"""

import os
import subprocess
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
import time
from scipy.stats import qmc
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d
import traceback
from tqdm import tqdm

# SS

from utils.steady_state import find_steady_state

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
GLOBAL_SEED = 42
rng = np.random.default_rng(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = "data_t_ac_variation"
os.makedirs(data_dir, exist_ok=True)

# Define Fixed Parameters 
MU_FIXED = 1000.0
CV_FIXED = 0.5

# 2. Define Variable Parameter Bounds (The "Physics")
TAC_MIN = 5.0
TAC_MAX = 120.0

# Sobol coverage
N = 48
sobol = qmc.Sobol(d=1, scramble=True, seed=GLOBAL_SEED)
U = sobol.random_base2(int(np.ceil(np.log2(N))))[:N]   # shape (N, 1)

# Scale t_ac to range
t_ac_target = qmc.scale(U, [TAC_MIN], [TAC_MAX]).flatten()

# Create arrays for fixed parameters
mu_target = np.full(N, MU_FIXED)
cv_target = np.full(N, CV_FIXED)

max_runtime = 15 * 60 # seconds
size = 1000  # per parameter set
time_points = np.arange(0, 3000, 1.0)

# Results log CSV
results_path = os.path.join(data_dir, "IY015_simulation_t_ac_parameters_sobol.csv")
# start fresh if exists
if os.path.exists(results_path):
    # os.remove(results_path)
    subprocess.run(['sudo', 'rm', results_path], check=True)

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
success_count = 0
failure_count = 0
total_combinations = len(mu_target) 
all_records = []

print(f"=== Generating Temporal Control Dataset ===")
print(f"Fixed Parameters: Mu={MU_FIXED}, CV={CV_FIXED}")
print(f"Varying t_ac: [{TAC_MIN}, {TAC_MAX}] via Sobol sampling")
print(f"Total Datasets: {total_combinations}")
print(f"Started at: {datetime.now()}")

for combination_idx, (mu, t_ac, cv) in enumerate(
    tqdm(zip(mu_target, t_ac_target, cv_target), 
         total=total_combinations, desc="Simulating Sobol samples (t_ac variation)"),
        start=1,
):
    record = {
        "mu_target": mu,
        "t_ac_target": t_ac,
        "cv_target": cv,
        "success": False,
        "error_message": "",
        "rho": np.nan,
        "d": np.nan,
        "sigma_b": np.nan,
        "sigma_u": np.nan,
        "mu_observed": np.nan,
        "cv_observed": np.nan,
        "t_ac_observed": np.nan,
        "variance_observed": np.nan,
        "mean_rel_error_pct": np.nan,
        "cv_rel_error_pct": np.nan,
        "t_ac_rel_error_pct": np.nan,
        "trajectory_filename": "",
    }

    try:
        # --- Solve parameters for targets ---
        rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
        record.update({"rho": rho, "d": d, "sigma_b": sigma_b, "sigma_u": sigma_u})
        print(f"Testing mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f}")
        print(f"Parameters found: rho={rho:.4f}, d={d:.4f}, sigma_b={sigma_b:.4f}, sigma_u={sigma_u:.4f}")
        
        # --- Simulate ---
        parameter_set = [
            {"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}
        ]
        # time the simulation 
        start_time = time.time()
        df_results = simulate_telegraph_model(parameter_set, time_points, size)
        
        if time.time() - start_time > max_runtime:
            raise RuntimeError(
                f"simulate_telegraph_model exceeded the runtime limit of {max_runtime} s."
            )
        # extract label-free trajectories
        trajectories = df_results[df_results["label"] == 0].drop("label", axis=1).values
        
        # stat calculations
        mean_observed = calculate_mean(trajectories, parameter_set, use_steady_state=True)
        variance_observed = calculate_variance(trajectories, parameter_set, use_steady_state=True)
        cv_observed = calculate_cv(variance_observed, mean_observed)

        autocorr_results = calculate_autocorrelation(df_results)
        ac_mean = autocorr_results["stress_ac"].mean(axis=0)
        lags = autocorr_results["stress_lags"]
        ac_time_observed = calculate_ac_time_interp1d(ac_mean, lags)

        mean_rel_error = abs(mean_observed - mu) / mu
        cv_rel_error = abs(cv_observed - cv) / cv
        t_ac_rel_error = abs(ac_time_observed - t_ac) / t_ac
        # end of calculations

        record.update(
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

        trajectory_filename = f"mRNA_trajectories_{mu:.3f}_{cv:.3f}_{t_ac:.3f}.csv"
        trajectory_path = os.path.join(data_dir, trajectory_filename)
        trajectory_df = pd.DataFrame(trajectories, columns=[f"t_{i}" for i in range(len(time_points))])
        # save traj file to CSV
        try: 
            trajectory_df.to_csv(trajectory_path, index=False)
        except PermissionError:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                tmp_path = tmp_file.name
                trajectory_df.to_csv(tmp_path, index=False)
            # Move temp file to final location with sudo
            subprocess.run(['sudo', 'mv', tmp_path, trajectory_path], check=True)
            subprocess.run(['sudo', 'chown', f'{os.getenv("USER")}:{os.getenv("USER")}', trajectory_path], check=True)
        
        record["trajectory_filename"] = trajectory_filename

        print(f"  Target: mu={mu:.3f}, cv={cv:.3f}, t_ac={t_ac:.3f}")
        print(f"  Observed: mu={mean_observed:.3f}, cv={cv_observed:.3f}, t_ac={ac_time_observed:.3f}")
        print(f"  Errors: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}")

        # tolerance checks
        if (mean_rel_error < 0.2) and (cv_rel_error < 0.2) and (t_ac_rel_error < 0.2):
            success_count += 1
        else:
            record["error_message"] = (
                f"Tolerance exceeded: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
            )
            failure_count += 1

    except Exception as e:
        failure_count += 1
        error_msg = str(e)
        record["error_message"] = error_msg
        print(f"  FAILED: mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f} - Error: {error_msg}")

    # append results to rolling CSV
    try:
        pd.DataFrame([record]).to_csv(results_path, mode="a",
                                    header=not os.path.exists(results_path), index=False)
    except PermissionError:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
            pd.DataFrame([record]).to_csv(tmp_path, index=False, header=True)
        
        # Append temp file to results file with sudo
        if os.path.exists(results_path):
            # Append without header if file exists
            subprocess.run(['sudo', 'sh', '-c', f'tail -n +2 "{tmp_path}" >> "{results_path}"'], check=True)
        else:
            # Copy entire file if results file doesn't exist yet
            subprocess.run(['sudo', 'cp', tmp_path, results_path], check=True)
            subprocess.run(['sudo', 'chown', f'{os.getenv("USER")}:{os.getenv("USER")}', results_path], check=True)
        
        # Clean up temp file
        os.unlink(tmp_path)
                            
# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Control Data Generation Complete ===")
print(f"Total combinations: {total_combinations}")
print(f"Successful runs: {success_count}")
print(f"Failed runs: {failure_count}")
print(f"Data saved to: {data_dir}")


# -----------------------------------------------------------------------------
# Data Processing for Machine Learning (Post-Simulation): Steady State Extraction, NPZ Saving
# -----------------------------------------------------------------------------
print("\n=== Starting Data Processing for ML ===")

# Define paths using the existing configuration variables
# Ensure DATA_ROOT uses the same 'data_dir' defined at the top of your script
DATA_ROOT = Path(data_dir) 
# Use the results_path defined earlier in your script
RESULTS_PATH = Path(results_path) 

if not RESULTS_PATH.exists():
    print(f"Warning: Results file not found at {RESULTS_PATH}. Skipping processing.")
else:
    df_params = pd.read_csv(RESULTS_PATH)
    
    # Reconstruct the list of file paths and parameter sets from the CSV
    # This ensures we process exactly what was just simulated
    traj_paths = [DATA_ROOT / fname for fname in df_params['trajectory_filename']]
    
    # Rebuild the parameter_sets list needed for find_steady_state
    # Structure: List of [ { "sigma_b": ..., "rho": ... } ]
    parameter_sets = []
    for _, row in df_params.iterrows():
        p_set = [{
            "sigma_b": row['sigma_b'],
            "sigma_u": row['sigma_u'],
            "rho": row['rho'],
            "d": row['d'],
            "label": 0
        }]
        parameter_sets.append(p_set)

    # --- Step 1: Calculate Steady State Indices ---
    print("Calculating steady state indices for alignment...")
    ss_index_list = []
    for params in parameter_sets:
        # find_steady_state returns (distribution, index)
        _, ss_index = find_steady_state(params[0])
        ss_index_list.append(ss_index)

    # Find the maximum steady state index to cut all trajectories to the same length
    if ss_index_list:
        max_ss_index = max(ss_index_list)
        print(f"Max steady state index determined: {max_ss_index}")
        
        # Calculate the new time points vector corresponding to the sliced data
        # 'time_points' is the global variable defined in your configuration section
        new_time_points = time_points[max_ss_index:]
        
        # --- Step 2: Slice and Save as NPZ ---
        print(f"Processing {len(traj_paths)} files into .npz format...")
        
        for traj_file, params in tqdm(zip(traj_paths, parameter_sets), total=len(traj_paths), desc="Saving NPZs"):
            if not traj_file.exists():
                continue
                
            try:
                # Load the raw CSV trajectory data
                df_traj = pd.read_csv(traj_file)
                
                # Remove label column if present
                df_traj = df_traj.drop(columns=['label'], errors='ignore')
                
                # Truncate to the maximum steady state start time
                # This ensures all ML inputs have the same sequence length
                if max_ss_index < df_traj.shape[1]:
                    df_traj = df_traj.iloc[:, max_ss_index:]
                
                trajectories = df_traj.values.astype(np.float32)
                
                # Prepare dictionary for NPZ
                trajectory_data = {
                    'trajectories': trajectories,
                    'time_points': new_time_points.astype(np.float32),
                    'size': int(trajectories.shape[0]),
                    'parameters': params,
                }
                
                # Save compressed NPZ
                npz_path = traj_file.with_suffix('.npz')
                try:
                    np.savez_compressed(npz_path, **trajectory_data)
                except PermissionError:
                    # Handle permissions if running in a restricted environment
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
                        tmp_path = tmp_file.name
                        np.savez_compressed(tmp_path, **trajectory_data)
                    
                    # Move temp file to final location with sudo
                    subprocess.run(['sudo', 'mv', tmp_path, str(npz_path)], check=True)
                    subprocess.run(['sudo', 'chown', f'{os.getenv("USER")}:{os.getenv("USER")}', str(npz_path)], check=True)
                    
            except Exception as e:
                print(f"Error processing {traj_file.name}: {e}")

    print("Data processing complete.")