#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import tqdm
from simulation.mean_var_autocorr import find_parameters
from simulation.mean_cv_autocorr import quick_find_parameters
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems, simulate_one_telegraph_model_system
from utils.steady_state import save_steady_state

###############################################################################
# 1) Define target mean, CV and autocorrelations, and some parameters to start with
###############################################################################
cv_target = 1.30  # Fixed normal CV
mu_target = 5                 # Mean (same for both)
autocorr_target = 1
parameters = {"sigma_b": 0.5}

###############################################################################
# 2) Compute parameters 
#########################################################

try:
    rho, sigma_u, d = find_parameters(
        parameters, mu_target, cv_target=cv_target, autocorr_target=autocorr_target
    )
    results = {"rho": rho, "sigma_u": sigma_u, "d": d}
    print(f"✅ Found: {results}")
except ValueError as e:
    print(f'{e}')
    print(f"❌ No suitable solution found.")
    results = None

# Only proceed if we have results for both conditions
if results is not None:
    parameter_sets = [
        {
            "sigma_b": parameters["sigma_b"], 
            "sigma_u": results['sigma_u'], 
            "rho": results['rho'], 
            "d": results['d'], 
            "label": 0
        }
    ]
    # Output the results
    print("Updated Parameter Sets:", parameter_sets)
    
    # Save all parameter sets calculated to CSV
    df_params = pd.DataFrame(parameter_sets)
    param_out_path = f'data/parameter_sets_{cv_target}_{mu_target}_{autocorr_target}.csv'
    df_params.to_csv(param_out_path, index=False)
    print(f"Saved {len(parameter_sets)} parameter sets to {param_out_path}")

    # Simulation parameters
    min_d = min(pset['d'] for pset in parameter_sets)
    steady_state_time = int(10 / min_d)
    time_points = np.arange(0, 144.0, 1.0)
    extended_time_points = np.arange(
        time_points[0],
        len(time_points) + steady_state_time,
        time_points[1] - time_points[0]
    )
    size = 200
    num_iterations = 10
    
###########################################################################
# 3) Simulate & Save data
###########################################################################  
    for i in range(num_iterations):
        df_results = simulate_one_telegraph_model_system(parameter_sets, extended_time_points, size)

        output_dir = f"data/mRNA_trajectories_{cv_target}_{mu_target}_{autocorr_target}"
        os.makedirs(output_dir, exist_ok=True)
        
        # save full time series
        output_file = f"{output_dir}/m_traj_{cv_target}_{i}.csv"
        df_results.to_csv(output_file, index=False)

        # get only the steady state part of the data
        save_path = f'{output_dir}/steady_state_trajectories/'
        remaining_time_points, steady_state_series = save_steady_state(output_file, parameter_sets, time_points, save_path=save_path)                  
