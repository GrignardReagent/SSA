#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import tqdm
from simulation.mean_var_autocorr_sigma_u import find_parameters
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems
from utils.steady_state import save_steady_state

###############################################################################
# 1) Define target mean, CV and autocorrelations, and some parameters to start with
###############################################################################
cv_target_normal = 0.13  # Fixed normal CV
mu_target = 0.05                 # Mean (same for both)
cv_ratios = np.arange(0.1, 3.0, 0.01)
autocorr_target = 100

parameters = {
    "stress": {"sigma_u": 0.02},
    "normal": {"sigma_u": 0.01}
}

# Initialize list to store all parameter sets
all_parameter_sets = []

###############################################################################
# Compute parameters for normal condition ONCE (since it doesn't change)
###############################################################################
try:
    rho_normal, sigma_b_normal, d_normal = find_parameters(
        parameters["normal"], mu_target, cv_target=cv_target_normal, autocorr_target=autocorr_target
    )
    normal_results = {"rho": rho_normal, "sigma_b": sigma_b_normal, "d": d_normal}
    print(f"[NORMAL] ✅ Found: {normal_results}")
except ValueError as e:
    print(f'{e}')
    print(f"[NORMAL] ❌ No suitable solution found.")
    normal_results = None

###############################################################################
# 2) Loop over different CV ratios (only compute stress condition each time)
###############################################################################
for ratio in tqdm.tqdm(cv_ratios, desc="Running CV Ratio Simulations"):
    # For the stress condition, we define cv_target_stress by ratio
    cv_target_stress = ratio * cv_target_normal

    # Only compute for stress condition
    try:
        rho_stress, sigma_b_stress, d_stress = find_parameters(
            parameters["stress"], mu_target, cv_target=cv_target_stress, autocorr_target=autocorr_target
        )
        stress_results = {"rho": rho_stress, "sigma_b": sigma_b_stress, "d": d_stress}
        print(f"[STRESS] ✅ Found: {stress_results}")
        
        # Only proceed if we have results for both conditions
        if normal_results is not None:
            parameter_sets = [
                {"cv_ratio": ratio,
                 "sigma_u": parameters["stress"]["sigma_u"], 
                 "sigma_b": stress_results['sigma_b'], 
                 "rho": stress_results['rho'], 
                 "d": stress_results['d'], 
                 "label": 0},
                
                {"cv_ratio": ratio,
                 "sigma_u": parameters["normal"]["sigma_u"], 
                 "sigma_b": normal_results['sigma_b'], 
                 "rho": normal_results['rho'], 
                 "d": normal_results['d'], 
                 "label": 1}
            ]

            # Add to the collection
            all_parameter_sets.extend(parameter_sets)

            # Output the results
            print("Updated Parameter Sets:", parameter_sets)
            
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
                df_results = simulate_two_telegraph_model_systems(parameter_sets, extended_time_points, size)

                output_dir = f"data/mRNA_trajectories_variance_{int(cv_target_stress)}_{int(cv_target_normal)}"
                os.makedirs(output_dir, exist_ok=True)
                
                # save full time series
                output_file = f"{output_dir}/m_traj_{cv_target_stress}_{cv_target_normal}_{i}.csv"
                df_results.to_csv(output_file, index=False)

                # get only the steady state part of the data
                save_path = f'{output_dir}/steady_state_trajectories/'
                remaining_time_points, steady_state_series = save_steady_state(output_file, parameter_sets, time_points, save_path=save_path)                  
            
    except ValueError as e:
        print(f'{e}')
        print(f"[STRESS] ❌ No suitable solution found.")

# Save all parameter sets calculated to CSV
df_params = pd.DataFrame(all_parameter_sets)
param_out_path = f'data/cv_parameter_sets_{cv_target_normal}.csv'
df_params.to_csv(param_out_path, index=False)
print(f"Saved {len(all_parameter_sets)} parameter sets to {param_out_path}")