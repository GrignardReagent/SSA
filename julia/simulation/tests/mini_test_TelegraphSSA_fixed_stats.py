#!/usr/bin/env python3

# import the simulation functions
import multiprocessing
import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
num_cores = multiprocessing.cpu_count()
os.environ["JULIA_NUM_THREADS"] = str(num_cores)  # or set to desired number of threads
import numpy as np
import pandas as pd 
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
# plotting modules
import matplotlib.pyplot as plt
from visualisation.plots import plot_mRNA_dist

'''
Mini Test script (mini version of test_TelegraphSSA_fixed_stats.py) to replot distribution plots: testing whether the julia implementation of the telegraph SSA can produce trajectorie s matching a set of prescribed stats.
Python package modularised simulation function ``simulate_telegraph_model`` was used for simulation. 
'''

def run_simulation_test(mu_target, t_ac_target, cv_target, size=1000, output_dir = "mini_test_telegraph_fixed_stats_results"):
    """Run a single simulation test for given target statistics."""
    ###############################################################
    # Simulation
    ###############################################################
    print(f"\n{'='*60}")
    print(f"Testing: μ={mu_target}, t_ac={t_ac_target}, CV={cv_target}")
    print(f"{'='*60}")
    
    # Get the parameters using the scaled equations
    rho, d, sigma_b, sigma_u = find_tilda_parameters(mu_target, t_ac_target, cv_target)
    print(f"Parameters: ρ={rho:.6f}, d={d:.6f}, σ_b={sigma_b:.6f}, σ_u={sigma_u:.6f}")
    
    # Prepare parameter sets
    parameter_sets = [{
        'sigma_b': sigma_b,
        'sigma_u': sigma_u,
        'rho': rho,
        'd': d,
        'label': 0
    }]
    
    # Set time points - longer simulation for better autocorrelation estimation
    time_points = np.arange(0, 10_000, 1.0)
    
    # simulation
    df = simulate_telegraph_model(parameter_sets, time_points, size)
    
    ###############################################################
    # Plotting 
    ###############################################################
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot data distribution
    fig, ax = plot_mRNA_dist(parameter_sets, df, kde=False)
    # Save figure with descriptive filename
    dist_filename = f"{output_dir}/telegraph_test_mu{mu_target}_tac{t_ac_target}_cv{cv_target}_dist.png"
    fig.savefig(dist_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    print(f"Data distribution plot saved to: {dist_filename}")

    return {
        'mu_target': mu_target,
        't_ac_target': t_ac_target,
        'cv_target': cv_target,
        'parameters': {'rho': rho, 'd': d, 'sigma_b': sigma_b, 'sigma_u': sigma_u}
    }

###############################################################
# Main execution
###############################################################

# Define test targets
mu_targets = [10, 100]
t_ac_targets = [1, 5]
cv_targets = [0.5, 1.0]
size = 1000  # number of trajectories per simulation
output_dir = "mini_test_telegraph_fixed_stats_results"

# Run all combinations
results = []
total_tests = len(mu_targets) * len(t_ac_targets) * len(cv_targets)
current_test = 0

for mu_target in mu_targets:
    for t_ac_target in t_ac_targets:
        for cv_target in cv_targets:
            current_test += 1
            print(f"\nRunning test {current_test}/{total_tests}")
            
            try:
                result = run_simulation_test(mu_target, t_ac_target, cv_target, size=size, output_dir=output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error in test μ={mu_target}, t_ac={t_ac_target}, CV={cv_target}: {e}")
                continue

# Create summary table
print(f"\n{'='*80}")
print("SUMMARY OF ALL TESTS")
print(f"{'='*80}")

summary_df = pd.DataFrame(results)
if not summary_df.empty:

    # Save summary to CSV
    summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    print(f"\nSummary saved to: {output_dir}/summary_results.csv")

else:
    print("No successful tests completed.")

print(f"\nAll tests completed. Results saved in '{output_dir}' directory.")
