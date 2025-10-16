#!/usr/bin/env python3

# import the simulation functions
import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ["JULIA_NUM_THREADS"] = "8"  # or set to desired number of threads
from juliacall import Main as jl
import numpy as np
import pandas as pd 
from simulation.mean_cv_t_ac import find_tilda_parameters
# import the stats functions
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d
# plotting modules
import matplotlib.pyplot as plt
from visualisation.plots import plot_mRNA_trajectory, plot_mRNA_dist

'''
Test script: testing whether the julia implementation of the telegraph SSA can produce trajectorie s matching a set of prescribed stats.
'''

def run_simulation_test(mu_target, t_ac_target, cv_target, size=1000):
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
    
    # Python → Julia conversion handled automatically
    jl.parameter_sets = parameter_sets
    jl.time_points = time_points
    
    # Run the simulation in Julia
    jl.seval(f'df = simulate_telegraph_model(parameter_sets, time_points, {size})')
    
    # Convert Julia DataFrame directly to Python
    labels = np.array(jl.seval('Int64.(df.label)'))
    counts_matrix = np.array(jl.seval('Int64.(Matrix(df[:, Not(:label)]))'))
    
    # Create pandas DataFrame
    df_labels = pd.DataFrame(labels, columns=['label'])
    df_counts = pd.DataFrame(counts_matrix, columns=[f"time_{ti}" for ti in time_points])
    df = pd.concat([df_labels, df_counts], axis=1)
    
    ###############################################################
    # Now calculate the stats from the simulated data
    ###############################################################
    mean_observed = calculate_mean(df_counts.T, parameter_sets, use_steady_state=False) # note the df is transposed
    variance_observed = calculate_variance(df_counts.T, parameter_sets, use_steady_state=False)
    cv_observed = calculate_cv(variance_observed, mean_observed)
    
    # Calculate autocorrelation
    autocorr_results = calculate_autocorrelation(df)
    ac_mean = autocorr_results['stress_ac'].mean(axis=0)
    lags = autocorr_results['stress_lags']
    t_ac_observed = calculate_ac_time_interp1d(ac_mean, lags)

    # Calculate errors
    mean_error_pct = 100 * abs(np.mean(mean_observed) - mu_target) / mu_target
    cv_error_pct = 100 * abs(np.mean(cv_observed) - cv_target) / cv_target
    t_ac_error_pct = 100 * abs(t_ac_observed - t_ac_target) / t_ac_target

    ###############################################################
    # Reporting 
    ###############################################################
    # Print results
    print(f"Mean: Target = {mu_target}, Observed = {np.mean(mean_observed):.3f} (Error: {mean_error_pct:.1f}%)")
    print(f"CV: Target = {cv_target}, Observed = {np.mean(cv_observed):.3f} (Error: {cv_error_pct:.1f}%)")
    print(f"Variance: Observed = {np.mean(variance_observed):.3f}")
    print(f"AC Time: Target = {t_ac_target}, Observed = {t_ac_observed:.3f} (Error: {t_ac_error_pct:.1f}%)")
    
    ###############################################################
    # Plotting 
    ###############################################################
    # Create output directory if it doesn't exist
    output_dir = "test_telegraph_fixed_stats_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save traj plot
    fig, ax = plot_mRNA_trajectory(parameter_sets, time_points, df_counts)  
    # Save figure with descriptive filename
    traj_filename = f"{output_dir}/telegraph_test_mu{mu_target}_tac{t_ac_target}_cv{cv_target}_traj.png"
    fig.savefig(traj_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    print(f"Trajectory plot saved to: {traj_filename}")

    # Plot data distribution
    fig, ax = plot_mRNA_dist(parameter_sets, df_counts)
    # Save figure with descriptive filename
    dist_filename = f"{output_dir}/telegraph_test_mu{mu_target}_tac{t_ac_target}_cv{cv_target}_dist.png"
    fig.savefig(dist_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    print(f"Data distribution plot saved to: {dist_filename}")

    # Plot autocorrelation
    plt.figure(figsize=(10, 6))
    # Find indices for positive lags only (autocorrelation is symmetric)
    positive_mask = lags >= 0
    positive_lags = lags[positive_mask]
    positive_mean_ac = ac_mean[positive_mask]
    # Limit to first 200 points
    max_lag_points = min(200, len(positive_lags))
    plot_lags = positive_lags[:max_lag_points]
    plot_ac = positive_mean_ac[:max_lag_points]
    # Plot stress condition (zoomed in)
    plt.plot(plot_lags, plot_ac, color='blue', label=f'AC Time: {t_ac_observed:.2f}')
    # Plot the AC time lines, show the values (only if within the plot range)
    plt.axhline(y=1/np.e, color='gray', linestyle='--', label='AC = 1/e')
    if t_ac_observed <= plot_lags[-1]:  # Only show vertical line if it's within the plot range
        plt.axvline(x=t_ac_observed, color='blue', linestyle='--', alpha=0.7)
    plt.title('Autocorrelation (First 200 Lag Points)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.xlim(0, plot_lags[-1])  # Set x-axis limits explicitly
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # saving the figure
    tac_filename = f"{output_dir}/telegraph_test_mu{mu_target}_tac{t_ac_target}_cv{cv_target}_tac.png"
    plt.savefig(tac_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"Autocorrelation plot saved to: {tac_filename}")
    
    
    return {
        'mu_target': mu_target,
        't_ac_target': t_ac_target,
        'cv_target': cv_target,
        'mu_observed': np.mean(mean_observed),
        'cv_observed': np.mean(cv_observed),
        't_ac_observed': t_ac_observed,
        'variance_observed': np.mean(variance_observed),
        'mean_error_pct': mean_error_pct,
        'cv_error_pct': cv_error_pct,
        't_ac_error_pct': t_ac_error_pct,
        'parameters': {'rho': rho, 'd': d, 'sigma_b': sigma_b, 'sigma_u': sigma_u}
    }

###############################################################
# Main execution
###############################################################

# Initialize Julia environment once
print("Initializing Julia environment...")
jl.seval('using Pkg; Pkg.activate("/home/ianyang/stochastic_simulations/julia"); Pkg.instantiate()')
jl.seval('using DataFrames, NPZ, Base.Threads')
jl.include("/home/ianyang/stochastic_simulations/julia/simulation/TelegraphSSA.jl")
jl.seval('using .TelegraphSSA')

# sanity check: how many threads did we get?
nthreads = int(jl.seval('nthreads()'))
print(f"Julia nthreads = {nthreads}")

# Define test targets
mu_targets = [10, 100, 1000]
t_ac_targets = [1, 5, 10]
cv_targets = [0.5, 1.0, 2.0]

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
                result = run_simulation_test(mu_target, t_ac_target, cv_target)
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
    print(summary_df[['mu_target', 't_ac_target', 'cv_target', 
                     'mu_observed', 'cv_observed', 't_ac_observed',
                     'mean_error_pct', 'cv_error_pct', 't_ac_error_pct']].round(3))

    # Save summary to CSV
    summary_df.to_csv("test_telegraph_fixed_stats_results/summary_results.csv", index=False)
    print(f"\nSummary saved to: test_telegraph_fixed_stats_results/summary_results.csv")

    # Print overall statistics
    print(f"\nOverall Error Statistics:")
    print(f"Mean Error: {summary_df['mean_error_pct'].mean():.2f}% ± {summary_df['mean_error_pct'].std():.2f}%")
    print(f"CV Error: {summary_df['cv_error_pct'].mean():.2f}% ± {summary_df['cv_error_pct'].std():.2f}%")
    print(f"AC Time Error: {summary_df['t_ac_error_pct'].mean():.2f}% ± {summary_df['t_ac_error_pct'].std():.2f}%")

    # Check for any large errors
    large_errors = summary_df[(summary_df['mean_error_pct'] > 10) | 
                             (summary_df['cv_error_pct'] > 15) | 
                             (summary_df['t_ac_error_pct'] > 20)]
    if not large_errors.empty:
        print(f"\n⚠️ Tests with large errors (>10% mean, >15% CV, >20% AC time):")
        print(large_errors[['mu_target', 't_ac_target', 'cv_target', 
                           'mean_error_pct', 'cv_error_pct', 't_ac_error_pct']].round(2))
else:
    print("No successful tests completed.")

print(f"\nAll tests completed. Results saved in 'telegraph_test_results' directory.")
