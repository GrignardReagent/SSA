#!/usr/bin/python

from utils.steady_state import find_steady_state
from stats.mean import calculate_mean
from stats.variance import calculate_variance

################## Reports and Statistical Analysis
def statistical_report(parameter_sets: list, stress_trajectories=None, normal_trajectories=None, use_steady_state=True):
    """
    Generate a statistical report for the simulated systems, including mean and variance at steady state.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array, optional): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
        use_steady_state (bool, optional): Whether to use only steady state portion for calculations. Defaults to True.
        
    Outputs:
        - Prints the mean and variance of mRNA counts at steady state for available conditions.
        - Returns a dictionary with the computed statistics.
    """
    
    results = {}
    
    # Handle stress trajectories if provided
    if stress_trajectories is not None:
        mean_mRNA_stress = calculate_mean(stress_trajectories, parameter_sets, use_steady_state)
        var_mRNA_stress = calculate_variance(stress_trajectories, parameter_sets, use_steady_state)
        
        results.update({
            "Stressed Mean": mean_mRNA_stress,
            "Stressed Variance": var_mRNA_stress,
        })
        
        if use_steady_state:
            steady_state_time_stress, _ = find_steady_state(parameter_sets[0])
            results["Stressed Steady State Time"] = steady_state_time_stress
    
    # Handle normal trajectories if provided
    if normal_trajectories is not None:
        param_index = 1 if len(parameter_sets) > 1 else 0
        param_set_for_normal = [parameter_sets[param_index]]
        
        mean_mRNA_normal = calculate_mean(normal_trajectories, param_set_for_normal, use_steady_state)
        var_mRNA_normal = calculate_variance(normal_trajectories, param_set_for_normal, use_steady_state)
        
        results.update({
            "Normal Mean": mean_mRNA_normal,
            "Normal Variance": var_mRNA_normal,
        })
        
        if use_steady_state:
            steady_state_time_normal, _ = find_steady_state(parameter_sets[param_index])
            results["Normal Steady State Time"] = steady_state_time_normal
    
    # Print Report
    print("\n=== Statistical Report ===")
    if use_steady_state:
        print("\n📊 **Steady-State Statistics:**")
    else:
        print("\n📊 **Overall Trajectory Statistics:**")
    
    if "Stressed Mean" in results:
        if use_steady_state and "Stressed Steady State Time" in results:
            print(f"  Stressed Condition (after {results['Stressed Steady State Time']:.1f} min):")
        else:
            print(f"  Stressed Condition:")
        print(f"    - Mean mRNA Count: {results['Stressed Mean']:.2f}")
        print(f"    - Variance: {results['Stressed Variance']:.2f}")
    
    if "Normal Mean" in results:
        if use_steady_state and "Normal Steady State Time" in results:
            print(f"\n  Normal Condition (after {results['Normal Steady State Time']:.1f} min):")
        else:
            print(f"\n  Normal Condition:")
        print(f"    - Mean mRNA Count: {results['Normal Mean']:.2f}")
        print(f"    - Variance: {results['Normal Variance']:.2f}")

    return results