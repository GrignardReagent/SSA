#!/usr/bin/python

from utils.steady_state import find_steady_state

################## Reports and Statistical Analysis
def statistical_report(parameter_sets: list, stress_trajectories, normal_trajectories):
    """
    Generate a statistical report for the simulated systems, including mean and variance at steady state.

    Parameters:
        time_points (numpy array): Array of time points for the simulation.
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array): Array of mRNA trajectories for normal condition.
    """

    # Find steady-state time points
    steady_state_time_stress, steady_state_index_stress = find_steady_state(parameter_sets[0])
    steady_state_time_normal, steady_state_index_normal = find_steady_state(parameter_sets[1])
    
    # Extract steady-state portions
    steady_state_traj_stress = stress_trajectories[:, steady_state_index_stress:]
    steady_state_traj_normal = normal_trajectories[:, steady_state_index_normal:]

    # Compute mean and variance after steady state
    mean_mRNA_stress_ss = steady_state_traj_stress.mean()
    var_mRNA_stress_ss = steady_state_traj_stress.var()
    mean_mRNA_normal_ss = steady_state_traj_normal.mean()
    var_mRNA_normal_ss = steady_state_traj_normal.var()

    # Print Report
    print("\n=== Statistical Report ===")
    print("\n📊 **Steady-State Statistics:**")
    print(f"  Stressed Condition (after {steady_state_time_stress:.1f} min):")
    print(f"    - Mean mRNA Count: {mean_mRNA_stress_ss:.2f}")
    print(f"    - Variance: {var_mRNA_stress_ss:.2f}")

    print(f"\n  Normal Condition (after {steady_state_time_normal:.1f} min):")
    print(f"    - Mean mRNA Count: {mean_mRNA_normal_ss:.2f}")
    print(f"    - Variance: {var_mRNA_normal_ss:.2f}")

    return {
        "Stressed Mean": mean_mRNA_stress_ss, "Stressed Variance": var_mRNA_stress_ss, "Stressed Steady State Time": steady_state_time_stress},{"Normal Mean": mean_mRNA_normal_ss, "Normal Variance": var_mRNA_normal_ss, "Normal Steady State Time": steady_state_time_normal
        }