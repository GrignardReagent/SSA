import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################## Reports and Statistical Analysis
# def find_steady_state(time_points, mean_trajectory, threshold=0.05):
#     """
#     (RETIRED)
#     Determine the time point when the system reaches steady state.
    
#     Parameters:
#         time_points (numpy array): Array of time points.
#         mean_trajectory (numpy array): Mean mRNA counts over time.
#         threshold (float): Relative change threshold for steady state detection, increase the detection threshold if the defined system struggles to reach steady state.
        
#     Returns:
#         steady_state_time (float): Time when steady state is reached.
#         steady_state_index (int): Index corresponding to the steady state time.
#     """
#     window_size = int(len(time_points) * 0.05)  # Look at changes over a window size relative to the total time points defined
#     for i in range(len(mean_trajectory) - window_size):
#         recent_change = np.abs(np.diff(mean_trajectory[i:i + window_size])).mean()
#         if recent_change < threshold * mean_trajectory[i]:  
#             return time_points[i], i
#     return time_points[-1], len(time_points) - 1  # Default to last time point if no steady state detected

def find_steady_state(parameter_set: dict):
    """
    Determine the time point when the system reaches steady state, this is usually 10/degradation rate, so this function take the parameter_set as the input and return the steady state time.
    
    Parameters:
        parameter_set (dict): Parameter set for the simulation.
        
    Returns:
        steady_state_time (float): Time when steady state is reached.
    """
    # Extract the degradation rate from the parameter set
    degradation_rate = parameter_set['d']
    steady_state_time = 10 / degradation_rate  # Time to reach steady state is usually 10 / degradation rate

    return steady_state_time, int(steady_state_time) # time, index

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
    # print(f"Steady-State Reached: {'Yes' if ss_reached_stress else 'No'} (Stress) | {'Yes' if ss_reached_normal else 'No'} (Normal)")

    # if not ss_reached_stress:
    #     print("⚠️ Warning: Steady-state not clearly reached in stressed condition. Consider extending simulation time.")

    # if not ss_reached_normal:
    #     print("⚠️ Warning: Steady-state not clearly reached in normal condition. Consider extending simulation time.")

    print("\n📊 **Steady-State Statistics:**")
    print(f"  Stressed Condition (after {steady_state_time_stress:.1f} min):")
    print(f"    - Mean mRNA Count: {mean_mRNA_stress_ss:.2f}")
    print(f"    - Variance: {var_mRNA_stress_ss:.2f}")

    print(f"\n  Normal Condition (after {steady_state_time_normal:.1f} min):")
    print(f"    - Mean mRNA Count: {mean_mRNA_normal_ss:.2f}")
    print(f"    - Variance: {var_mRNA_normal_ss:.2f}")

    return {
        # "Steady State Reached": {"Stress": ss_reached_stress, "Normal": ss_reached_normal},
        "Stress Stats": {"Mean": mean_mRNA_stress_ss, "Variance": var_mRNA_stress_ss, "Steady State Time": steady_state_time_stress},
        "Normal Stats": {"Mean": mean_mRNA_normal_ss, "Variance": var_mRNA_normal_ss, "Steady State Time": steady_state_time_normal}
    }

################## Mean mRNA counts over time
def plot_mRNA_trajectory(parameter_sets: list, time_points, stress_trajectories, normal_trajectories):
    """
    Plot mRNA trajectories for each condition and determine steady-state behavior.
    
    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        time_points (numpy array): Array of time points for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array): Array of mRNA trajectories for normal condition.
    """
    # Compute mean trajectories
    mean_stress = stress_trajectories.mean(axis=0)
    mean_normal = normal_trajectories.mean(axis=0)

    # Find steady-state time points
    steady_state_time_stress, _ = find_steady_state(parameter_sets[0])
    steady_state_time_normal, _ = find_steady_state(parameter_sets[1])

    # Plot mRNA trajectories
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean trajectory for each condition
    ax.plot(time_points, mean_stress, color='blue', label='Stressed Condition (Mean)', linewidth=2)
    ax.plot(time_points, mean_normal, color='green', label='Normal Condition (Mean)', linewidth=2)

    # Mark steady-state time
    ax.axvline(steady_state_time_stress, color='blue', linestyle='--', label=f"Steady-State (Stress) @ {steady_state_time_stress:.1f}", alpha=0.5)
    ax.axvline(steady_state_time_normal, color='green', linestyle='--', label=f"Steady-State (Normal) @ {steady_state_time_normal:.1f}", alpha=0.5)

    # Labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("mRNA Count")
    ax.set_title(f"Yeast mRNA Trajectories Under Different Conditions (Mean for {stress_trajectories.shape[0]} cells)")
    ax.legend()
    ax.grid(True)

    # Show plot
    plt.show()

################## Variance of mRNA counts over time
def plot_mRNA_variance(parameter_sets: list, time_points, stress_trajectories, normal_trajectories):
    """
    Plot the variance of mRNA counts over time for each condition, ensuring variance is calculated after steady state.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        time_points (numpy array): Array of time points for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array): Array of mRNA trajectories for normal condition.
    """
    # Find the time index at which steady state is reached
    steady_state_time_stress, steady_state_index_stress = find_steady_state(parameter_sets[0])
    steady_state_time_normal, steady_state_index_normal = find_steady_state(parameter_sets[1])

    # Extract steady-state portions
    steady_state_traj_stress = stress_trajectories[:, steady_state_index_stress:]
    steady_state_traj_normal = normal_trajectories[:, steady_state_index_normal:]

    # Compute mean and variance at steady state (same as statistical_report)
    stress_mean_ss = steady_state_traj_stress.mean()
    normal_mean_ss = steady_state_traj_normal.mean()
    stress_var_ss = steady_state_traj_stress.var()
    normal_var_ss = steady_state_traj_normal.var()

    # Compute variance over time for plotting
    stress_var_over_time = stress_trajectories.var(axis=0)
    normal_var_over_time = normal_trajectories.var(axis=0)

    # Plot the variance of the mRNA counts over time for each condition
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_points, stress_var_over_time, color='blue', label='Stressed Condition', linewidth=2)
    ax.plot(time_points, normal_var_over_time, color='green', label='Normal Condition', linewidth=2)
    ax.set_ylim([0, max(stress_var_over_time.max(), normal_var_over_time.max()) * 1.1])

    # Mark steady-state time
    ax.axvline(steady_state_time_stress, color='blue', linestyle='--', label=f"Steady-State (Stress) @ {steady_state_time_stress:.1f}")
    ax.axvline(steady_state_time_normal, color='green', linestyle='--', label=f"Steady-State (Normal) @ {steady_state_time_normal:.1f}")

    # Labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Variance of mRNA Count')
    ax.set_title('Variance of mRNA Counts Over Time')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Print variance at steady state (consistent with statistical_report)
    print("\n=== Variance at Steady State ===")
    print(f"  Stressed Condition (after {steady_state_time_stress:.1f} min): Mean = {stress_mean_ss:.2f}, Variance = {stress_var_ss:.2f}")
    print(f"  Normal Condition (after {steady_state_time_normal:.1f} min): Mean = {normal_mean_ss:.2f}, Variance = {normal_var_ss:.2f}")

    # return {
    #     "Stress Variance at Steady State": stress_var_ss,
    #     "Normal Variance at Steady State": normal_var_ss,
    #     "Stress Mean at Steady State": stress_mean_ss,
    #     "Normal Mean at Steady State": normal_mean_ss,
    #     "Steady State Time": {"Stress": steady_state_time_stress, "Normal": steady_state_time_normal}
    # }

################## Distribution of mRNA counts after reaching steady state (data from all the timepoints)
def plot_mRNA_dist(parameter_sets: list, stress_trajectories, normal_trajectories):
    """
    Plot the probability density function (PDF) of mRNA counts at steady state.
    
    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array): Array of mRNA trajectories for normal condition.
    """

    # Find the time index at which steady state is reached
    _, steady_state_index_stress = find_steady_state(parameter_sets[0])
    _, steady_state_index_normal = find_steady_state(parameter_sets[1])

    # Extract mRNA counts after steady state is reached
    stress_ss = stress_trajectories[:, steady_state_index_stress:].flatten()
    normal_ss = normal_trajectories[:, steady_state_index_normal:].flatten()

    # Plot KDE (smooth curve)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(stress_ss, fill=True, color='blue', label='Stressed Condition', linewidth=2)
    sns.kdeplot(normal_ss, fill=True, color='green', label='Normal Condition', linewidth=2)

    # Labels and title
    ax.set_xlabel("mRNA Count at Steady-State")
    ax.set_ylabel("Probability Density")
    ax.set_title("Distribution of mRNA Counts at Steady-State")
    ax.legend()
    ax.grid(True)

    # Show plot
    plt.show()