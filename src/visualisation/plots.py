#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from utils.steady_state import find_steady_state
from stats.autocorrelation import autocrosscorr

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

################## Plot variance of mRNA counts over time
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

################## Plot distribution of mRNA counts after reaching steady state (data from all the timepoints)
def plot_mRNA_dist(parameter_sets: list, stress_trajectories, normal_trajectories, steady_state=False):
    """
    Plot the probability density function (PDF) of mRNA counts at steady state.
    
    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array): Array of mRNA trajectories for normal condition.
        steady_state (bool): If True, the data is already at steady state, otherwise the time index for steady state will be calculated from the parameter sets.
    """
    # Find the time index at which steady state is reached, if data is not already steady state
    if not steady_state:
        _, steady_state_index_stress = find_steady_state(parameter_sets[0])
        _, steady_state_index_normal = find_steady_state(parameter_sets[1])

        # Extract mRNA counts after steady state is reached
        stress_ss = stress_trajectories[:, steady_state_index_stress:].flatten()
        normal_ss = normal_trajectories[:, steady_state_index_normal:].flatten()
    else:
        # Data is already at steady state
        stress_ss = stress_trajectories.flatten()
        normal_ss = normal_trajectories.flatten()

    # Plot KDE (smooth curve)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.kdeplot(stress_ss, fill=True, color='blue', label='Stressed Condition', linewidth=2)
    # sns.kdeplot(normal_ss, fill=True, color='green', label='Normal Condition', linewidth=2)
        # Labels and title
    # ax.set_xlabel("mRNA Count at Steady-State")
    # ax.set_ylabel("Probability Density")
    # ax.set_title("Distribution of mRNA Counts at Steady-State")
    # ax.legend()
    # ax.grid(True)

    # Plot histogram (recommended for Poisson-distributed data)
    # # Determine maximum mRNA count to set bin range
    # max_count = max(stress_ss.max(), normal_ss.max())

    # # Set up bins explicitly for integer values (Poisson data)
    # bins = np.arange(0, max_count + 1.5) - 0.5  # shift bins by 0.5 to center integer counts

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(stress_ss, density=True, alpha=0.6, color='blue', label='Stressed Condition', edgecolor='black')
    plt.hist(normal_ss, density=True, alpha=0.6, color='green', label='Normal Condition', edgecolor='black')

    # Labels and title
    plt.xlabel("mRNA Count at Steady-State")
    plt.ylabel("Probability Density")
    plt.title("Distribution of mRNA Counts at Steady-State")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_autocrosscorr(parameter_sets: list, stress_trajectories, normal_trajectories):
    """
    Plot the autocorrelation and cross-correlation of mRNA counts over time for each condition.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array): Array of mRNA trajectories for normal condition.
    """
    # Find the time index at which steady state is reached
    _, steady_state_index_stress = find_steady_state(parameter_sets[0])
    _, steady_state_index_normal = find_steady_state(parameter_sets[1])

    # Extract steady-state portions
    steady_state_traj_stress = stress_trajectories[:, steady_state_index_stress:]
    steady_state_traj_normal = normal_trajectories[:, steady_state_index_normal:]

    # Compute autocorrelation and cross-correlation
    stress_autocorr, lags_stress = autocrosscorr(steady_state_traj_stress)
    normal_autocorr, lags_normal = autocrosscorr(steady_state_traj_normal)
    stress_crosscorr, lags_crosscorr = autocrosscorr(steady_state_traj_stress, steady_state_traj_normal)

    # Plot the autocorrelation and cross-correlation of mRNA counts over time for each condition
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(lags_stress, np.nanmean(stress_autocorr, axis=0), color='blue', label='Stressed Condition')
    ax[0].plot(lags_normal, np.nanmean(normal_autocorr, axis=0), color='green', label='Normal Condition')
    ax[0].set_title('Autocorrelation of mRNA Counts at Steady-State')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('Autocorrelation')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(lags_crosscorr, np.nanmean(stress_crosscorr, axis=0), color='purple', label='Cross-correlation: Stressed vs. Normal')
    ax[1].set_title('Cross-correlation of mRNA Counts at Steady-State')
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('Crosscorrelation')
    ax[1].legend()
    ax[1].grid(True)

########## PCA for Visualization ##########
def pca_plot(mRNA_traj_file):
    """
    Load the mRNA trajectories dataset and perform PCA for visualization.
    
    Parameters:
        mRNA_traj_file: Path to the mRNA trajectories dataset
    """
    # Load the mRNA trajectories dataset
    df_results = pd.read_csv(mRNA_traj_file)

    # Extract features (mRNA trajectories) and labels
    X = df_results.iloc[:, 1:].values  # All time series data
    y = df_results["label"].values  # Labels: 0 (Stressed Condition) or 1 (Normal Condition)

    # Scatter plot of two PCA components for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='Stressed Condition', alpha=0.5)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='green', label='Normal Condition', alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of mRNA Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()
