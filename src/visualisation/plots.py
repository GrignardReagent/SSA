#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from utils.steady_state import find_steady_state
from utils.data_processing import _ensure_numpy, _safe_slice
from stats.autocorrelation import autocrosscorr

################## Mean mRNA counts over time
def plot_mRNA_trajectory(parameter_sets: list, time_points, stress_trajectories, normal_trajectories=None):
    """
    Plot mRNA trajectories for each condition and determine steady-state behavior.
    
    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        time_points (numpy array): Array of time points for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
    """
    # Compute mean trajectories
    mean_stress = stress_trajectories.mean(axis=0)
    
    # Find steady-state time points
    steady_state_time_stress, _ = find_steady_state(parameter_sets[0])
    
    # Plot mRNA trajectories
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean trajectory for stressed condition
    ax.plot(time_points, mean_stress, color='blue', label='Stressed Condition (Mean)', linewidth=2)
    
    # Mark steady-state time for stressed condition
    ax.axvline(steady_state_time_stress, color='blue', linestyle='--', label=f"Steady-State (Stress) @ {steady_state_time_stress:.1f}", alpha=0.5)
    
    # Plot normal condition if provided
    if normal_trajectories is not None:
        mean_normal = normal_trajectories.mean(axis=0)
        steady_state_time_normal, _ = find_steady_state(parameter_sets[1])
        ax.plot(time_points, mean_normal, color='green', label='Normal Condition (Mean)', linewidth=2)
        ax.axvline(steady_state_time_normal, color='green', linestyle='--', label=f"Steady-State (Normal) @ {steady_state_time_normal:.1f}", alpha=0.5)

    # Labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("mRNA Count")
    ax.set_title(f"mRNA Trajectories (Mean for {stress_trajectories.shape[0]} cells)")
    ax.legend()
    ax.grid(True)
    plt.show()
    
    # return the plot object
    return fig, ax

################## Plot variance of mRNA counts over time
def plot_mRNA_variance(parameter_sets: list, time_points, stress_trajectories, normal_trajectories=None):
    """
    Plot the variance of mRNA counts over time for each condition, ensuring variance is calculated after steady state.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        time_points (numpy array): Array of time points for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
    """
    # Find the time index at which steady state is reached
    steady_state_time_stress, steady_state_index_stress = find_steady_state(parameter_sets[0])

    # Extract steady-state portions
    steady_state_traj_stress = _safe_slice(stress_trajectories, steady_state_index_stress)

    # Compute mean and variance at steady state for stressed condition
    stress_mean_ss = steady_state_traj_stress.mean()
    stress_var_ss = steady_state_traj_stress.var()

    # Compute variance over time for plotting
    stress_var_over_time = stress_trajectories.var(axis=0)
    
    # Plot the variance of the mRNA counts over time for each condition
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_points, stress_var_over_time, color='blue', label='Stressed Condition', linewidth=2)
    
    # Mark steady-state time for stressed condition
    ax.axvline(steady_state_time_stress, color='blue', linestyle='--', label=f"Steady-State (Stress) @ {steady_state_time_stress:.1f}")
    
    max_var = stress_var_over_time.max()
    
    # Handle normal condition if provided
    if normal_trajectories is not None:
        steady_state_time_normal, steady_state_index_normal = find_steady_state(parameter_sets[1])
        steady_state_traj_normal = _safe_slice(normal_trajectories, steady_state_index_normal)
        normal_mean_ss = steady_state_traj_normal.mean()
        normal_var_ss = steady_state_traj_normal.var()
        normal_var_over_time = normal_trajectories.var(axis=0)
        
        ax.plot(time_points, normal_var_over_time, color='green', label='Normal Condition', linewidth=2)
        ax.axvline(steady_state_time_normal, color='green', linestyle='--', label=f"Steady-State (Normal) @ {steady_state_time_normal:.1f}")
        max_var = max(stress_var_over_time.max(), normal_var_over_time.max())
    
    ax.set_ylim([0, max_var * 1.1])

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
    if normal_trajectories is not None:
        print(f"  Normal Condition (after {steady_state_time_normal:.1f} min): Mean = {normal_mean_ss:.2f}, Variance = {normal_var_ss:.2f}")

    return fig, ax


################## Plot distribution of mRNA counts after reaching steady state (data from all the timepoints)
def plot_mRNA_dist(parameter_sets: list, stress_trajectories, normal_trajectories=None, bins=None, kde=False):
    """
    Plot the probability density function (PDF) of mRNA counts at steady state.
    
    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
    """
    # Find the time index at which steady state is reached, if data is not already steady state
    _, steady_state_index_stress = find_steady_state(parameter_sets[0])
    steady_state_traj_stress = _safe_slice(stress_trajectories, steady_state_index_stress)
    # Flatten to get all mRNA counts from all trajectories and time points
    stress_ss_flat = steady_state_traj_stress.flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    if kde:
        # Plot KDE (smooth curve)
        sns.kdeplot(stress_ss_flat, fill=True, color='blue', label='Stressed Condition', linewidth=2)
    else:
        if bins is None:
            # Determine maximum mRNA count to set bin range
            max_count = stress_ss_flat.max()
            # Set up bins explicitly for integer values (Poisson data)
            bins = np.arange(0, max_count + 1.5) - 0.5  # shift bins by 0.5 to center integer counts
        # Plot histograms
        ax.hist(stress_ss_flat, bins=bins, density=True, alpha=0.6, color='blue', label='Stressed Condition', edgecolor='black')
    #------------------------ Normal condition if provided ------------------------#
    if normal_trajectories is not None:
        _, steady_state_index_normal = find_steady_state(parameter_sets[1])
        steady_state_traj_normal = _safe_slice(normal_trajectories, steady_state_index_normal)
        normal_ss_flat = steady_state_traj_normal.flatten()
        
        if kde:
            # Plot KDE (smooth curve)
            sns.kdeplot(normal_ss_flat, fill=True, color='red', label='Normal Condition', linewidth=2)
        else:
            if bins is None:
                max_count = max(stress_ss_flat.max(), normal_ss_flat.max())
                bins = np.arange(0, max_count + 1.5) - 0.5
            ax.hist(normal_ss_flat, bins=bins, density=True, alpha=0.6, color='red', label='Normal Condition', edgecolor='black')

    # Labels and title
    ax.set_xlabel("mRNA Count at Steady-State")
    ax.set_ylabel("Probability Density")
    ax.set_title("Distribution of mRNA Counts at Steady-State")
    ax.legend()
    ax.grid(True)
    plt.show()

    return fig, ax

#### Autocorrelation and Cross-correlation of mRNA counts over time
def plot_autocorr(parameter_sets: list, stress_trajectories, normal_trajectories=None):
    """
    Plot the autocorrelation of mRNA counts over time for each condition.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
    """
    # Find the time index at which steady state is reached
    _, steady_state_index_stress = find_steady_state(parameter_sets[0])
    
    # Use safe slicing helper
    steady_state_traj_stress = _safe_slice(stress_trajectories, steady_state_index_stress)

    # Compute autocorrelation for stressed condition 
    # (!! The input for this function must be a numpy array!!)
    stress_autocorr, lags_stress = autocrosscorr(steady_state_traj_stress)

    # Plot the autocorrelation of mRNA counts over time for each condition
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lags_stress, np.nanmean(stress_autocorr, axis=0), color='blue', label='Stressed Condition')

    if normal_trajectories is not None:
        _, steady_state_index_normal = find_steady_state(parameter_sets[1])
        # Use safe slicing helper
        steady_state_traj_normal = _safe_slice(normal_trajectories, steady_state_index_normal)
        
        normal_autocorr, lags_normal = autocrosscorr(steady_state_traj_normal)
        ax.plot(lags_normal, np.nanmean(normal_autocorr, axis=0), color='green', label='Normal Condition')
    
    ax.set_title('Autocorrelation of mRNA Counts at Steady-State')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.legend()
    ax.grid(True)
    plt.show()
    
    return fig, ax

def plot_crosscorr(parameter_sets: list, stress_trajectories, normal_trajectories=None):
    """
    Plot the cross-correlation of mRNA counts over time between stressed and normal conditions.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
    """
    # Cross-correlation requires both datasets
    if normal_trajectories is None:
        print("Warning: Cross-correlation requires both stress and normal trajectories. Skipping plot.")
        return
    
    # Find the time index at which steady state is reached
    _, steady_state_index_stress = find_steady_state(parameter_sets[0])
    _, steady_state_index_normal = find_steady_state(parameter_sets[1])

    # Slice steady-state portions
    steady_state_traj_stress = _safe_slice(stress_trajectories, steady_state_index_stress)
    steady_state_traj_normal = _safe_slice(normal_trajectories, steady_state_index_normal)

    # Compute cross-correlation
    crosscorr, lags_crosscorr = autocrosscorr(steady_state_traj_stress, steady_state_traj_normal)

    # Plot the cross-correlation of mRNA counts over time
    plt.figure()
    plt.plot(lags_crosscorr, np.nanmean(crosscorr, axis=0), color='blue', label='Cross-correlation: Stressed vs. Normal')
    plt.title('Crosscorrelation of mRNA Counts at Steady-State')
    plt.xlabel('Lag')
    plt.ylabel('Crosscorrelation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_autocrosscorr(parameter_sets: list, stress_trajectories, normal_trajectories=None):
    """
    Plot the autocorrelation and cross-correlation of mRNA counts over time for each condition.

    Parameters:
        parameter_sets (list): List of parameter sets (dict) for the simulation.
        stress_trajectories (numpy array): Array of mRNA trajectories for stressed condition.
        normal_trajectories (numpy array, optional): Array of mRNA trajectories for normal condition.
    """
    # Find the time index at which steady state is reached
    _, steady_state_index_stress = find_steady_state(parameter_sets[0])
    
    # Use safe slicing helper
    steady_state_traj_stress = _safe_slice(stress_trajectories, steady_state_index_stress)

    # Compute autocorrelation for stressed condition
    # (!! The input for this function must be a numpy array!!)
    stress_autocorr, lags_stress = autocrosscorr(steady_state_traj_stress)
    
    # Determine the figure layout based on whether normal trajectories are provided
    if normal_trajectories is not None:
        # Both autocorrelation and cross-correlation plots
        _, steady_state_index_normal = find_steady_state(parameter_sets[1])
        steady_state_traj_normal = _safe_slice(normal_trajectories, steady_state_index_normal)
        normal_autocorr, lags_normal = autocrosscorr(steady_state_traj_normal)
        stress_crosscorr, lags_crosscorr = autocrosscorr(steady_state_traj_stress, steady_state_traj_normal)
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Autocorrelation plot
        ax[0].plot(lags_stress, np.nanmean(stress_autocorr, axis=0), color='blue', label='Stressed Condition')
        ax[0].plot(lags_normal, np.nanmean(normal_autocorr, axis=0), color='green', label='Normal Condition')
        ax[0].set_title('Autocorrelation of mRNA Counts at Steady-State')
        ax[0].set_xlabel('Lag')
        ax[0].set_ylabel('Autocorrelation')
        ax[0].legend()
        ax[0].grid(True)
        
        # Cross-correlation plot
        ax[1].plot(lags_crosscorr, np.nanmean(stress_crosscorr, axis=0), color='purple', label='Cross-correlation: Stressed vs. Normal')
        ax[1].set_title('Cross-correlation of mRNA Counts at Steady-State')
        ax[1].set_xlabel('Lag')
        ax[1].set_ylabel('Crosscorrelation')
        ax[1].legend()
        ax[1].grid(True)
    else:
        # Only autocorrelation plot for stressed condition
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lags_stress, np.nanmean(stress_autocorr, axis=0), color='blue', label='Stressed Condition')
        ax.set_title('Autocorrelation of mRNA Counts at Steady-State')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

########## PCA for Visualization ##########
# TODO: consider getting rid of this - not sure how this is helpful? 
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
