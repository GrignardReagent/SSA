import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

################## Reports and Statistical Analysis
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

#TODO: Move plots into a separate moedule in the visualisation folder
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

############# Autocorrelation and Cross-correlation ##############
def autocrosscorr(
    yA,
    yB=None,
    stationary=False,
    normalised=True,
    only_pos=False,
):
    """
    Calculate normalised auto- or cross-correlations as a function of lag.

    Lag is given in multiples of the unknown time interval between data
    points, and normalisation is by the product of the standard
    deviation over time for each replicate for each variable.

    For the cross-correlation between sA and sB, the closest peak to zero
    lag should be in the positive lags if sA is delayed compared to
    signal B and in the negative lags if sA is advanced compared to
    signal B.

    Parameters
    ----------
    yA: array
        An array of signal values, with each row a replicate measurement
        and each column a time point.
    yB: array (required for cross-correlation only)
        An array of signal values, with each row a replicate measurement
        and each column a time point.
    stationary: boolean
        If True, the underlying dynamic process is assumed to be
        stationary with the mean a constant, estimated from all
        data points.
    normalised: boolean (optional)
        If True, normalise the result for each replicate by the standard
        deviation over time for that replicate.
    only_pos: boolean (optional)
        If True, return results only for positive lags.

    Returns
    -------
    corr: array
        An array of the correlations with each row the result for the
        corresponding replicate and each column a time point
    lags: array
        A 1D array of the lags in multiples of the unknown time interval

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    Define a sine signal with 200 time points and 333 replicates

    >>> t = np.linspace(0, 4, 200)
    >>> ts = np.tile(t, 333).reshape((333, 200))
    >>> s = 3*np.sin(2*np.pi*ts + 2*np.pi*np.random.rand(333, 1))

    Find and plot the autocorrelaton

    >>> ac, lags = autocrosscorr(s)
    >>> plt.figure()
    >>> plt.plot(lags, np.nanmean(ac, axis=0))
    >>> plt.show()

    Reference
    ---------
    Dunlop MJ, Cox RS, Levine JH, Murray RM, Elowitz MB (2008). Regulatory
    activity revealed by dynamic correlations in gene expression noise.
    Nat Genet, 40, 1493-1498.
    """
    # number of replicates & number of time points
    nr, nt = yA.shape
    # autocorrelation
    if yB is None:
        yB = yA
    # find deviation from the mean
    dyA, stdA = _dev(yA, nr, nt, stationary)
    dyB, stdB = _dev(yB, nr, nt, stationary)
    # calculate correlation
    # lag r runs over positive lags
    pos_corr = np.nan * np.ones(yA.shape)
    for r in range(nt):
        prods = [dyA[:, t] * dyB[:, t + r] for t in range(nt - r)]
        pos_corr[:, r] = np.nanmean(prods, axis=0)
    # lag r runs over negative lags
    # use corr_AB(-k) = corr_BA(k)
    neg_corr = np.nan * np.ones(yA.shape)
    for r in range(nt):
        prods = [dyB[:, t] * dyA[:, t + r] for t in range(nt - r)]
        neg_corr[:, r] = np.nanmean(prods, axis=0)
    if normalised:
        # normalise by standard deviation
        pos_corr = pos_corr / stdA / stdB
        neg_corr = neg_corr / stdA / stdB
    # combine lags
    lags = np.arange(-nt + 1, nt)
    corr = np.hstack((np.flip(neg_corr[:, 1:], axis=1), pos_corr))
    # return correlation and lags
    if only_pos:
        return corr[:, int(lags.size / 2) :], lags[int(lags.size / 2) :]
    else:
        return corr, lags

def _dev(y, nr, nt, stationary=False):
    # calculate deviation from the mean
    if stationary:
        # mean calculated over time and over replicates
        dy = y - np.nanmean(y)
    else:
        # mean calculated over replicates at each time point
        dy = y - np.nanmean(y, axis=0).reshape((1, nt))
    # standard deviation calculated for each replicate
    stdy = np.sqrt(np.nanmean(dy**2, axis=1).reshape((nr, 1)))
    return dy, stdy

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
