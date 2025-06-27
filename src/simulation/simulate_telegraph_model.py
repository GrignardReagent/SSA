
import multiprocessing
import tqdm
import numpy as np
import pandas as pd

# gillespie code
from simulation.gillespie_algorithm import gillespie_ssa, telegraph_model_propensity, update_matrix

# Worker function must be at the top level to be picklable
def run_simulation(args):
    """
    Runs a single simulation for a given parameter set.
    """
    param_set, time_points, size = args
    sigma_u, sigma_b, rho, d, label = param_set.values()
    population_0 = np.array([1, 0, 0], dtype=int)

    # Store mRNA trajectories
    samples = np.empty((size, len(time_points)), dtype=int)

    # for the size of the simulation traj defined, run the simulation with the specified parameters.
    for i in range(size):
        samples[i, :] = gillespie_ssa(
            telegraph_model_propensity, update_matrix, population_0, time_points,
            args=(sigma_u, sigma_b, rho, d))[:, 2]

    # Save each trajectory as a row with label
    return [[label] + list(trajectory) for trajectory in samples]

def simulate_two_telegraph_model_systems(parameter_sets, time_points, size, num_cores=None):
    """
    Simulates two systems using stochastic gene expression model without forcing steady-state extension.
    
    Parameters:
        parameter_sets (list): List of parameter dictionaries for each system.
        time_points (numpy array): Array of time points for the simulation.
        size (int): Number of simulations per condition.
        num_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
    
    Returns:
        pd.DataFrame: DataFrame containing simulation results.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    df_results = None  # Store simulation results

    for system_index, param_set in tqdm.tqdm(enumerate(parameter_sets), total=len(parameter_sets), desc="Simulating Telegraph Model Systems"):
        with multiprocessing.Pool(num_cores) as pool:
            print(f"Running simulations on {num_cores} cores...\nSystem {system_index + 1} parameters: {param_set}")
            results = pool.map(run_simulation, [(param_set, time_points, size)])

        # Flatten results
        results = [item for sublist in results for item in sublist]

        # Convert to DataFrame
        columns = ["label"] + [f"time_{t}" for t in time_points]
        df_new_results = pd.DataFrame(results, columns=columns)

        # Merge with existing results
        if df_results is not None:
            df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        else:
            df_results = df_new_results

    # Check for NaN values before returning
    if df_results.isna().sum().sum() > 0:
        print("⚠️ Warning: NaN values detected in df_results!")
        print(df_results.isna().sum()[df_results.isna().sum() > 0])

    return df_results

def simulate_one_telegraph_model_system(parameter_set, time_points, size, num_cores=None):
    """
    Simulates a single system using stochastic gene expression model without forcing steady-state extension.
    
    Parameters:
        parameter_set (dict): Parameter dictionary for the system.
        time_points (numpy array): Array of time points for the simulation.
        size (int): Number of simulations per condition.
        num_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
    
    Returns:
        pd.DataFrame: DataFrame containing simulation results.
    """
    return simulate_two_telegraph_model_systems([parameter_set], time_points, size, num_cores)