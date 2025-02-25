import multiprocessing
import tqdm
import numpy as np
import pandas as pd
import scipy.stats as st
from ssa_analysis import find_steady_state

# Define the update matrix for the reactions
# Columns: G, G*, M
# Columns: G, G*, M
update_matrix = np.array([
    [-1, 1, 0],   # G -> G* (Gene activation)
    [1, -1, 0],   # G* -> G (Gene deactivation)
    [0, 0, 1],    # G -> G + M (mRNA production)
    [0, 0, -1],   # M -> 0 (mRNA degradation)
], dtype=int)

def telegraph_model_propensity(propensities, population, t, sigma_u, sigma_b, rho, d):
    """
    Updates the propensities for the telegraph model reactions based on defined rates.

    Parameters:
        propensities: Array of propensities
        population: Array of species populations [G, G*, M]
        t: Time (not used but required by signature)
        sigma_u: Rate of G -> G* (deactivation)
        sigma_b: Rate of G* -> G (activation)
        rho: Rate of G -> G + M (mRNA production)
        d: Rate of M -> 0 (mRNA degradation)
    """
    # Unpack population
    G, G_star, M = population
    
    # Update propensities for each reaction
    propensities[0] = sigma_u * G          # G -> G*
    propensities[1] = sigma_b * G_star     # G* -> G
    propensities[2] = rho * G              # G -> G + M
    propensities[3] = d * M                # M -> 0

def sample_discrete_scipy(probs):
    """Randomly sample an index with probability given by probs."""
    return st.rv_discrete(values=(range(len(probs)), probs)).rvs()

def sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()
    
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1

def gillespie_draw(propensity_func, propensities, population, t, args=()):
    """
    Draws a reaction and the time it took to do that reaction.
    
    Parameters
    ----------
    propensity_func : function
        Function with call signature propensity_func(population, t, *args)
        used for computing propensities. This function must return
        an array of propensities.
    population : ndarray
        Current population of particles
    t : float
        Value of the current time.
    args : tuple, default ()
        Arguments to be passed to `propensity_func`.
        
    Returns
    -------
    rxn : int
        Index of reaction that occured.
    time : float
        Time it took for the reaction to occur.
    """
    # Compute propensities
    propensity_func(propensities, population, t, *args)
    
    # Sum of propensities
    props_sum = propensities.sum()
    
    # Compute next time
    time = np.random.exponential(1.0 / props_sum)
    
    # Compute discrete probabilities of each reaction
    rxn_probs = propensities / props_sum
    
    # Draw reaction from this distribution
    rxn = sample_discrete(rxn_probs)
    
    return rxn, time

def gillespie_ssa(propensity_func, update, population_0, time_points, args=()):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.
    
    Parameters
    ----------
    propensity_func : function
        Function of the form f(params, t, population) that takes the current
        population of particle counts and return an array of propensities
        for each reaction.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
    args : tuple, default ()
        The set of parameters to be passed to propensity_func.        

    Returns
    -------
    sample : ndarray, shape (num_time_points, num_chemical_species)
        Entry i, j is the count of chemical species j at time
        time_points[i].
    """

    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=int)

    # Initialize and perform simulation
    i_time = 1
    i = 0
    t = time_points[0]
    population = population_0.copy()
    pop_out[0,:] = population
    propensities = np.zeros(update.shape[0])
    while i < len(time_points):
        while t < time_points[i_time]:
            # draw the event and time step
            event, dt = gillespie_draw(propensity_func, propensities, population, t, args)
                
            # Update the population
            population_previous = population.copy()
            population += update[event,:]
                
            # Increment time
            t += dt

        # Update the index
        i = np.searchsorted(time_points > t, True)
        
        # Update the population
        pop_out[i_time:min(i,len(time_points))] = population_previous
        
        # Increment index
        i_time = i
                           
    return pop_out

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

    for i in range(size):
        samples[i, :] = gillespie_ssa(
            telegraph_model_propensity, update_matrix, population_0, time_points,
            args=(sigma_u, sigma_b, rho, d))[:, 2]

    # Save each trajectory as a row with label
    return [[label] + list(trajectory) for trajectory in samples]

#### ORIGINAL FUNCTION ####
# def simulate_two_telegraph_model_systems(parameter_sets, time_points, size, force_steady_state=True, max_extension_factor=5, num_cores=None):
#     """
#     Simulates two systems using stochastic gene expression model.
    
#     Parameters:
#         parameter_sets (list): List of parameter dictionaries for each system.
#         time_points (numpy array): Array of time points for the simulation.
#         size (int): Number of simulations per condition.
#         force_steady_state (bool): If True, extends simulation time until steady-state is reached.
#         max_extension_factor (int): Maximum number of extensions to prevent infinite looping.
#         num_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
    
#     Returns:
#         pd.DataFrame: DataFrame containing simulation results.
#     """
#     if num_cores is None:
#         num_cores = multiprocessing.cpu_count()
#     df_results = None  # Store simulation results across iterations
#     # initial time points
#     initial_time_points = time_points.copy()

#     for system_index, param_set in list(tqdm.tqdm(enumerate(parameter_sets), total=len(parameter_sets), desc="Simulating Telegraph Model Systems")):
#         extended = False  # Track if simulation was extended for this system

#         for extension_round in range(max_extension_factor):
#             with multiprocessing.Pool(num_cores) as pool:
#                 print(f"Running simulations on {num_cores} cores... (Attempt {extension_round + 1})"
#                       f"\nSystem {system_index + 1} parameters: {param_set}")
                
#                 results = pool.map(run_simulation, [(param_set, time_points, size)])

#             # Flatten results
#             results = [item for sublist in results for item in sublist]

#             # Convert to DataFrame
#             columns = ["label"] + [f"time_{t}" for t in time_points]
#             df_new_results = pd.DataFrame(results, columns=columns)

#             # Merge with existing results if extending simulation
#             if df_results is not None:
#                 df_new_results = df_new_results[df_results.columns]  # Align columns before merging
#                 df_results = pd.concat([df_results, df_new_results], ignore_index=True)
#             else:
#                 df_results = df_new_results

#             # Check if steady-state is reached
#             trajectories = df_results[df_results["label"] == system_index].iloc[:, 1:].values
#             mean_trajectory = trajectories.mean(axis=0)

#             steady_state_time, steady_state_index = find_steady_state(time_points, mean_trajectory)

#             # If steady-state is not reached
#             if steady_state_index == (len(time_points) - 1):
#                 if force_steady_state:
#                     print(f"⚠️ Warning: Steady-state not reached for system {system_index + 1}.")
#                     time_points = np.arange(0, time_points[-1] * 1.5, time_points[1] - time_points[0])  # Extend time by 50%
#                     print(f"Extending simulation time range to {time_points[-1]} minutes for system {system_index + 1}...")
#                     extended = True
#                 else:
#                     print(f"⚠️ Warning: Steady-state not reached for system {system_index + 1}, but simulation will not be extended as per user choice.")
#                     break
#             else:
#                 print(f"✅ Steady-state reached for system {system_index + 1} at {steady_state_time} minutes.")
#                 break

#         if extended:
#             print(f"✅ Final simulation time range for system {system_index + 1}: {time_points[-1]} minutes")
#             # reset timepoint to original timepoints after extending
#             time_points = initial_time_points.copy()

#     # check for NaN values before returning        
#     if df_results.isna().sum().sum() > 0:
#         print("⚠️ Warning: NaN values detected in df_results!")
#         print(df_results.isna().sum()[df_results.isna().sum() > 0])

#     return df_results

#####DEEPSEEK#####
# def simulate_two_telegraph_model_systems(parameter_sets, time_points, size, force_steady_state=True, max_extension_factor=5, num_cores=None):
#     """
#     Simulates two systems using stochastic gene expression model.
    
#     Parameters:
#         parameter_sets (list): List of parameter dictionaries for each system.
#         time_points (numpy array): Array of time points for the simulation.
#         size (int): Number of simulations per condition.
#         force_steady_state (bool): If True, extends simulation time until steady-state is reached.
#         max_extension_factor (int): Maximum number of extensions to prevent infinite looping.
#         num_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
    
#     Returns:
#         pd.DataFrame: DataFrame containing simulation results.
#     """
#     if num_cores is None:
#         num_cores = multiprocessing.cpu_count()
#     df_results = None  # Store simulation results across iterations
#     # initial time points
#     initial_time_points = time_points.copy()

#     for system_index, param_set in list(tqdm.tqdm(enumerate(parameter_sets), total=len(parameter_sets), desc="Simulating Telegraph Model Systems")):
#         extended = False  # Track if simulation was extended for this system

#         for extension_round in range(max_extension_factor):
#             with multiprocessing.Pool(num_cores) as pool:
#                 print(f"Running simulations on {num_cores} cores... (Attempt {extension_round + 1})"
#                       f"\nSystem {system_index + 1} parameters: {param_set}")
                
#                 results = pool.map(run_simulation, [(param_set, time_points, size)])

#             # Flatten results
#             results = [item for sublist in results for item in sublist]

#             # Convert to DataFrame
#             columns = ["label"] + [f"time_{t}" for t in time_points]
#             df_new_results = pd.DataFrame(results, columns=columns)

#             # Merge with existing results if extending simulation
#             if df_results is not None:
#                 # Ensure columns match by reindexing
#                 df_new_results = df_new_results.reindex(columns=df_results.columns, fill_value=np.nan)
#                 df_results = pd.concat([df_results, df_new_results], ignore_index=True)
#             else:
#                 df_results = df_new_results

#             # Check if steady-state is reached
#             trajectories = df_results[df_results["label"] == system_index].iloc[:, 1:].values
#             mean_trajectory = trajectories.mean(axis=0)

#             steady_state_time, steady_state_index = find_steady_state(time_points, mean_trajectory)

#             # If steady-state is not reached
#             if steady_state_index == (len(time_points) - 1):
#                 if force_steady_state:
#                     print(f"⚠️ Warning: Steady-state not reached for system {system_index + 1}.")
#                     new_end_time = time_points[-1] * 1.5  # Extend time by 50%
#                     new_step = time_points[1] - time_points[0]
#                     time_points = np.arange(0, new_end_time, new_step)
#                     print(f"Extending simulation time range to {time_points[-1]} minutes for system {system_index + 1}...")
#                     extended = True
#                 else:
#                     print(f"⚠️ Warning: Steady-state not reached for system {system_index + 1}, but simulation will not be extended as per user choice.")
#                     break
#             else:
#                 print(f"✅ Steady-state reached for system {system_index + 1} at {steady_state_time} minutes.")
#                 break

#         if extended:
#             print(f"✅ Final simulation time range for system {system_index + 1}: {time_points[-1]} minutes")
#             # Do NOT reset time_points here to allow subsequent systems to use the extended time points

#     # check for NaN values before returning        
#     if df_results.isna().sum().sum() > 0:
#         print("⚠️ Warning: NaN values detected in df_results!")
#         print(df_results.isna().sum()[df_results.isna().sum() > 0])

#     return df_results

#####GPT 03#####
def run_simulation_continuous(args): # Worker function  
    """
    Runs a simulation segment starting from given initial states.
    
    Parameters:
        args: tuple of (param_set, time_points, size, initial_populations)
              - param_set: dictionary of parameters (including a 'label' key)
              - time_points: time grid for this segment (starting at 0)
              - size: number of simulation replicates
              - initial_populations: numpy array of shape (size, 3) giving the starting state for each replicate;
                                     if None, default initial state [1,0,0] is used.
    
    Returns:
        results: list of lists; each row is [label, value_at_t0, value_at_t1, ...] (here we record mRNA counts)
        final_states: numpy array of shape (size, 3) with the final state (G, G*, M) for each replicate.
    """
    param_set, time_points, size, initial_populations = args
    sigma_u, sigma_b, rho, d, label = param_set.values()
    dt = time_points[1] - time_points[0]
    if initial_populations is None:
        # Create default initial state for each replicate
        initial_populations = np.tile(np.array([1, 0, 0], dtype=int), (size, 1))
    segment_results = []
    final_states = np.empty((size, 3), dtype=int)
    for i in range(size):
        traj = gillespie_ssa(
            telegraph_model_propensity, update_matrix,
            initial_populations[i], time_points,
            args=(sigma_u, sigma_b, rho, d))
        # We record the mRNA counts (column index 2)
        segment_results.append(traj[:,2])
        final_states[i,:] = traj[-1, :]
    # Prepend label to each trajectory for later DataFrame construction
    results = [[label] + list(traj) for traj in segment_results]
    return results, final_states


def simulate_two_telegraph_model_systems(parameter_sets, time_points, size,
                                         force_steady_state=True,
                                         max_extension_factor=5, num_cores=None):
    """
    Simulates two systems continuously using a stochastic gene expression model.
    
    For each system, the simulation is run in segments.
    The first segment is simulated using the supplied time_points.
    If the steady-state (based on the mean mRNA trajectory) has not been reached
    by the end of the segment, a new segment is simulated starting from the final state
    of the previous segment and appended (with time shifted) to the overall trajectory.
    
    Parameters:
        parameter_sets (list): List of parameter dictionaries for each system.
        time_points (numpy array): Array of time points for the first segment (assumed uniformly spaced).
        size (int): Number of simulation replicates per condition.
        force_steady_state (bool): If True, simulation is extended until steady state is reached or max_extension_factor is hit.
        max_extension_factor (int): Maximum number of segments to simulate.
        num_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
    
    Returns:
        pd.DataFrame: DataFrame with one row per simulation replicate. The first column is 'label',
                      and subsequent columns record the mRNA count at the overall (continuous) time points.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    dt = time_points[1] - time_points[0]
    
    df_results_list = []
    # Loop over systems
    for system_index, param_set in enumerate(tqdm.tqdm(parameter_sets,
                                                       desc="Simulating Telegraph Model Systems")):
        # overall_results: list of rows (one per replicate) for the current system
        overall_results = None  
        # overall_time: the continuous time grid (will be built segment by segment)
        overall_time = None  
        current_offset = 0  # time offset for the current segment
        current_segment = time_points.copy()  # time grid for the current segment (starts at 0)
        initial_populations = None  # for the first segment, use default initial state
        for extension_round in range(max_extension_factor):
            # Run this simulation segment using a multiprocessing pool
            with multiprocessing.Pool(num_cores) as pool:
                seg_data = pool.map(run_simulation_continuous,
                                     [(param_set, current_segment, size, initial_populations)])
            seg_results, seg_final_states = seg_data[0]
            # Shift the current segment time by the current_offset
            seg_time = current_segment + current_offset
            if overall_results is None:
                overall_results = seg_results
                overall_time = seg_time
            else:
                # For later segments, drop the first time point (duplicate) before concatenation
                seg_results = [ [row[0]] + row[2:] for row in seg_results ]
                overall_results = [ prev + seg[1:] for prev, seg in zip(overall_results, seg_results)]
                overall_time = np.concatenate([overall_time, seg_time[1:]])
            # Compute the mean mRNA trajectory (ignoring the label column)
            # traj_array = np.array([row[1:] for row in overall_results])
            # mean_traj = traj_array.mean(axis=0)
            # Use the overall time grid to check for steady state
            steady_state_time, steady_state_index = find_steady_state(param_set)
            if steady_state_index < len(overall_time) - 1:
                print(f"✅ Steady-state reached for system {system_index + 1} at {steady_state_time} minutes.")
                break
            else:
                if force_steady_state:
                    print(f"⚠️ Warning: Steady-state not reached for system {system_index + 1} "
                          f"(current final time = {overall_time[-1]} minutes). Extending simulation...")
                    # Update offset to be the final time of the current overall trajectory
                    current_offset = overall_time[-1]
                    # Extend the segment duration by 50%
                    new_duration = (current_segment[-1] - current_segment[0]) * 1.5
                    num_points = int(new_duration / dt) + 1
                    current_segment = np.linspace(0, new_duration, num_points)
                    # Use the final states from this segment as the new starting states
                    initial_populations = seg_final_states
                else:
                    print(f"⚠️ Warning: Steady-state not reached for system {system_index + 1} "
                          f"and force_steady_state is False. Stopping extension.")
                    break
        
        # Construct a DataFrame for this system using the overall time grid
        columns = ["label"] + [f"time_{t}" for t in overall_time]
        df_system = pd.DataFrame(overall_results, columns=columns)
        df_results_list.append(df_system)
    
    df_results = pd.concat(df_results_list, ignore_index=True)
    
    # Optional: warn if any NaN values are present
    if df_results.isna().sum().sum() > 0:
        print("⚠️ Warning: NaN values detected in simulation results!")
        print(df_results.isna().sum()[df_results.isna().sum() > 0])
    
    return df_results
