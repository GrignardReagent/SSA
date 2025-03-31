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