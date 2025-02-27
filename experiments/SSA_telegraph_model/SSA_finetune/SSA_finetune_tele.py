import matplotlib.pyplot as plt
import multiprocessing
import tqdm
import numpy as np
import os
import scipy.stats as st
import numba
import biocircuits
import itertools

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

# --------------------------------

# Worker function for running simulations
def run_simulation(params):
    sigma_u, sigma_b, rho, d = params
    time_points = np.linspace(0, 1000, 1001)

    # To ensure steady state, set time_points[-1] >> 10 / d                 
    if time_points[-1] < 10 / d:
        time_points = np.linspace(0, 10 / d, int(10 / d * 10) + 1)

    population_0 = np.array([1, 0, 0], dtype=int)  # Initial [G, G*, M]
    samples = np.empty((size, len(time_points), 3), dtype=int)

    for i in range(size):
        samples[i, :, :] = gillespie_ssa(telegraph_model_propensity, update_matrix, population_0, time_points, args=(sigma_u, sigma_b, rho, d))
    
    # Calculate Fano factor for each time point
    fano_m_time = samples[:, :, 2].var(axis=0) / samples[:, :, 2].mean(axis=0)
    # we only want the Fano factor at steady state, so we take the last value
    fano_m = fano_m_time[-1]
    # Calculate the approximate Fano factor for the telegraph model
    expected_fano_m_approx = 1 +  rho / sigma_u
    # Calculate exact Fano factor for telegraph model
    expected_fano_m_exact = 1 + (rho * sigma_u) / ((sigma_b + sigma_u) * (sigma_b + d + sigma_u))

    # Ensure directory exists
    output_dir = "data/"
    os.makedirs(output_dir, exist_ok=True)

    # Save samples to the directory
    sample_file = os.path.join(output_dir, f"samples_{sigma_u}_{sigma_b}_{rho}_{d}.npy")
    np.save(sample_file, samples)

    return [sigma_u, sigma_b, rho, d, fano_m, expected_fano_m_approx, expected_fano_m_exact]

# --------------------------------

# Define the rates for each reaction (constants)
# To match theoretical and actual Fano factors, ensure:
# 1. Set sigma_u >> sigma_b
# 2. Set rho >> sigma_u
# 3. Reaction sampled at Steady State, and t_ss >> 10 / d

sigma_u = np.logspace(-2, 2, 10)    # Rate of G -> G* (deactivation)
sigma_b = np.logspace(-4, 0, 10)    # Rate of G* -> G (activation)
rho = np.logspace(0, 3, 10)         # Rate of G -> G + M (mRNA production)
d = np.logspace(-2, 0, 5)          # Rate of M -> 0 (mRNA degradation)

size = 10000  # Number of simulations
# Generate parameter grid with filtering conditions to enforce beta >> alpha_off >> alpha_on
filtered_param_grid = [
    (s_u, s_b, r, d_d)
    for s_u, s_b, r, d_d in itertools.product(sigma_u, sigma_b, rho, d)
    if r > 10 * s_u > 100 * s_b  # Enforce rho >> sigma_u >>> sigma_b
]

# CSV File Path
output_file = "data/SSA_finetune_results.csv"

# Write header only if file doesn't exist
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("sigma_u,sigma_b,rho,d,fano_m,expected_fano_m_approx,expected_fano_m_exact\n")

if __name__ == "__main__":
    with multiprocessing.Pool() as pool: # automatically uses all CPU cores, use multiprocessing.Pool(processes=num_cores_to_use) to modify the number of cores
        with tqdm.tqdm(total=len(filtered_param_grid), desc='Overall Progress', position=0, dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(run_simulation, filtered_param_grid):

                # unpack the results
                sigma_u, sigma_b, rho, d, fano_m, expected_fano_m_approx, expected_fano_m_exact = result
                # update progress bar description
                pbar.set_description(f"sigma_u={sigma_u:.2f}, sigma_b={sigma_b:.2f}, rho={rho:.2f}, d={d:.2f}")

                # save results after each iteration
                with open(output_file, "a") as f:
                    f.write(",".join(map(str, result)) + "\n")

                # progress bar increment by 1
                pbar.update(1)