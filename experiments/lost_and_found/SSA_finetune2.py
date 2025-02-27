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
# Columns: G, G*, M, P
update_matrix = np.array([
    [-1, 1, 0, 0],   # G -> G* (Gene activation)
    [1, -1, 0, 0],   # G* -> G (Gene deactivation)
    [0, 0, 1, 0],    # G -> G + M (mRNA production)
    [0, 0, -1, 0],   # M -> 0 (mRNA degradation)
    [0, 0, 0, 1],    # M -> M + P (protein production)
    [0, 0, 0, -1]    # P -> 0 (protein degradation)
], dtype=int)

def circuit_propensity(propensities, population, t, alpha_off, alpha_on, beta, gamma_m, gamma_p, protein_degradation):
    """
    Updates the propensities for the circuit reactions based on defined rates.

    Parameters:
        propensities: Array of propensities
        population: Array of species populations [G, G*, M, P]
        t: Time (not used but required by signature)
        alpha_off: Rate of G -> G*
        alpha_on: Rate of G* -> G
        beta: Rate of G -> G + M (mRNA production)
        gamma_m: Rate of M -> 0 (mRNA degradation)
        gamma_p: Rate of M -> M + P (protein production)
        protein_degradation: Rate of P -> 0 (protein degradation)
    """
    # Unpack population
    G, G_star, M, P = population
    
    # Update propensities for each reaction
    propensities[0] = alpha_off * G         # G -> G*
    propensities[1] = alpha_on * G_star     # G* -> G
    propensities[2] = beta * G              # G -> G + M
    propensities[3] = gamma_m * M           # M -> 0
    propensities[4] = gamma_p * M           # M -> M + P
    propensities[5] = protein_degradation * P  # P -> 0

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

# Worker function for running simulations
def run_simulation(params):
    a_off, a_on, b, g_m, g_p, p_d = params
    time_points = np.linspace(0, 200, 201)

    # Ensure steady-state condition t_ss >> 10 / gamma_m
    if len(time_points) < int(10 / g_m):
        time_points = np.linspace(0, int(10 / g_m), int(10 / g_m + 1))

    population_0 = np.array([1, 0, 0, 0], dtype=int)  # Initial [G, G*, M, P]
    samples = np.empty((size, len(time_points), 4), dtype=int)

    for i in range(size):
        samples[i, :, :] = gillespie_ssa(circuit_propensity, update_matrix, population_0, time_points, args=(a_off, a_on, b, g_m, g_p, p_d))
    
    # Compute Fano factor
    fano_m = samples[:, :, 2].var() / samples[:, :, 2].mean()
    expected_fano_m = 1 + b / a_off

    # Save samples to a file
    sample_file = f"samples_{a_off}_{a_on}_{b}_{g_m}_{g_p}_{p_d}.npy"
    np.save(sample_file, samples)

    return [a_off, a_on, b, g_m, g_p, p_d, fano_m, expected_fano_m]

# Define the rates for each reaction (constants)
# To have matching theoretical and actual Fano factors, we need to: 
# 1. Set alpha_off >> alpha_on
# 2. Set beta >> alpha_off
# 3. Reaction sampled at Steady State, and t_ss >> 10 / gamma_m

# define a list of parameters for running simulations, but we need to make sure beta >> alpha_off >> alpha_on
alpha_off = np.logspace(-1, 2.3, 10)  # 0.1 to ~200, logarithmic
alpha_on = np.logspace(-2, 2, 10)    # 0.01 to 100, logarithmic
beta = np.linspace(1, 500, 10)       # 1 to 500, linear
gamma_m = np.logspace(-2, 2, 5)     # 0.01 to 100, logarithmic
gamma_p = np.logspace(-2, 2, 5)     # 0.01 to 100, logarithmic
protein_degradation = np.linspace(1, 10, 5)  # 1 to 10, linear

size = 1000  # Number of simulations
# Generate parameter grid with filtering conditions to enforce beta >> alpha_off >> alpha_on
filtered_param_grid = [
    (a_off, a_on, b, g_m, g_p, p_d)
    for a_off, a_on, b, g_m, g_p, p_d in itertools.product(alpha_off, alpha_on, beta, gamma_m, gamma_p, protein_degradation)
    if b > 10 * a_off > 10 * a_on  # Enforce beta >> alpha_off >> alpha_on
]

# CSV File Path
output_file = "SSA_finetune_results.csv"

# Write header only if file doesn't exist
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.write("alpha_off,alpha_on,beta,gamma_m,gamma_p,protein_degradation,fano_m,expected_fano_m\n")

if __name__ == "__main__":
    with multiprocessing.Pool() as pool: # automatically uses all CPU cores, use multiprocessing.Pool(processes=num_cores_to_use) to modify the number of cores
        with tqdm.tqdm(total=len(filtered_param_grid), desc='Overall Progress', position=0, dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(run_simulation, filtered_param_grid):

                # unpack the results
                a_off, a_on, b, g_m, g_p, p_d, *_ = result
                # update progress bar description
                pbar.set_description(f"a_off={a_off:.2f}, a_on={a_on:.2f}, b={b:.2f}, g_m={g_m:.2f}, g_p={g_p:.2f}, p_d={p_d:.2f}")

                # save results after each iteration
                with open(output_file, "a") as f:
                    f.write(",".join(map(str, result)) + "\n")

                # progress bar increment by 1
                pbar.update(1)