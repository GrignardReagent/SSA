import matplotlib.pyplot as plt
import multiprocessing
import tqdm
import numpy as np
import scipy.stats as st
import numba
import biocircuits


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


# Define the rates for each reaction (constants)
# To have matching theoretical and actual Fano factors, we need to: 
# 1. Set alpha_off >> alpha_on
# 2. Set beta >> alpha_off
# 3. Reaction sampled at Steady State, and t_ss >> 10 / gamma_m


# define a list of parameters for running simulations, but we need to make sure beta >> alpha_off >> alpha_on
alpha_off = [0.1, 1, 10, 100, 200]
alpha_on = [0.01, 0.1, 1, 10, 100]
beta = [1, 10, 100, 200, 500]
gamma_m = [0.01, 0.1, 1, 10, 100]
gamma_p = [0.01, 0.1, 1, 10, 100]
protein_degradation = [1,10] 

population_0 = np.array([1, 0, 0, 0], dtype=int)  # Initial [G, G*, M, P]
size = 1000  # Number of stochastic simulations


# run the simulation for all the parameters
results = []
for a_off in tqdm.tqdm(alpha_off, desc="alpha_off"):
    for a_on in tqdm.tqdm(alpha_on, desc="alpha_on", leave=False):
        for b in tqdm.tqdm(beta, desc="beta", leave=False):
            for g_m in tqdm.tqdm(gamma_m, desc="gamma_m", leave=False):
                for g_p in tqdm.tqdm(gamma_p, desc="gamma_p", leave=False):
                    for p_d in tqdm.tqdm(protein_degradation, desc="protein_degradation", leave=False):
                        params = (a_off, a_on, b, g_m, g_p, p_d)
                        time_points = np.linspace(0, 200, 201)      # Time range
                        
                        # make sure we have t_ss >> 10 / gamma_m
                        if len(time_points) < int(10/g_m):
                            time_points = np.linspace(0, int(10/g_m), int(10/g_m+1))

                        # Seed random number generator for reproducibility
                        np.random.seed(42)
                        samples = np.empty((size, len(time_points), 4), dtype=int)
                        for i in range(size):
                            samples[i, :, :] = gillespie_ssa(circuit_propensity, update_matrix, population_0, time_points, args=params)
                            
                        fano_m = samples[:,:,2].var() / samples[:,:,2].mean()
                        expected_fano_m = 1 + b / a_off
                        results.append([a_off, a_on, b, g_m, g_p, p_d, fano_m, expected_fano_m])

# Save the results to a file
results = np.array(results)
np.savetxt("results.csv", results, delimiter=",", header="alpha_off,alpha_on,beta,gamma_m,gamma_p,protein_degradation,fano_m,expected_fano_m", comments="")
