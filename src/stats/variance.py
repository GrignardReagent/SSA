#!/usr/bin/python

from utils.steady_state import find_steady_state

def calculate_variance(trajectories, parameter_set, use_steady_state=True):
    """
    Calculate the variance from trajectories.
    
    Parameters:
        trajectories (numpy array): Array of trajectories with shape (n_trajectories, n_timepoints).
        parameter_set (dict): Parameter set containing the degradation rate 'd' to determine steady state.
        use_steady_state (bool, optional): Whether to use only steady state portion. Defaults to True.
        
    Returns:
        float: Variance of trajectories (steady state or entire trajectory based on use_steady_state).
    """
    if use_steady_state:
        # Find steady state time and index
        _, steady_state_index = find_steady_state(parameter_set[0])
        
        # Extract steady state trajectories
        steady_state_trajectories = trajectories[:, steady_state_index:]
        
        # Calculate and return variance
        return steady_state_trajectories.var()
    else:
        # Calculate variance from entire trajectory
        return trajectories.var()

def calculate_variance_from_params(rho, d, sigma_b, sigma_u):
    '''
    Calculate variance from parameters.
    
    Parameters:
        rho (float): The value of rho.
        d (float): The value of d.
        sigma_b (float): The value of sigma_b.
        sigma_u (float): The value of sigma_u.
    
    Returns:
        float: The calculated variance.
    '''
    sigma_sum = sigma_b + sigma_u
    term1 = sigma_b * rho / (d * sigma_sum)
    term2 = (sigma_u * sigma_b * (rho**2)) / (d * (sigma_sum + d) * (sigma_sum**2))
    return term1 + term2
