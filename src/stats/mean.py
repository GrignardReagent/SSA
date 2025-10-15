#!/usr/bin/python

from utils.steady_state import find_steady_state
from utils.data_processing import _ensure_numpy, _safe_slice

def calculate_mean(trajectories, parameter_set, use_steady_state=True):
    """
    Calculate the mean from trajectories.
    
    Parameters:
        trajectories (numpy array): Array of trajectories with shape (n_trajectories, n_timepoints).
        parameter_set (dict): Parameter set containing the degradation rate 'd' to determine steady state.
        use_steady_state (bool, optional): Whether to use only steady state portion. Defaults to True.
        
    Returns:
        float: Mean of trajectories (steady state or entire trajectory based on use_steady_state).
    """
    if use_steady_state:
        # Find steady state time and index
        _, steady_state_index = find_steady_state(parameter_set[0])
        
        # Extract steady state trajectories
        steady_state_trajectories = _safe_slice(trajectories, steady_state_index)
        
        # Calculate and return mean
        return steady_state_trajectories.mean()
    else:
        # Calculate mean from entire trajectory
        return trajectories.mean()
    
def calculate_mean_from_params(rho, d, sigma_b, sigma_u):
    '''
    Calculate mean from parameters.
    
    Parameters:
        rho (float): The value of rho.
        d (float): The value of d.
        sigma_b (float): The value of sigma_b.
        sigma_u (float): The value of sigma_u.
        
    Returns:
        float: The calculated mean.
    '''
    return sigma_b * rho / ( d * (sigma_b + sigma_u) ) 