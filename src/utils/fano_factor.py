#!/usr/bin/python

def calculate_fano_factor(variance, mean):
    """
    Calculate the Fano factor for a given variance and mean.
    
    Args:
        variance (float): The variance of the distribution.
        mean (float): The mean of the distribution.

    Returns:
        float: The calculated Fano factor.
    """
    if mean == 0:
        print("WARNING: Mean is zero, cannot calculate Fano factor.")
        return float('inf')  # Avoid division by zero
    fano_factor = variance / mean
    return fano_factor

def calculate_fano_factor_from_params(rho, sigma_b, d, sigma_u):
    """
    Calculate the Fano factor for a given set of parameters.
    
    Args:
        rho (float): The value of rho.
        sigma_b (float): The value of sigma_b.
        d (float): The value of d.
        sigma_u (float): The value of sigma_u.

    Returns:
        float: The calculated Fano factor.
    """
    fano_factor = 1 + (rho * sigma_u) / ((sigma_b + sigma_u) * (sigma_b + d + sigma_u))
    return fano_factor