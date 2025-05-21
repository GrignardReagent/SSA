#!/usr/bin/python

def calculate_fano_factor(rho, sigma_b, d, sigma_u):
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