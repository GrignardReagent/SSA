#!/usr/bin/python
from stats.mean import calculate_mean_from_params
from stats.variance import calculate_variance_from_params
import numpy as np

def calculate_cv(variance, mean):
    """
    Calculate the coefficient of variation (CV).
    
    Args:
        variance (float): The variance.
        mean (float): The mean.
        
    Returns:
        float: The calculated coefficient of variation.
    """
    import numpy as np
    cv = np.sqrt(variance) / mean
    return cv

def calculate_cv_from_params(rho, d, sigma_b, sigma_u):
    """
    Calculate the coefficient of variation (CV) from parameters.
    Equation: CV = sqrt(Var)/Mean
    
    Parameters:
        rho (float): The value of rho.
        d (float): The value of d.
        sigma_b (float): The value of sigma_b.
        sigma_u (float): The value of sigma_u.
        
    Returns:
        float: The calculated coefficient of variation.
    """
    mu = calculate_mean_from_params(rho, d, sigma_b, sigma_u)
    var = calculate_variance_from_params(rho, d, sigma_b, sigma_u)
    return np.sqrt(var) / mu