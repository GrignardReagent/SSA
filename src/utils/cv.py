#!/usr/bin/python

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
