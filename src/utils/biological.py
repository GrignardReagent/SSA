#!/usr/bin/python

from stats.fano_factor import calculate_fano_factor
from stats.cv import calculate_cv

def check_biological_appropriateness(
    variance_target, 
    mu_target, 
    watch_fano_factor = False, 
    max_fano_factor=20, 
    min_fano_factor=1, 
    max_cv=5.0
    ):
    '''
    Check if the solution is biologically appropriate based on Fano factor and CV.
    Args:
        variance_target (float): Target variance.
        mu_target (float): Target mean.
        watch_fano_factor (bool): If True, check Fano factor. 
        max_fano_factor (float): Maximum allowed Fano factor.
        min_fano_factor (float): Minimum allowed Fano factor.
        max_cv (float): Maximum allowed coefficient of variation.
    Returns:
        bool: True if the system is biologically appropriate, False otherwise.
    '''
    
    # Check Fano factor
    fano_factor = calculate_fano_factor(variance_target, mu_target)
    
    # Check coefficient of variation 
    cv = calculate_cv(variance_target, mu_target)
    
    # Initialize appropriateness as False
    appropriateness = False
    
    if cv >= max_cv:
        print(f"❌ WARNING: CV {cv:.2f} > {max_cv}, consider changing the target variance or mean.")
    elif fano_factor >= max_fano_factor:
        print(f"⚠️ WARNING: Fano factor {fano_factor:.2f} > {max_fano_factor}, consider changing the target parameters.")
        if not watch_fano_factor:
            appropriateness = True
    elif fano_factor < min_fano_factor:
        print(f"⚠️ WARNING: Fano factor {fano_factor:.2f} < {min_fano_factor}, consider changing the target parameters.")
        if not watch_fano_factor:
            appropriateness = True
    else:
        print(f"✅ System is biologically appropriate with Fano factor: {fano_factor:.2f}, CV: {cv:.2f}")
        appropriateness = True
    
    return appropriateness

def find_biological_variance_mean(
    desired_fano_factor, 
    desired_cv, 
    max_fano_factor=20, 
    max_cv=5.0
    ):
    """
    Find biologically appropriate levels of variance and mean based on Fano factor and CV constraints.
    
    This function uses the relationships:
    - fano_factor = variance / mean
    - cv = sqrt(variance) / mean
    
    Args:
        desired_fano_factor (float): Target Fano factor.
        desired_cv (float): Target coefficient of variation.
        max_fano_factor (float): Maximum allowed Fano factor. Default is 20.0.
        max_cv (float): Maximum allowed coefficient of variation. Default is 5.0.
        
    Returns:
        tuple: A tuple containing (variance, mean) that satisfy the biological constraints.
    """
    # Check if the desired values exceed the maximum allowed values
    if desired_fano_factor > max_fano_factor:
        print(f"Warning: Desired Fano factor {desired_fano_factor} exceeds maximum allowed {max_fano_factor}, setting to max ({max_fano_factor}).")
        desired_fano_factor = max_fano_factor
    
    if desired_cv > max_cv:
        print(f"Warning: Desired CV {desired_cv} exceeds maximum allowed {max_cv}, setting to max ({max_cv}).")
        desired_cv = max_cv
    
    # From the two equations:
    # fano_factor = variance / mean
    # cv = sqrt(variance) / mean
    #
    # We can derive:
    # variance = fano_factor * mean
    # cv^2 = variance / mean^2
    # Substituting:
    # cv^2 = (fano_factor * mean) / mean^2
    # cv^2 = fano_factor / mean
    # mean = fano_factor / cv^2
    # variance = fano_factor * mean = fano_factor^2 / cv^2
    
    mean = desired_fano_factor / (desired_cv**2)
    variance = desired_fano_factor * mean
    
    print(f"For Fano factor = {desired_fano_factor} and CV = {desired_cv}:")
    print(f"  Mean = {mean:.2f}")
    print(f"  Variance = {variance:.2f}")

    return  variance, mean