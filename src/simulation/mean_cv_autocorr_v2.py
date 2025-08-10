#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, check_grad, minimize_scalar, root_scalar
from stats.fano_factor import calculate_fano_factor, calculate_fano_factor_from_params
from stats.cv import calculate_cv, calculate_cv_from_params
from stats.mean import calculate_mean_from_params
from stats.variance import calculate_variance_from_params
from stats.autocorrelation import calculate_ac_from_params

def check_biological_appropriateness(variance_target, mu_target, max_fano_factor=20, min_fano_factor=1, max_cv=5.0):
    '''
    Check if the solution is biologically appropriate based on Fano factor and CV.
    Args:
        variance_target (float): Target variance.
        mu_target (float): Target mean.
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
        print(f"⚠️ WARNING: CV {cv:.2f} > {max_cv}, consider changing the target variance or mean.")
    elif fano_factor >= max_fano_factor:
        print(f"⚠️ WARNING: Fano factor {fano_factor:.2f} > {max_fano_factor}, consider changing the target parameters.")
    elif fano_factor < min_fano_factor:
        print(f"⚠️ WARNING: Fano factor {fano_factor:.2f} < {min_fano_factor}, consider changing the target parameters.")
    else:
        print(f"✅ System is biologically appropriate with Fano factor: {fano_factor:.2f}, CV: {cv:.2f}")
        appropriateness = True
    
    return appropriateness

# # Rescale parameters
# rho_tilda = rho / (sigma_b + sigma_u)
# d_tilda = d / (sigma_b + sigma_u)
# sigma_b_tilda = sigma_b / (sigma_b + sigma_u)
# sigma_u_tilda = sigma_u / (sigma_b + sigma_u)
# # t_tilda is the only one that's multiplied by the scaling factor 
# t_tilda = t * (sigma_b + sigma_u)

############## EXPERIMENTAL #####################
# Version 2 of quick_find_parameters, to fix mean, autocorrelation and CV ONLY, but rescalling parameters so it's easier to solve
def find_tilda_parameters(sigma_sum: float,
                          mu_target: float,
                          t_ac_target: float,
                          cv_target: float,
                          ac_target: float = np.exp(-1),
                          check_biological: bool = False,
                          max_fano_factor: float = 20,
                          min_fano_factor: float = 1,
                          max_cv: float = 5.0
                          ):
    ''''
    Find the rescaled parameters rho_tilda, d_tilda, sigma_b_tilda, sigma_u_tilda given target mean, autocorrelation, CV and the scaling factor sigma_sum.
    v = sigma_u_tilda * rho_tilda / (1 + d_tilda) # may add to computational cost...
    
    Equation A: mean
    mean_eqn = (sigma_b_tilda * rho_tilda / d_tilda) - mu_target
    
    Equation B: CV^2 ; 
    cv_sq_eqn = ((1 + v) / mu_target) - cv_target ** 2
    
    Equation C: AC(t_tilda) ; solved by d_tilda
    ac_eqn = ((d_tilda* v * sp.exp(-t_tilda) + (d_tilda - v - 1) * sp.exp(-d_tilda * t_tilda)) / ((d_tilda -1) * (v + 1))) - sp.exp(-1)
    
    Parameters:
        sigma_sum (float): The sum of sigma_b and sigma_u, scaling factor used to simplify the mu, autocorr and cv equations.
        mu_target (float): Target mean of the system.
        ac_target (float): Target autocorrelation value, default is exp(-1) for steady state.
        t_ac_target (float): Target autocorrelation time.
        cv_target (float): Target coefficient of variation.
        check_biological (bool): Whether to check if the solution is biologically appropriate.
        max_fano_factor (float): Maximum allowed Fano factor for biological appropriateness.
        min_fano_factor (float): Minimum allowed Fano factor for biological appropriateness.
        max_cv (float): Maximum allowed coefficient of variation for biological appropriateness.
        
    Returns:
        tuple: A tuple containing the found parameters (rho, d, sigma_b, sigma_u).
        
    Raises:
        ValueError: If parameters are invalid or if biological check fails when enabled.
    '''
    
    # using equation B, we find v via mu_target & cv_target^2
    v = (mu_target * (cv_target ** 2)) - 1
    
    # check if v > 0, if not, there's no solution
    if v <= 0:
        raise ValueError(f"Invalid parameters: v = (mu_target * cv_target ** 2) - 1 must be positive, got v = {v}. Reconsider mu_target and cv_target choices.")
    
    # t_tilda = (sigma_b + sigma_u) * t; 
    t_tilda = t_ac_target * sigma_sum
    
    # find d_tilda from equation C, via t_tilda, v
    def scaled_ac_equation(d_tilda):
        '''Equation C: AC(t_tilda)'''
        
        scaled_ACmRNA_eq = ((d_tilda * v * np.exp(- t_tilda) + (d_tilda - v - 1) * np.exp(- d_tilda * t_tilda)) / ((d_tilda -1) * (v + 1)))

        
        return float(scaled_ACmRNA_eq - ac_target)
    
    ################ Use root scalar to find d_tilda, AC(t_tilda) - AC_target = 0  is a root-finding problem, so root_scalar is more appropriate ##################
    
    # the AC(t_tilda) formula has the denominator (d_tilda -1)(v + 1), and d_tilda = 1 is undefined, so the search space needs to exclude 1.0
    lower, upper = 1e-3, 1e3
    candidates = [(lower, 0.999), (1.001, upper)] if (lower < 1.0 < upper) else [(lower, upper)]
    
    d_tilda = None
    for a, b in candidates:
        try:
            result = root_scalar(scaled_ac_equation, bracket=[a, b],method='brentq', # Brent's method for root finding, generally considered the best for this. 
                                 xtol=1e-10, rtol=1e-10, maxiter=200)
            
            if result.converged:
                d_tilda_solution = result.root
                
                # error trap for no solution found for d_tilda_solution
                # 1. checks that absolute value of residue is less than 1e-6
                # 2. checks that d_tilda_solution is not None
                # 3. checks that the value of d_tilda_solution is not close to the bounds plus/minus the absolute tolerance (atol)
                residue = scaled_ac_equation(d_tilda_solution)
                if abs(residue) > 1e-6 or d_tilda_solution is None or np.isclose(d_tilda_solution, 1e3, atol=1e-4):
                    raise  ValueError(f"⚠️No valid solution found for parameter d_tilda: residual={residue:.2e}, d_tilda={d_tilda_solution:.4f}")
                
                d_tilda = d_tilda_solution
                
        except ValueError:
            pass
    if d_tilda is None: 
        raise ValueError(f'Could not find a valid solution for d_tilda using root_scalar method.')
    
    ################ Use root scalar to find d_tilda ##################
            
    # find rho_tilda from mu_target, d_tilda and v; by rearranging Equation A, definition of v in terms of tilda_params.
    #TODO: explanation of maths...
    rho_tilda = mu_target * d_tilda + (d_tilda + 1) * v
    
    # find sigma_b_tilda using solved parameters
    sigma_b_tilda = mu_target * d_tilda / (mu_target * d_tilda + (d_tilda + 1) * v)

    # find sigma_u_tilda using solved parameters
    sigma_u_tilda = (d_tilda + 1) * v / (mu_target * d_tilda + (d_tilda + 1) * v)
    
    # find sigma_b by scaling back
    sigma_b = sigma_b_tilda * sigma_sum
    sigma_u = sigma_u_tilda * sigma_sum
    rho = rho_tilda * sigma_sum 
    d = d_tilda * sigma_sum

    # Perform biological check if requested
    if check_biological:
        # Calculate the implied variance from CV and mean
        variance_target = (cv_target * mu_target) ** 2
        
        # Check biological appropriateness
        is_appropriate = check_biological_appropriateness(
            variance_target, mu_target, max_fano_factor, min_fano_factor, max_cv
        )
        
        if not is_appropriate:
            raise ValueError(
                f"Solution is not biologically appropriate. "
                f"Consider adjusting target parameters: mu={mu_target}, cv={cv_target}, "
                f"which gives variance={variance_target:.2f}"
            )

    return rho, d, sigma_b, sigma_u
