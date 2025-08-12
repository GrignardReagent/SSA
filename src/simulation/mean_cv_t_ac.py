#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import root_scalar
from stats.cv import calculate_cv_from_params
from stats.mean import calculate_mean_from_params
from stats.autocorrelation import calculate_ac_from_params
from utils.biological import check_biological_appropriateness

# # Rescale parameters
# rho_tilda = rho / (sigma_b + sigma_u)
# d_tilda = d / (sigma_b + sigma_u)
# sigma_b_tilda = sigma_b / (sigma_b + sigma_u)
# sigma_u_tilda = sigma_u / (sigma_b + sigma_u)
# # t_ac_tilda is the only one that's multiplied by the scaling factor 
# t_ac_tilda = t_ac * (sigma_b + sigma_u)

# Compute rescaled parameters for a fixed ``sigma_sum``.
#
# This helper solves the mean, CV and autocorrelation time equations in the
# rescaled space assuming a known sum of switching rates ``sigma_sum`` (default = 1).
# It is intentionally kept private; the public :func:`find_tilda_parameters`
# routine wraps this function and automatically searches for an appropriate
# ``sigma_sum``.
def find_tilda_parameters(
    mu_target: float,
    t_ac_target: float,
    cv_target: float,
    sigma_sum: float = 1.0,
    ac_target: float = np.exp(-1),
    res_limit=1e-3,
):
    """Solve for ``rho``, ``d``, ``sigma_b`` and ``sigma_u`` given ``sigma_sum``.

    The equations are expressed in a rescaled space where all parameters are divided by ``sigma_sum`` to simplify the algebra.  

    Parameters
    ----------
    mu_target : float
        Target mean of the system.
    t_ac_target : float
        Target autocorrelation time.
    cv_target : float
        Desired coefficient of variation.
    sigma_sum : float, optional
        The sum ``sigma_b + sigma_u`` used to scale the system. Default to be 1.0. Changing this might lead to unexpected bahaviours!
    ac_target : float, optional
        Target autocorrelation value at ``t_ac_target``. Defaults to ``exp(-1)``.
    res_limit : float, optional
        Residual limit for root-finding. Defaults to ``1e-3``. This is used to check if the solution found is valid or not.

    Returns
    -------
    tuple
        ``(rho, d, sigma_b, sigma_u)`` that reproduce the requested targets.

    Raises
    ------
    ValueError
        If no valid solution for ``d`` can be found.
    """
    
    # using equation B, we find v via mu_target & cv_target^2
    v = (mu_target * (cv_target ** 2)) - 1
    
    # check if v > 0, if not, there's no solution
    if v <= 0:
        raise ValueError(f"Invalid parameters: v = (mu_target * cv_target ** 2) - 1 must be positive, got v = {v}. Reconsider mu_target and cv_target choices.")
    
    # t_ac_tilda = (sigma_b + sigma_u) * t; 
    t_ac_tilda = t_ac_target * sigma_sum
    
    # find d_tilda from equation C, via t_ac_tilda, v
    def scaled_ac_equation(d_tilda):
        '''
        AC(t_ac_tilda), rescaled equation of AC(t) to find d_tilda:
        '''
        
        scaled_ACmRNA_eq = ((d_tilda * v * np.exp(- t_ac_tilda) + (d_tilda - v - 1) * np.exp(- d_tilda * t_ac_tilda)) / ((d_tilda -1) * (v + 1)))

        
        return float(scaled_ACmRNA_eq - ac_target)
    
    ################ Use root scalar to find d_tilda, AC(t_ac_tilda) - AC_target = 0  is a root-finding problem, so root_scalar is more appropriate ##################
    
    # the AC(t_ac_tilda) formula has the denominator (d_tilda -1)(v + 1), and d_tilda = 1 is undefined, so the search space needs to exclude 1.0
    lower, upper = 1e-3, 1e3
    candidates = [(lower, 0.999), (1.001, upper)] if (lower < 1.0 < upper) else [(lower, upper)]  # several decimal places are intentionally used here to avoid values close to 1.0
    
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
                # res_limit of 1e-6 may be too strict 
                if abs(residue) > res_limit or d_tilda_solution is None or np.isclose(d_tilda_solution, 1e3, atol=1e-4):
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

    return rho, d, sigma_b, sigma_u