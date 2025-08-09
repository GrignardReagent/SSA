#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, check_grad, minimize_scalar, root_scalar
from stats.fano_factor import calculate_fano_factor,calculate_fano_factor_from_params
from stats.cv import calculate_cv

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
                          ):
    ''''Find the rescaled parameters rho_tilda, d_tilda, sigma_b_tilda, sigma_u_tilda given target mean, autocorrelation, CV and the scaling factor sigma_sum.
    Parameters:
        sigma_sum (float): The sum of sigma_b and sigma_u, scaling factor used to simplify the mu, autocorr and cv equations.
        mu_target (float): Target mean of the system.
        ac_target (float): Target autocorrelation value, default is exp(-1) for steady state.
        t_ac_target (float): Target autocorrelation time.
        cv_target (float): Target coefficient of variation.
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

    ################ Use minimization ##################
    # # Square the equation for solutions where f(d_tilda) = 0
    # scaled_ac_equation_sq = lambda D_tilda: scaled_ac_equation(D_tilda) ** 2
    # try:
    #     result = minimize_scalar(scaled_ac_equation_sq, method='bounded', bounds=(1e-3, 1e3))
        
    #     if result.success:
    #         d_tilda_solution = result.x
            
    #         # error trap for no solution found for d_tilda_solution
    #         # 1. checks that absolute value of residue is less than 1e-6
    #         # 2. checks that d_tilda_solution is not None
    #         # 3. checks that the value of d_tilda_solution is not close to the bounds plus/minus the absolute tolerance (atol)
    #         if abs(result.fun) > 1e-6 or d_tilda_solution is None or np.isclose(d_tilda_solution, 1e3, atol=1e-4):
    #             raise  ValueError(f"⚠️No valid solution found for parameter d_tilda: residual={result.fun:.2e}, d_tilda={d_tilda_solution:.4f}")
            
    #         d_tilda = d_tilda_solution
    # except Exception as e: 
    #     raise ValueError(f'Could not find a valid solution for d_tilda: {e}')
    ################ Use minimization ##################
    
    ################ Use root scalar to find d_tilda, AC(t_tilda) - AC_target = 0  is a root-finding problem, so root_scalar is more appropriate ##################
    
    # the AC(t_tilda) formula has the denominator (d_tilda -1)(v + 1), and d_tilda = 1 is undefined, so the search space needs to exclude 1.0
    lower, upper = 1e-3, 1e3
    if lower < 1.0 < upper:
        candidates = [(lower, 0.999),
                        (1.001, upper)]
    else:
        candidates = [(lower, upper)]
    
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
                    raise  ValueError(f"⚠️No valid solution found for parameter d_tilda: residual={result.fun:.2e}, d_tilda={d_tilda_solution:.4f}")
                
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


# # define rescaled equations A, B and C
# def scaled_equations(tilda_params, mu_target, t_ac_target, cv_target):
#     '''Rescaled equations to solve for tilda parameters.'''
#     rho_tilda, d_tilda, sigma_b_tilda, sigma_u_tilda = tilda_params
#     # v = sigma_u_tilda * rho_tilda / (1 + d_tilda) # may add to computational cost...
    
#     # Equation A, mean
#     mean_eqn = (sigma_b_tilda * rho_tilda / d_tilda) - mu_target
    
#     # Equation B, CV^2 ; 
#     cv_sq_eqn = ((1 + v) / mu_target) - cv_target ** 2
    
#     # Equation C, AC(t_tilda) ; solved by d_tilda
#     ac_eqn = ((d_tilda* v * sp.exp(-t_tilda) + (d_tilda - v - 1) * sp.exp(-d_tilda * t_tilda)) / ((d_tilda -1) * (v + 1))) - sp.exp(-1)
    
    
#     # Return the equations to be solved
#     return [
#         mean_eqn,
#         cv_sq_eqn,
#         ac_eqn
#     ]