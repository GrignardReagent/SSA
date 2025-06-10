#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, check_grad
from simulation.mean_var_autocorr import equations

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

#  a quick version of find_parameters, to fix mean, autocorrelation and CV ONLY
def quick_find_parameters(sigma_b, mu_target=None, autocorr_target=None, cv_target=None):
    ''' Providing we are only fixing CV, mean and autocorrelation.
    Args:
        sigma_b (float): Value of sigma_b.
        mu_target (float, optional): Target mean.
        autocorr_target (float, optional): Target autocorrelation time.
        cv_target (float, optional): Target coefficient of variation.
    Returns:
        list: List of parameters (rho, sigma_u, d) that satisfy the equations.
    '''
    # Check if we have the required targets for this specialized solver
    if not (mu_target is not None and cv_target is not None and autocorr_target is not None):
        raise ValueError("quick_find_parameters requires mu_target, cv_target, and autocorr_target to be specified")
    
    # Step 1: Find d using numerical root finding with exponential transformation
    # Use d = exp(D) to ensure d is always positive
    def d_equation_exp(D):
        # Convert D to d using exponential transformation
        d_val = np.exp(D)
        
        # we use CV^2 to make calculations easier for the solvers
        cv_sq_target = cv_target ** 2
        # Use np.exp consistently for numerical calculations
        return -1/np.exp(1) + (-(d_val*mu_target*(1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target))) + (-1 + cv_sq_target*mu_target**2 + cv_sq_target**2*mu_target**2 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target)**2)*sigma_b)/(cv_sq_target*np.exp(autocorr_target*d_val)*mu_target*(-(d_val*mu_target) + (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))

    # Use a more robust solver with multiple initial guesses in log space
    # Try values around log(1/autocorr_target), covering a reasonable range
    center = np.log(1.0 / autocorr_target)
    spread = 5.0  # +/- 5 in log space gives good coverage
    D_guesses = np.linspace(center - spread, center + spread, 15)
    d_value = None
    
    for D_guess in D_guesses:
        try:
            result = fsolve(d_equation_exp, D_guess, full_output=True)
            if result[2] == 1:  # Check if converged
                D_solution = result[0][0]
                d_candidate = np.exp(D_solution)  # Transform back to d
                # d is guaranteed to be positive due to exp transform
                d_value = d_candidate
                
                # Step 2: Calculate the corresponding sigma and rho
                # need to use cv_sq_target so it's easier to solve
                cv_sq_target = cv_target ** 2
                sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_value + sigma_b))/(-(d_value*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))

                # Step 3: Calculate rho using the mean equation
                rho = (d_value * mu_target * (sigma_b + sigma_u)) / sigma_b
                
                # Check if the solutions are valid (all parameters positive), if not, try the next guess
                if sigma_u <= 0 or rho <= 0:
                    print(f"Warning: Invalid solution for D={D_guess}: rho={rho}, sigma_u={sigma_u}, d={d_value}")
                    continue
                else:
                    print(f"Found valid solution for D={D_guess}: rho={rho}, sigma_u={sigma_u}, d={d_value}")
                    break
        except Exception as e:
            print(f"Warning: Error during solving for D={D_guess}: {str(e)}")
            continue
    
    if d_value is None:
        raise ValueError("Could not find a valid solution for parameter d")
    
    # Return the solution as a tuple
    solution = (rho, sigma_u, d_value)
    
    # Verify the solution by checking residuals
    residuals = equations(solution, sigma_b, mu_target, None, autocorr_target, cv_target, None)
    if all(abs(res) < 1e-4 for res in residuals):
        print(f"Found solution: rho={rho:.4f}, sigma_u={sigma_u:.4f}, d={d_value:.4f}")

        return solution
    else:
        print(f"Warning: Solution found but residuals are high: {residuals}")
        return solution