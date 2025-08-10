#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from simulation.mean_var_autocorr import equations

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

################## EXPERIMENTAL, THIS IS NOT WORKING VERY WELL #####################
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
    def d_equation(D):
        # Convert D to d using exponential transformation
        d_val = D
        
        # we use CV^2 to make calculations easier for the solvers
        cv_sq_target = cv_target ** 2
        # Use np.exp consistently for numerical calculations
        return -1/np.exp(1) + (-(d_val*mu_target*(1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target))) + (-1 + cv_sq_target*mu_target**2 + cv_sq_target**2*mu_target**2 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target)**2)*sigma_b)/(cv_sq_target*np.exp(autocorr_target*d_val)*mu_target*(-(d_val*mu_target) + (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))
    
    def d_equation_sq_minus_sigma_u(D):
        '''This is f(d)^2 - sigma_u'''

        d_val = D
        
        # we use CV^2 to make calculations easier for the solvers
        cv_sq_target = cv_target ** 2
        
        sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_val + sigma_b))/(-(d_val*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))
        
        # Use np.exp consistently for numerical calculations
        return (-1/np.exp(1) + (-(d_val*mu_target*(1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target))) + (-1 + cv_sq_target*mu_target**2 + cv_sq_target**2*mu_target**2 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target)**2)*sigma_b)/(cv_sq_target*np.exp(autocorr_target*d_val)*mu_target*(-(d_val*mu_target) + (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b)))**2 - sigma_u
    
    # Square the equation for solutions where f(d) = 0
    d_equation_sq = lambda D: d_equation(D) ** 2
    
    # initialise system parameters
    D_guess = 1.0 / autocorr_target
    d_value = None
    rho = None
    sigma_u = None

    try:
        ################ Use minimization on d_equation_sq ##################
        result = minimize_scalar(d_equation_sq, method='bounded', bounds=(1e-3, 1e3))
        
        if result.success:
            D_solution = result.x
            d_value = D_solution
            # error trap for no solution found for d
            if result.fun > 1e-6 or d_value is None or d_value <= 0 or np.isclose(d_value, 1e3, atol=1e-4):
                # print(f'⚠️ WARNING: No solution found for d')
                raise ValueError("⚠️No valid solution found for parameter d")

            # Compute sigma_u safely
            cv_sq_target = cv_target ** 2
            sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_value + sigma_b))/(-(d_value*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))

            if sigma_u < 0:
                raise ValueError(f"Minimization led to negative sigma_u: {sigma_u:.4f}")

            # Compute rho from mean constraint
            rho = (d_value * mu_target * (sigma_b + sigma_u)) / sigma_b

            if rho <= 0:
                raise ValueError(f"Invalid rho computed: {rho:.4f}")
        ################ Use minimization on d_equation_sq ##################
    
            # Check if the solutions are valid (all parameters positive), if not, try the next guess
            if sigma_u <= 0 or rho <= 0:
                print(f"Warning: Invalid solution for D={D_guess}: rho={rho}, sigma_u={sigma_u}, d={d_value}")
                ########### MINIMIZATION ROUTINE ###########
                print("Trying bounded minimization...")
                result = minimize_scalar(d_equation_sq_minus_sigma_u, method='bounded', bounds=(1e-3, 1e3))

                if result.success:
                    D_solution = result.x
                    d_value = D_solution
                    
                    # error trap for no solution found for d
                    if result.fun > 1e-6 or d_value is None or d_value <= 0 or np.isclose(d_value, 1e3, atol=1e-4):
                        print(f'⚠️ WARNING: No solution found for d')
                        raise ValueError("No valid solution found for parameter d")

                    # Compute sigma_u safely
                    cv_sq_target = cv_target ** 2
                    sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_value + sigma_b))/(-(d_value*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))

                    if sigma_u < 0:
                        raise ValueError(f"Minimization led to negative sigma_u: {sigma_u:.4f}")

                    # Compute rho from mean constraint
                    rho = (d_value * mu_target * (sigma_b + sigma_u)) / sigma_b

                    if rho <= 0:
                        raise ValueError(f"Invalid rho computed: {rho:.4f}")
                else:
                    raise ValueError(f"Minimize_scalar routine failed: {result.message}")
                ########### MINIMIZATION ROUTINE ###########
            else:
                print(f"Found valid solution for D={D_guess}: rho={rho}, sigma_u={sigma_u}, d={d_value}")
    except Exception as e:
        print(f"Warning: Error during solving for D={D_guess}: {str(e)}")

    # DEBUG 
    # print(f"Final d_value: {d_value}, rho: {rho}, sigma_u: {sigma_u}")
    if d_value is None or rho is None or sigma_u is None:
        print(f"One of the following values is None: final d_value: {d_value}, rho: {rho}, sigma_u: {sigma_u}")
        try:
            ########### MINIMIZATION ROUTINE ###########
            print("Trying bounded minimization...")
            result = minimize_scalar(d_equation_sq_minus_sigma_u, method='bounded', bounds=(1e-3, 1e3))
            
            #DEBUG: Print the result of minimization
            print(f'res.fun for minimize_scalar: {result.fun}')

            if result.success:
                D_solution = result.x
                d_value = D_solution
                
                # error trap for no solution found for d
                if result.fun > 1e-6 or d_value is None or d_value <= 0 or np.isclose(d_value, 1e3, atol=1e-4):
                    print(f'⚠️ WARNING: No solution found for d')
                    raise ValueError("No valid solution found for parameter d")

                # Compute sigma_u safely
                cv_sq_target = cv_target ** 2
                sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_value + sigma_b))/(-(d_value*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))

                if sigma_u < 0:
                    raise ValueError(f"Minimization led to negative sigma_u: {sigma_u:.4f}")

                # Compute rho from mean constraint
                rho = (d_value * mu_target * (sigma_b + sigma_u)) / sigma_b

                if rho <= 0:
                    raise ValueError(f"Invalid rho computed: {rho:.4f}")
            else:
                raise ValueError(f"Minimize_scalar routine failed: {result.message}")
            ########### MINIMIZATION ROUTINE ###########
        except Exception as e:
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
    
################## EXPERIMENTAL, THIS IS NOT WORKING VERY WELL #####################