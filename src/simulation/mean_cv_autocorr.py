#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, minimize_scalar
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


    def d_equation_exp_sq_minus_sigma_u(D):
        '''This is f(d)^2 - sigma_u'''
        
        # Convert D to d using exponential transformation
        d_val = np.exp(D)
        
        # we use CV^2 to make calculations easier for the solvers
        cv_sq_target = cv_target ** 2
        
        sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_val + sigma_b))/(-(d_val*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))
        
        # Use np.exp consistently for numerical calculations
        return (-1/np.exp(1) + (-(d_val*mu_target*(1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target))) + (-1 + cv_sq_target*mu_target**2 + cv_sq_target**2*mu_target**2 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target)**2)*sigma_b)/(cv_sq_target*np.exp(autocorr_target*d_val)*mu_target*(-(d_val*mu_target) + (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b)))**2 - sigma_u

    # Use a more robust solver with multiple initial guesses in log space
    # Try values around log(1/autocorr_target)
    D_guess = np.log(1.0 / autocorr_target)
    d_value = None
    rho = None
    sigma_u = None

    try:
        # # Square the equation for solutions where f(d) = 0
        d_equation_exp_sq = lambda D: d_equation_exp(D) ** 2 
        result = fsolve(d_equation_exp_sq, D_guess, full_output=True)
        
        #DEBUG: Print the result of fsolve
        # print(f"fsolve result: {result}")
        if result[2] == 1:  # Check if converged
            D_solution = result[0][0]
            d_candidate = np.exp(D_solution)  # Transform back to d
            # d is guaranteed to be positive due to exp transform
            d_value = d_candidate
            #DEBUG: Print the found d value
            # print(f"Found d: {d_value:.4f} from D={D_solution:.4f}")

            # Step 2: Calculate the corresponding sigma and rho
            # need to use cv_sq_target so it's easier to solve
            cv_sq_target = cv_target ** 2
            sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_value + sigma_b))/(-(d_value*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))

            # Step 3: Calculate rho using the mean equation
            rho = (d_value * mu_target * (sigma_b + sigma_u)) / sigma_b

        ################ Use minimization on d_equation_exp_sq ##################
        # result = minimize_scalar(d_equation_exp_sq, method='bounded', bounds=(np.log(1e-3), np.log(1e3)))
        # if result.success:
        #     D_solution = result.x
        #     d_candidate = np.exp(D_solution)
        #     d_value = d_candidate

        #     # Compute sigma_u safely
        #     cv_sq_target = cv_target ** 2
        #     sigma_u = -(((-1 + cv_sq_target*mu_target)*sigma_b*(d_value + sigma_b))/(-(d_value*mu_target) + (-1 + cv_sq_target*mu_target)*sigma_b))

        #     if sigma_u < 0:
        #         raise ValueError(f"Minimization led to negative sigma_u: {sigma_u:.4f}")

        #     # Compute rho from mean constraint
        #     rho = (d_value * mu_target * (sigma_b + sigma_u)) / sigma_b

        #     if rho <= 0:
        #         raise ValueError(f"Invalid rho computed: {rho:.4f}")
        # else:
        #     raise ValueError(f"Minimize_scalar routine failed: {result.message}")
        
        
        ################ Use minimization on d_equation_exp_sq ##################
        
            # Check if the solutions are valid (all parameters positive), if not, try the next guess
            if sigma_u <= 0 or rho <= 0:
                print(f"Warning: Invalid solution for D={D_guess}: rho={rho}, sigma_u={sigma_u}, d={d_value}")
                ########### MINIMIZATION ROUTINE ###########
                print("Trying bounded minimization...")
                # Use bounded minimization for stability and positivity
                result = minimize_scalar(d_equation_exp_sq_minus_sigma_u, method='bounded', bounds=(np.log(1e-3), np.log(1e3)))

                if result.success:
                    D_solution = result.x
                    d_candidate = np.exp(D_solution)
                    d_value = d_candidate

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
    print(f"Final d_value: {d_value}, rho: {rho}, sigma_u: {sigma_u}")
    if d_value is None or rho is None or sigma_u is None:
        print(f"One of the following values is None: final d_value: {d_value}, rho: {rho}, sigma_u: {sigma_u}")
        try:
            ########### MINIMIZATION ROUTINE ###########
            print("Trying bounded minimization...")
            # Use bounded minimization for stability and positivity
            result = minimize_scalar(d_equation_exp_sq_minus_sigma_u, method='bounded', bounds=(np.log(1e-3), np.log(1e3)))
            #DEBUG: Print the result of minimization
            print(f'res.fun for minimize_scalar: {result.fun}')

            if result.success:
                D_solution = result.x
                d_candidate = np.exp(D_solution)
                d_value = d_candidate

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