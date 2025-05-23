#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve
from utils.fano_factor import calculate_fano_factor
from utils.cv import calculate_cv

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

def equations(vars, sigma_u, mu_target=None, variance_target=None, autocorr_target=None):
    rho, sigma_b, d = vars

    eqs = []

    # Mean
    if mu_target is not None:
        mean_eqn = sigma_b * rho / (d * (sigma_b + sigma_u)) - mu_target
        eqs.append(float(mean_eqn))

    # Variance
    if variance_target is not None:
        variance_eqn = (
            sigma_b * rho / (d * (sigma_b + sigma_u)) +
            ((sigma_u * sigma_b) * rho**2 / (d * (sigma_b + sigma_u + d) * (sigma_u + sigma_b)**2))
        ) - variance_target
        eqs.append(float(variance_eqn))

    # Autocorrelation
    if autocorr_target is not None:
        ACmRNA_eq = sp.exp(-d * t) * (
            d * sp.exp((d - sigma_u - sigma_b) * t) * rho * sigma_u
            - (sigma_u + sigma_b) * (-d**2 + rho * sigma_u + (sigma_u + sigma_b) ** 2)
        ) / (
            (d - sigma_u - sigma_b) * (rho * sigma_u + d * (sigma_u + sigma_b) + (sigma_u + sigma_b) ** 2)
        )
        autocorr_eqn = ACmRNA_eq.subs(t, 1) - autocorr_target
        eqs.append(float(autocorr_eqn))

    return eqs

def jacobian(vars, sigma_u, mu_target=None, variance_target=None, autocorr_target=None):
    rho_val, sigma_b_val, d_val = vars
    rho_sym, sigma_b_sym, d_sym = sp.symbols('rho sigma_b d', real=True, positive=True)

    eqs = []

    # Mean
    if mu_target is not None:
        mean_eqn = sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u)) - mu_target
        eqs.append(mean_eqn)

    # Variance
    if variance_target is not None:
        variance_eqn = (
            sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u)) +
            ((sigma_u * sigma_b_sym) * rho_sym**2 / (d_sym * (sigma_b_sym + sigma_u + d_sym) * (sigma_u + sigma_b_sym)**2))
        ) - variance_target
        eqs.append(variance_eqn)

    # Autocorrelation
    if autocorr_target is not None:
        ACmRNA_eq = sp.exp(-d_sym * t) * (
            d_sym * sp.exp((d_sym - sigma_u - sigma_b_sym) * t) * rho_sym * sigma_u
            - (sigma_u + sigma_b_sym) * (-d_sym**2 + rho_sym * sigma_u + (sigma_u + sigma_b_sym)**2)
        ) / (
            (d_sym - sigma_u - sigma_b_sym) * (rho_sym * sigma_u + d_sym * (sigma_u + sigma_b_sym) + (sigma_u + sigma_b_sym)**2)
        )
        autocorr_eqn = ACmRNA_eq.subs(t, 1) - autocorr_target
        eqs.append(autocorr_eqn)

    J = sp.Matrix(eqs).jacobian([rho_sym, sigma_b_sym, d_sym])
    J_func = sp.lambdify((rho_sym, sigma_b_sym, d_sym), J, "numpy")
    return np.array(J_func(rho_val, sigma_b_val, d_val)).astype(np.float64)


def find_parameters(parameter_set, mu_target=None, variance_target=None, autocorr_target=None,
                    rho_range=(1, 1000), sigma_b_range=(0.1, 1000), d_range=(0.1, 5), num_guesses=1000, 
                    check_biological=True, max_fano_factor=20, max_cv=5.0):
    """
    Find parameters rho, sigma_b, and d that satisfy the equations for given target mean/variance/autocorrelation (at t=1).
    Args:s
        parameter_set (dict): Dictionary with at least a 'sigma_u' key.
        mu_target (float): Target mean.
        variance_target (float): Target variance.
        autocorr_target (float): Target autocorrelation.
        rho_range (tuple): Range for rho.
        sigma_b_range (tuple): Range for sigma_b.
        d_range (tuple): Range for d.
        num_guesses (int): Number of random guesses to try.
        check_biological (bool): Whether to check if the solution is biologically appropriate.
        max_fano_factor (float): Maximum allowed Fano factor for a biologically appropriate solution.
        max_cv (float): Maximum allowed coefficient of variation for a biologically appropriate solution.
    Returns:
        tuple: A tuple containing the found parameters (rho, sigma_b, d).
    """
    
    sigma_u_val = parameter_set.get("sigma_u")
    if sigma_u_val is None:
        raise ValueError("parameter_set must include a 'sigma_u' key.")

    to_solve = []
    fixed = {}

    if mu_target is not None:
        to_solve.append("rho")
    else:
        fixed["rho"] = parameter_set.get("rho")
        if fixed["rho"] is None:
            raise ValueError("rho must be in parameter_set if mu_target is not specified.")

    if variance_target is not None:
        to_solve.append("sigma_b")
    else:
        fixed["sigma_b"] = parameter_set.get("sigma_b")
        if fixed["sigma_b"] is None:
            raise ValueError("sigma_b must be in parameter_set if variance_target is not specified.")

    if autocorr_target is not None:
        to_solve.append("d")
    else:
        fixed["d"] = parameter_set.get("d")
        if fixed["d"] is None:
            raise ValueError("d must be in parameter_set if autocorr_target is not specified.")

    if all(target is None for target in [mu_target, variance_target, autocorr_target]):
        raise ValueError("At least one of mu_target, variance_target, or autocorr_target must be specified.")

    max_attempts = 10
    max_factor = 2.0
    max_guesses = 2000
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")

        # only expand search space if this is not the first attempt
        if attempt == 0: 
            current_rho_range = rho_range
            current_sigma_b_range = sigma_b_range
            current_d_range = d_range
            current_num_guesses = num_guesses
        else:
            # Expand ranges by 30% each failed attempt
            factor = min(1 + attempt * 0.3, max_factor)
            current_rho_range = (rho_range[0], rho_range[1] * factor)
            current_sigma_b_range = (sigma_b_range[0], sigma_b_range[1] * factor)
            current_d_range = (d_range[0], d_range[1] * factor)
            current_num_guesses = min(int(num_guesses * (1 + attempt * 0.2)), max_guesses)

        # Generate guesses
        if "rho" in to_solve:
            rho_guesses = np.random.uniform(*current_rho_range, current_num_guesses)
        else:
            rho_guesses = [fixed["rho"]] * current_num_guesses
        if "sigma_b" in to_solve:
            sigma_b_guesses = np.random.uniform(*current_sigma_b_range, current_num_guesses)
        else:
            sigma_b_guesses = [fixed["sigma_b"]] * current_num_guesses
        if "d" in to_solve:
            d_guesses = np.random.uniform(*current_d_range, current_num_guesses)
        else:
            d_guesses = [fixed["d"]] * current_num_guesses

        initial_guesses = list(zip(rho_guesses, sigma_b_guesses, d_guesses))

        # print(f"Initial guesses: {initial_guesses}")

        for initial_guess in initial_guesses:
            try:
                solution = fsolve(
                    equations, initial_guess,
                    args=(sigma_u_val, mu_target, variance_target, autocorr_target),
                    fprime=jacobian,
                    xtol=1e-8
                )

                residuals = equations(solution, sigma_u_val, mu_target, variance_target, autocorr_target)
                residuals = np.abs(residuals)

                if all(res < 1e-4 for res in residuals) and all(x > 0 for x in solution):
                    # Check biological appropriateness if required
                    if check_biological:
                        rho_val, sigma_b_val, d_val = solution
                        
                        # Check Fano factor
                        fano_factor = calculate_fano_factor(rho_val, sigma_b_val, d_val, sigma_u_val)
                        
                        # Check coefficient of variation 
                        cv = calculate_cv(variance_target, mu_target)
                        if cv >= max_cv:
                            print(f"WARNING: Solution found but CV {cv:.2f} > {max_cv}, consider changing the target variance or mean.")
                        if fano_factor >= max_fano_factor:
                            print(f"WARNING: Solution found but Fano factor {fano_factor:.2f} > {max_fano_factor}, consider changing the target parameters.")
                        else:
                            print(f"Found biologically appropriate solution with Fano factor: {fano_factor:.2f}")
                        
                    return solution
            except Exception:
                continue

    raise ValueError("No suitable solution found after multiple attempts. Try increasing num_guesses or widening the ranges.")
