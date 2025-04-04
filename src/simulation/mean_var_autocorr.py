#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

# # Equation system
# def equations(vars, sigma_u, mu_target, variance_target, autocorr_target):
#     rho, sigma_b, d = vars

#     # Mean
#     mean_eqn = sigma_b * rho / (d * (sigma_b + sigma_u))

#     # Variance
#     variance_eqn = (
#         sigma_b * rho / (d * (sigma_b + sigma_u)) +
#         ((sigma_u * sigma_b) * rho**2 / (d * (sigma_b + sigma_u + d) * (sigma_u + sigma_b)**2))
#     )

#     # Autocorrelation
#     ACmRNA_eq = sp.exp(-d * t) * (
#         d * sp.exp((d - sigma_u - sigma_b) * t) * rho * sigma_u
#         - (sigma_u + sigma_b) * (-d**2 + rho * sigma_u + (sigma_u + sigma_b) ** 2)
#     ) / (
#         (d - sigma_u - sigma_b) * (rho * sigma_u + d * (sigma_u + sigma_b) + (sigma_u + sigma_b) ** 2)
#     )
#     autocorr_eqn = ACmRNA_eq.subs(t, 1)

#     eq1 = mean_eqn - mu_target
#     eq2 = variance_eqn - variance_target
#     eq3 = autocorr_eqn - autocorr_target

#     return [float(eq1), float(eq2), float(eq3)]


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


# # Jacobian
# def jacobian(vars, sigma_u, mu_target, variance_target, autocorr_target):
#     rho_val, sigma_b_val, d_val = vars
#     rho_sym, sigma_b_sym, d_sym = sp.symbols('rho sigma_b d', real=True, positive=True)

#     # Define expressions symbolically
#     mean_eqn = sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u))
#     variance_eqn = (
#         sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u)) +
#         ((sigma_u * sigma_b_sym) * rho_sym**2 / (d_sym * (sigma_b_sym + sigma_u + d_sym) * (sigma_u + sigma_b_sym)**2))
#     )
#     ACmRNA_eq = sp.exp(-d_sym * t) * (
#         d_sym * sp.exp((d_sym - sigma_u - sigma_b_sym) * t) * rho_sym * sigma_u
#         - (sigma_u + sigma_b_sym) * (-d_sym**2 + rho_sym * sigma_u + (sigma_u + sigma_b_sym)**2)
#     ) / (
#         (d_sym - sigma_u - sigma_b_sym) * (rho_sym * sigma_u + d_sym * (sigma_u + sigma_b_sym) + (sigma_u + sigma_b_sym)**2)
#     )
#     autocorr_eqn = ACmRNA_eq.subs(t, 1)

#     # System
#     eqs = [
#         mean_eqn - mu_target,
#         variance_eqn - variance_target,
#         autocorr_eqn - autocorr_target
#     ]
    
#     J = sp.Matrix(eqs).jacobian([rho_sym, sigma_b_sym, d_sym])
#     J_func = sp.lambdify((rho_sym, sigma_b_sym, d_sym), J, "numpy")

#     return np.array(J_func(rho_val, sigma_b_val, d_val)).astype(np.float64)

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


# # Solver wrapper
# v1
# def find_parameters(sigma_u_val, mu_target=None, variance_target=None, autocorr_target=None,
#                     rho_range=(1, 1000), sigma_b_range=(0.1, 1000), d_range=(0.1, 5), num_guesses=1000):

#     """
#     Find parameters rho, sigma_b, and d that satisfy the equations for given targets.
#     Args:
#         sigma_u_val (float): Value of sigma_u.
#         mu_target (float): Target mean.
#         variance_target (float): Target variance.
#         autocorr_target (float): Target autocorrelation.
#         rho_range (tuple): Range for rho.
#         sigma_b_range (tuple): Range for sigma_b.
#         d_range (tuple): Range for d.
#         num_guesses (int): Number of random guesses to try.
#     Returns:
#         tuple: A tuple containing the found parameters (rho, sigma_b, d).
#     """
#     max_attempts = 10
#     for attempt in range(max_attempts):
#         print(f"Attempt {attempt + 1}/{max_attempts}")
    
#         rho_guesses = np.random.uniform(*rho_range, num_guesses)
#         sigma_b_guesses = np.random.uniform(*sigma_b_range, num_guesses)
#         d_guesses = np.random.uniform(*d_range, num_guesses)
#         initial_guesses = list(zip(rho_guesses, sigma_b_guesses, d_guesses))
        
#         found = False
#         for initial_guess in initial_guesses:
#             try:
#                 solution = fsolve(
#                     equations, initial_guess,
#                     args=(sigma_u_val, mu_target, variance_target, autocorr_target),
#                     fprime=jacobian,
#                     xtol=1e-8
#                 )
#                 residuals = equations(solution, sigma_u_val, mu_target, variance_target, autocorr_target)
#                 residuals = np.abs(residuals)

#                 if all(res < 1e-4 for res in residuals) and all(x > 0 for x in solution):
#                     print(f"Found solution: {solution}")
#                     found = True
#                     return solution
#             except Exception:
#                 continue

#     if not found:
#         print(f"No suitable solution found, consider increasing num_gueses or expanding the parameter ranges.")

# v2
def find_parameters(parameter_set, mu_target=None, variance_target=None, autocorr_target=None,
                    rho_range=(1, 1000), sigma_b_range=(0.1, 1000), d_range=(0.1, 5), num_guesses=1000):
    """
    Find parameters rho, sigma_b, and d that satisfy the equations for given target mean/variance/autocorrelation (at t=1).
    Args:
        parameter_set (dict): Dictionary with at least a 'sigma_u' key.
        mu_target (float): Target mean.
        variance_target (float): Target variance.
        autocorr_target (float): Target autocorrelation.
        rho_range (tuple): Range for rho.
        sigma_b_range (tuple): Range for sigma_b.
        d_range (tuple): Range for d.
        num_guesses (int): Number of random guesses to try.
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
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")
        
        if "rho" in to_solve:
            rho_guesses = np.random.uniform(*rho_range, num_guesses)
        else:
            rho_guesses = [fixed["rho"]] * num_guesses
        if "sigma_b" in to_solve:
            sigma_b_guesses = np.random.uniform(*sigma_b_range, num_guesses)
        else:
            sigma_b_guesses = [fixed["sigma_b"]] * num_guesses
        if "d" in to_solve:
            d_guesses = np.random.uniform(*d_range, num_guesses)
        else:
            d_guesses = [fixed["d"]] * num_guesses
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
                # print(f"Solution: {solution}")

                residuals = equations(solution, sigma_u_val, mu_target, variance_target, autocorr_target)
                residuals = np.abs(residuals)
                # print(f"Residuals: {residuals}")

                if all(res < 1e-4 for res in residuals) and all(x > 0 for x in solution):
                    # print(f"Found solution: {solution}")
                    return solution
            except Exception:
                continue

    raise ValueError("No suitable solution found after multiple attempts. Try increasing num_guesses or widening the ranges.")

# v3
# def find_parameters(parameter_set, mu_target=None, variance_target=None, autocorr_target=None,
#                     rho_range=(1, 1000), sigma_b_range=(0.001, 10), d_range=(0.01, 5), num_guesses=1000):
#     """
#     Solve for one or more of [rho, sigma_b, d], based on targets provided.
#     Values not being solved for must be present in parameter_set.
#     """
#     sigma_u_val = parameter_set.get("sigma_u")
#     if sigma_u_val is None:
#         raise ValueError("parameter_set must include a 'sigma_u' key.")

#     to_solve = []
#     fixed = {}

#     if mu_target is not None:
#         to_solve.append("rho")
#     else:
#         fixed["rho"] = parameter_set.get("rho")
#         if fixed["rho"] is None:
#             raise ValueError("rho must be in parameter_set if mu_target is not specified.")

#     if variance_target is not None:
#         to_solve.append("sigma_b")
#     else:
#         fixed["sigma_b"] = parameter_set.get("sigma_b")
#         if fixed["sigma_b"] is None:
#             raise ValueError("sigma_b must be in parameter_set if variance_target is not specified.")

#     if autocorr_target is not None:
#         to_solve.append("d")
#     else:
#         fixed["d"] = parameter_set.get("d")
#         if fixed["d"] is None:
#             raise ValueError("d must be in parameter_set if autocorr_target is not specified.")

#     if all(target is None for target in [mu_target, variance_target, autocorr_target]):
#         raise ValueError("At least one of mu_target, variance_target, or autocorr_target must be specified.")
    
#     # Map indices
#     var_order = ["rho", "sigma_b", "d"]

#     # Retry loop
#     max_attempts = 10
#     for attempt in range(max_attempts):
#         print(f"Attempt {attempt + 1}/{max_attempts}")

#         # Sample random guesses for the variables we're solving for
#         guesses = {
#             "rho": np.random.uniform(*rho_range, num_guesses) if "rho" in to_solve else [fixed["rho"]] * num_guesses,
#             "sigma_b": np.random.uniform(*sigma_b_range, num_guesses) if "sigma_b" in to_solve else [fixed["sigma_b"]] * num_guesses,
#             "d": np.random.uniform(*d_range, num_guesses) if "d" in to_solve else [fixed["d"]] * num_guesses,
#         }

#         for i in range(num_guesses):
#             initial_guess = [guesses[v][i] for v in var_order]

#             try:
#                 solution = fsolve(
#                     equations, initial_guess,
#                     args=(sigma_u_val, mu_target, variance_target, autocorr_target),
#                     fprime=jacobian,
#                     xtol=1e-8
#                 )
#                 residuals = equations(solution, sigma_u_val, mu_target, variance_target, autocorr_target)
#                 residuals = np.abs(residuals)

#                 if all(res < 1e-4 for res in residuals) and all(x > 0 for x in solution):
#                     result = dict(zip(var_order, solution))
#                     return tuple(result[v] for v in to_solve)
#             except Exception:
#                 continue

#     raise ValueError("No suitable solution found after multiple attempts.")
