#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

# Equation system
def equations(vars, sigma_u, mu_target, variance_target, autocorr_target):
    rho, sigma_b, d = vars

    # Mean
    mean_eqn = sigma_b * rho / (d * (sigma_b + sigma_u))

    # Variance
    variance_eqn = (
        sigma_b * rho / (d * (sigma_b + sigma_u)) +
        ((sigma_u * sigma_b) * rho**2 / (d * (sigma_b + sigma_u + d) * (sigma_u + sigma_b)**2))
    )

    # Autocorrelation
    ACmRNA_eq = sp.exp(-d * t) * (
        d * sp.exp((d - sigma_u - sigma_b) * t) * rho * sigma_u
        - (sigma_u + sigma_b) * (-d**2 + rho * sigma_u + (sigma_u + sigma_b) ** 2)
    ) / (
        (d - sigma_u - sigma_b) * (rho * sigma_u + d * (sigma_u + sigma_b) + (sigma_u + sigma_b) ** 2)
    )
    autocorr_eqn = ACmRNA_eq.subs(t, 1)

    eq1 = mean_eqn - mu_target
    eq2 = variance_eqn - variance_target
    eq3 = autocorr_eqn - autocorr_target

    return [float(eq1), float(eq2), float(eq3)]

# Jacobian
def jacobian(vars, sigma_u, mu_target, variance_target, autocorr_target):
    rho_val, sigma_b_val, d_val = vars
    rho_sym, sigma_b_sym, d_sym = sp.symbols('rho sigma_b d', real=True, positive=True)

    # Define expressions symbolically
    mean_eqn = sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u))
    variance_eqn = (
        sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u)) +
        ((sigma_u * sigma_b_sym) * rho_sym**2 / (d_sym * (sigma_b_sym + sigma_u + d_sym) * (sigma_u + sigma_b_sym)**2))
    )
    ACmRNA_eq = sp.exp(-d_sym * t) * (
        d_sym * sp.exp((d_sym - sigma_u - sigma_b_sym) * t) * rho_sym * sigma_u
        - (sigma_u + sigma_b_sym) * (-d_sym**2 + rho_sym * sigma_u + (sigma_u + sigma_b_sym)**2)
    ) / (
        (d_sym - sigma_u - sigma_b_sym) * (rho_sym * sigma_u + d_sym * (sigma_u + sigma_b_sym) + (sigma_u + sigma_b_sym)**2)
    )
    autocorr_eqn = ACmRNA_eq.subs(t, 1)

    # System
    eqs = [
        mean_eqn - mu_target,
        variance_eqn - variance_target,
        autocorr_eqn - autocorr_target
    ]
    
    J = sp.Matrix(eqs).jacobian([rho_sym, sigma_b_sym, d_sym])
    J_func = sp.lambdify((rho_sym, sigma_b_sym, d_sym), J, "numpy")

    return np.array(J_func(rho_val, sigma_b_val, d_val)).astype(np.float64)

# Solver wrapper
def find_parameters(sigma_u_val, mu_target, variance_target, autocorr_target, 
                    rho_range = (1, 1000), sigma_b_range = (0.1, 1000), d_range = (0.1, 5), num_guesses=1000):
    """
    Find parameters rho, sigma_b, and d that satisfy the equations for given targets.
    Args:
        sigma_u_val (float): Value of sigma_u.
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
    max_attempts = 10
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")
    
        rho_guesses = np.random.uniform(*rho_range, num_guesses)
        sigma_b_guesses = np.random.uniform(*sigma_b_range, num_guesses)
        d_guesses = np.random.uniform(*d_range, num_guesses)
        initial_guesses = list(zip(rho_guesses, sigma_b_guesses, d_guesses))
        
        found = False
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
                    print(f"Found solution: {solution}")
                    found = True
                    return solution
            except Exception:
                continue

    if not found:
        print(f"No suitable solution found, consider increasing num_gueses or expanding the parameter ranges.")