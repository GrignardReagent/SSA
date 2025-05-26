#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve
from utils.fano_factor import calculate_fano_factor,calculate_fano_factor_from_params
from utils.cv import calculate_cv

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

def equations(vars, sigma_u, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None):
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
        
    # Coefficient of Variation (CV)
    if cv_target is not None:
        cv_eqn = (sp.sqrt((
            sigma_b * rho / (d * (sigma_b + sigma_u)) +
            ((sigma_u * sigma_b) * rho**2 / (d * (sigma_b + sigma_u + d) * (sigma_u + sigma_b)**2))
        )) / mu_target) - cv_target
        eqs.append(float(cv_eqn))
    
    # Fano Factor
    if fano_factor_target is not None:
        fano_factor_eqn = 1 + (rho * sigma_u) / ((sigma_b + sigma_u) * (sigma_b + d + sigma_u)) - fano_factor_target
        eqs.append(float(fano_factor_eqn))

    return eqs

def jacobian(vars, sigma_u, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None):
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

    # Coefficient of Variation (CV)
    if cv_target is not None:
        cv_eqn = (sp.sqrt((
            sigma_b_sym * rho_sym / (d_sym * (sigma_b_sym + sigma_u)) +
            ((sigma_u * sigma_b_sym) * rho_sym**2 / (d_sym * (sigma_b_sym + sigma_u + d_sym) * (sigma_u + sigma_b_sym)**2))
        )) / mu_target) - cv_target
        eqs.append(cv_eqn)

    # Fano Factor
    if fano_factor_target is not None:
        fano_factor_eqn = 1 + (rho_sym * sigma_u) / ((sigma_b_sym + sigma_u) * (sigma_b_sym + d_sym + sigma_u)) - fano_factor_target
        eqs.append(fano_factor_eqn)

    J = sp.Matrix(eqs).jacobian([rho_sym, sigma_b_sym, d_sym])
    J_func = sp.lambdify((rho_sym, sigma_b_sym, d_sym), J, "numpy")
    return np.array(J_func(rho_val, sigma_b_val, d_val)).astype(np.float64)

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

def find_parameters(parameter_set, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None,
                    rho_range=(1, 1000), sigma_b_range=(0.1, 1000), d_range=(0.1, 5), num_guesses=1000, 
                    check_biological=True, max_fano_factor=20, max_cv=5.0):
    """
    Find parameters rho, sigma_b, and d that satisfy the equations for given target statistical properties.
    
    You can specify at most 3 of these statistical properties: mean, variance, autocorrelation, CV, and Fano factor.
    The function will solve for the corresponding telegraph model parameters (rho, sigma_b, d).
    
    Note: You cannot specify both variance and CV (they are related by CV = sqrt(variance)/mean).
    Note: You cannot specify both autocorrelation and Fano factor (would over-constrain the system).
    
    Statistical properties map to model parameters as follows:
    - Mean (mu_target) → rho
    - Variance (variance_target) or CV (cv_target) → sigma_b
    - Autocorrelation (autocorr_target) or Fano factor (fano_factor_target) → d
    
    Args:
        parameter_set (dict): Dictionary with at least a 'sigma_u' key.
        mu_target (float, optional): Target mean.
        variance_target (float, optional): Target variance.
        autocorr_target (float, optional): Target autocorrelation at t=1.
        cv_target (float, optional): Target coefficient of variation.
        fano_factor_target (float, optional): Target Fano factor.
        rho_range (tuple): Range for rho initial guesses.
        sigma_b_range (tuple): Range for sigma_b initial guesses.
        d_range (tuple): Range for d initial guesses.
        num_guesses (int): Number of random initial guesses to try.
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

    # Count how many statistical properties we're trying to fix
    # We can only solve for 3 parameters (rho, sigma_b, d), so we can only fix 3 properties
    target_count = sum(1 for target in [mu_target, variance_target, autocorr_target, cv_target, fano_factor_target] if target is not None)
    if target_count > 3:
        raise ValueError(f"Cannot fix more than 3 statistical properties (you specified {target_count})")
    
    # Determine which parameters to solve for based on target values
    # Mean -> rho
    if mu_target is not None:
        to_solve.append("rho")
    else:
        fixed["rho"] = parameter_set.get("rho")
        if fixed["rho"] is None:
            raise ValueError("rho must be in parameter_set if mu_target is not specified.")

    # Variance or CV -> sigma_b
    # Cannot specify both variance and CV (would be redundant/potentially conflicting)
    if variance_target is not None and cv_target is not None:
        raise ValueError("Cannot specify both variance_target and cv_target (they are related by CV = sqrt(variance)/mean)")
    
    if variance_target is not None or cv_target is not None:
        to_solve.append("sigma_b")
    else:
        fixed["sigma_b"] = parameter_set.get("sigma_b")
        if fixed["sigma_b"] is None:
            raise ValueError("sigma_b must be in parameter_set if neither variance_target nor cv_target is specified.")
    
    # Autocorrelation or Fano factor -> d
    # Cannot specify both autocorrelation and Fano factor
    if autocorr_target is not None and fano_factor_target is not None:
        raise ValueError("Cannot specify both autocorr_target and fano_factor_target (would over-constrain the system)")
    
    if autocorr_target is not None or fano_factor_target is not None:
        to_solve.append("d")
    else:
        fixed["d"] = parameter_set.get("d")
        if fixed["d"] is None:
            raise ValueError("d must be in parameter_set if neither autocorr_target nor fano_factor_target is specified.")

    if all(target is None for target in [mu_target, variance_target, autocorr_target, cv_target, fano_factor_target]):
        raise ValueError("At least one of mu_target, variance_target, autocorr_target, cv_target, or fano_factor_target must be specified.")

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
                    args=(sigma_u_val, mu_target, variance_target, autocorr_target, cv_target, fano_factor_target),
                    fprime=jacobian,
                    xtol=1e-8
                )

                residuals = equations(solution, sigma_u_val, mu_target, variance_target, autocorr_target, cv_target, fano_factor_target)
                residuals = np.abs(residuals)

                if all(res < 1e-4 for res in residuals) and all(x > 0 for x in solution):
                    # Check biological appropriateness if required
                    if check_biological:
                        is_appropriate = True  # Default to True
                        if variance_target is not None and mu_target is not None:
                            is_appropriate = check_biological_appropriateness(variance_target, mu_target)
                        elif cv_target is not None and mu_target is not None:
                            # Calculate the implied variance based on CV and mean
                            implied_variance = (cv_target * mu_target)**2
                            is_appropriate = check_biological_appropriateness(implied_variance, mu_target)
                        
                        # Only return solution if it's biologically appropriate or if we don't care
                        if is_appropriate:
                            return solution
                    else:
                        # If biological appropriateness checking is disabled, just return the solution
                        return solution
            except Exception:
                continue

    raise ValueError("No suitable solution found after multiple attempts. Try increasing num_guesses or widening the ranges.")
    
def find_biological_variance_mean(desired_fano_factor, desired_cv, max_fano_factor=20, max_cv=5.0):
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