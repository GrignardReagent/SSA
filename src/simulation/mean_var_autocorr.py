#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, check_grad
from utils.fano_factor import calculate_fano_factor,calculate_fano_factor_from_params
from utils.cv import calculate_cv

# Define symbols globally for reuse
rho, sigma_b, d, sigma_u, t, mu, sigma_sq, ac = sp.symbols('rho sigma_b d sigma_u t mu sigma_sq ac', real=True, positive=True)
init_printing(use_unicode=True)

# Precompute symbols and derivative functions for a fast Jacobian used when
# mean, CV, and autocorrelation targets are provided simultaneously.
mu_t, cv_t, ac_t = sp.symbols('mu_t cv_t ac_t')

# Mean equation derivatives
mean_drho_sym = sigma_b / (d * (sigma_b + sigma_u))
mean_dsigmab_sym = rho * sigma_u / (d * (sigma_b + sigma_u) ** 2)
mean_dd_sym = -sigma_b * rho / (d ** 2 * (sigma_b + sigma_u))

# CV equation and derivatives
cv_expr_sym = sp.sqrt(
    sigma_b * rho / (d * (sigma_b + sigma_u))
    + (sigma_u * sigma_b) * rho ** 2 / (d * (sigma_b + sigma_u + d) * (sigma_b + sigma_u) ** 2)
) / mu_t - cv_t
cv_drho_sym = sp.diff(cv_expr_sym, rho)
cv_dsigmab_sym = sp.diff(cv_expr_sym, sigma_b)
cv_dd_sym = sp.diff(cv_expr_sym, d)

# Autocorrelation equation derivatives evaluated at t=1
ACmRNA_eq = sp.exp(-d * t) * (
    d * sp.exp((d - sigma_u - sigma_b) * t) * rho * sigma_u
    - (sigma_u + sigma_b) * (-d**2 + rho * sigma_u + (sigma_u + sigma_b) ** 2)
) / (
    (d - sigma_u - sigma_b) * (rho * sigma_u + d * (sigma_u + sigma_b) + (sigma_u + sigma_b) ** 2)
)
autocorr_expr_sym = ACmRNA_eq.subs(t, 1) - ac_t
ac_drho_sym = sp.diff(autocorr_expr_sym, rho)
ac_dsigmab_sym = sp.diff(autocorr_expr_sym, sigma_b)
ac_dd_sym = sp.simplify(sp.diff(autocorr_expr_sym, d))

# Lambdify derivative expressions for numerical evaluation
_MEAN_DERIVS = (
    sp.lambdify((rho, sigma_b, d, sigma_u), mean_drho_sym, 'numpy'),
    sp.lambdify((rho, sigma_b, d, sigma_u), mean_dsigmab_sym, 'numpy'),
    sp.lambdify((rho, sigma_b, d, sigma_u), mean_dd_sym, 'numpy'),
)
_CV_DERIVS = (
    sp.lambdify((rho, sigma_b, d, sigma_u, mu_t), cv_drho_sym, 'numpy'),
    sp.lambdify((rho, sigma_b, d, sigma_u, mu_t), cv_dsigmab_sym, 'numpy'),
    sp.lambdify((rho, sigma_b, d, sigma_u, mu_t), cv_dd_sym, 'numpy'),
)
_AC_DERIVS = (
    sp.lambdify((rho, sigma_b, d, sigma_u, ac_t), ac_drho_sym, 'numpy'),
    sp.lambdify((rho, sigma_b, d, sigma_u, ac_t), ac_dsigmab_sym, 'numpy'),
    sp.lambdify((rho, sigma_b, d, sigma_u, ac_t), ac_dd_sym, 'numpy'),
)

def equations(vars, sigma_b, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None):
    '''
    Define the equations for the telegraph model based on the given parameters.
    Args:
        vars (tuple): Tuple of variables (rho, sigma_u, d).
        sigma_b (float): Value of sigma_b.
        mu_target (float, optional): Target mean.
        variance_target (float, optional): Target variance.
        autocorr_target (float, optional): Target autocorrelation at t=1.
        cv_target (float, optional): Target coefficient of variation.
        fano_factor_target (float, optional): Target Fano factor.
    Returns:
        list: List of equations evaluated at the given variables.
    '''

    rho, sigma_u, d = vars

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
        # Evaluate autocorrelation at t=1
        autocorr_eqn = ACmRNA_eq.subs(t, 1) - autocorr_target
        eqs.append(float(autocorr_eqn))
        
    # Coefficient of Variation (CV)
    if cv_target is not None:
        cv_sq_target = cv_target ** 2
        cv_sq_eqn = - cv_sq_target + (d**2 * (sigma_b + sigma_u)**2 * (
            (rho * sigma_b) / (d * (sigma_b + sigma_u)) +
            (rho**2 * sigma_b * sigma_u) / (d * (sigma_b + sigma_u)**2 * (d + sigma_b + sigma_u))
        )) / (rho**2 * sigma_b**2)

        # cv_eqn = (sp.sqrt((
        #     sigma_b * rho / (d * (sigma_b + sigma_u)) +
        #     ((sigma_u * sigma_b) * rho**2 / (d * (sigma_b + sigma_u + d) * (sigma_u + sigma_b)**2))
        # )) / mu_target) - cv_target
        
        eqs.append(float(cv_sq_eqn))
    
    # Fano Factor
    if fano_factor_target is not None:
        fano_factor_eqn = 1 + (rho * sigma_u) / ((sigma_b + sigma_u) * (sigma_b + d + sigma_u)) - fano_factor_target
        eqs.append(float(fano_factor_eqn))

    return eqs

# # TODO: we fixed sigma_b in the equation function, now we need to fix it in the Jacobian
# def jacobian(vars, sigma_b, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None):
#     ''' 
#     Calculate the Jacobian matrix for the equations defined above.
#     When mean, CV and autocorrelation targets are simultaneously supplied (with
#     neither variance nor Fano factor fixed), a custom Jacobian based on analytic
#     derivatives is returned for efficiency.
#     Args:
#         vars (tuple): Tuple of variables (rho, sigma_u, d).
#         sigma_b (float): Value of sigma_b.
#         mu_target (float, optional): Target mean.
#         variance_target (float, optional): Target variance.
#         autocorr_target (float, optional): Target autocorrelation at t=1.
#         cv_target (float, optional): Target coefficient of variation.
#         fano_factor_target (float, optional): Target Fano factor.
#     Returns:
#         np.ndarray: Jacobian matrix evaluated at the given variables.
#     '''

#     rho_val, sigma_u_val, d_val = vars

#     # Optimization: when mean, CV and autocorrelation targets are provided (and
#     # variance and Fano factor are not), use pre-computed analytic derivatives
#     # rather than relying on SymPy's jacobian each call.
#     if (
#         mu_target is not None
#         and cv_target is not None
#         and autocorr_target is not None
#         and variance_target is None
#         and fano_factor_target is None
#     ):
#         J = np.zeros((3, 3), dtype=float)
#         # Mean equation derivatives
#         J[0, 0] = _MEAN_DERIVS[0](rho_val, sigma_u_val, d_val, sigma_u)
#         J[0, 1] = _MEAN_DERIVS[1](rho_val, sigma_u_val, d_val, sigma_u)
#         J[0, 2] = _MEAN_DERIVS[2](rho_val, sigma_u_val, d_val, sigma_u)

#         # CV equation derivatives
#         J[1, 0] = _CV_DERIVS[0](rho_val, sigma_u_val, d_val, sigma_b, mu_target)
#         J[1, 1] = _CV_DERIVS[1](rho_val, sigma_u_val, d_val, sigma_b, mu_target)
#         J[1, 2] = _CV_DERIVS[2](rho_val, sigma_u_val, d_val, sigma_b, mu_target)

#         # Autocorrelation equation derivatives (explicit d derivative used)
#         J[2, 0] = _AC_DERIVS[0](rho_val, sigma_u_val, d_val, sigma_u, autocorr_target)
#         J[2, 1] = _AC_DERIVS[1](rho_val, sigma_u_val, d_val, sigma_u, autocorr_target)
#         J[2, 2] = _AC_DERIVS[2](rho_val, sigma_u_val, d_val, sigma_u, autocorr_target)
#         return J

#     rho_sym, sigma_u_sym, d_sym = sp.symbols('rho sigma_u d', real=True, positive=True)


#     eqs = []
#     # Mean
#     if mu_target is not None:
#         mean_eqn = sigma_b * rho_sym / (d_sym * (sigma_u_sym + sigma_b)) - mu_target
#         eqs.append(mean_eqn)
#     # Variance
#     if variance_target is not None:
#         variance_eqn = (
#             sigma_b * rho_sym / (d_sym * (sigma_b + sigma_u_sym)) +
#             ((sigma_u_sym * sigma_b) * rho_sym**2 / (d_sym * (sigma_b + sigma_u_sym + d_sym) * (sigma_u_sym + sigma_b)**2))
#         ) - variance_target
#         eqs.append(variance_eqn)
#     # Autocorrelation
#     if autocorr_target is not None:
#         ACmRNA_eq = sp.exp(-d_sym * t) * (
#             d_sym * sp.exp((d_sym - sigma_u_sym - sigma_b) * t) * rho_sym * sigma_u_sym
#             - (sigma_u_sym + sigma_b) * (-d_sym**2 + rho_sym * sigma_u_sym + (sigma_u_sym + sigma_b)**2)
#         ) / (
#             (d_sym - sigma_u_sym - sigma_b) * (rho_sym * sigma_u_sym + d_sym * (sigma_u_sym + sigma_b) + (sigma_u_sym + sigma_b)**2)
#         )
#         autocorr_eqn = ACmRNA_eq.subs(t, 1) - autocorr_target
#         eqs.append(autocorr_eqn)
#     # Coefficient of Variation (CV)
#     if cv_target is not None:
#         cv_eqn = (sp.sqrt((
#             sigma_b * rho_sym / (d_sym * (sigma_b + sigma_u_sym)) +
#             ((sigma_u_sym * sigma_b) * rho_sym**2 / (d_sym * (sigma_b + sigma_u_sym + d_sym) * (sigma_u_sym + sigma_b)**2))
#         )) / mu_target) - cv_target
#         eqs.append(cv_eqn)
        
#     # Fano Factor
#     if fano_factor_target is not None:
#         fano_factor_eqn = 1 + (rho_sym * sigma_u_sym) / ((sigma_b + sigma_u_sym) * (sigma_b + d_sym + sigma_u_sym)) - fano_factor_target
#         eqs.append(fano_factor_eqn)
#     # Calculate the Jacobian matrix
#     # Note: We use sympy's jacobian function to compute the Jacobian matrix
#     # and then convert it to a numpy array for numerical evaluation
#     J = sp.Matrix(eqs).jacobian([rho_sym, sigma_u_sym, d_sym])
#     # Convert the Jacobian to a numerical function
#     J_func = sp.lambdify((rho_sym, sigma_u_sym, d_sym), J, "numpy")
#     return np.array(J_func(rho_val, sigma_u_val, d_val)).astype(np.float64)

#TODO: Version 1
def jacobian(vars, sigma_b, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None):
    ''' 
    Calculate the Jacobian matrix for the equations defined above.
    Args:
        vars (tuple): Tuple of variables (rho, sigma_u, d).
        sigma_b (float): Value of sigma_b.
        mu_target (float, optional): Target mean.
        variance_target (float, optional): Target variance.
        autocorr_target (float, optional): Target autocorrelation at t=1.
        cv_target (float, optional): Target coefficient of variation.
        fano_factor_target (float, optional): Target Fano factor.
    Returns:
        np.ndarray: Jacobian matrix evaluated at the given variables.
    '''

    rho_val, sigma_u_val, d_val = vars
    rho_sym, sigma_u_sym, d_sym = sp.symbols('rho sigma_u d', real=True, positive=True)


    eqs = []
    # Mean
    if mu_target is not None:
        mean_eqn = sigma_b * rho_sym / (d_sym * (sigma_b + sigma_u_sym)) - mu_target
        eqs.append(mean_eqn)

    # Variance
    if variance_target is not None:
        variance_eqn = (
            sigma_b * rho_sym / (d_sym * (sigma_b + sigma_u_sym)) +
            ((sigma_u_sym * sigma_b) * rho_sym**2 / (d_sym * (sigma_b + sigma_u_sym + d_sym) * (sigma_u_sym + sigma_b)**2))
        ) - variance_target
        eqs.append(variance_eqn)

    # Autocorrelation
    if autocorr_target is not None:
        ACmRNA_eq = sp.exp(-d_sym * t) * (
            d_sym * sp.exp((d_sym - sigma_u_sym - sigma_b) * t) * rho_sym * sigma_u_sym
            - (sigma_u_sym + sigma_b) * (-d_sym**2 + rho_sym * sigma_u_sym + (sigma_u_sym + sigma_b)**2)
        ) / (
            (d_sym - sigma_u_sym - sigma_b) * (rho_sym * sigma_u_sym + d_sym * (sigma_u_sym + sigma_b) + (sigma_u_sym + sigma_b)**2)
        )
        autocorr_eqn = ACmRNA_eq.subs(t, 1) - autocorr_target
        eqs.append(autocorr_eqn)

    # Coefficient of Variation (CV)
    if cv_target is not None:
        cv_sq_target = cv_target ** 2
        cv_sq_eqn = - cv_sq_target + (d_sym**2 * (sigma_b + sigma_u_sym)**2 * (
            (rho_sym * sigma_b) / (d_sym * (sigma_b + sigma_u_sym)) +
            (rho_sym**2 * sigma_b * sigma_u_sym) / (d_sym * (sigma_b + sigma_u_sym)**2 * (d_sym + sigma_b + sigma_u_sym))
        )) / (rho_sym**2 * sigma_b**2)

        # cv_eqn = (sp.sqrt((
        #     sigma_b * rho_sym / (d_sym * (sigma_b + sigma_u_sym)) +
        #     ((sigma_u_sym * sigma_b) * rho_sym**2 / (d_sym * (sigma_b + sigma_u_sym + d_sym) * (sigma_u_sym + sigma_b)**2))
        # )) / mu_target) - cv_target
        
        eqs.append(cv_sq_eqn)

    # Fano Factor
    if fano_factor_target is not None:
        fano_factor_eqn = 1 + (rho_sym * sigma_u_sym) / ((sigma_b + sigma_u_sym) * (sigma_b + d_sym + sigma_u_sym)) - fano_factor_target
        eqs.append(fano_factor_eqn)

    # Calculate the Jacobian matrix
    # Note: We use sympy's jacobian function to compute the Jacobian matrix
    # and then convert it to a numpy array for numerical evaluation
    J = sp.Matrix(eqs).jacobian([rho_sym, sigma_u_sym, d_sym])
    # Convert the Jacobian to a numerical function
    J_func = sp.lambdify((rho_sym, sigma_u_sym, d_sym), J, "numpy")
    return np.array(J_func(rho_val, sigma_u_val, d_val)).astype(np.float64)

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

def validate_jacobian(sigma_b_val, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None):
    """
    Validate the Jacobian function using scipy.optimize.check_grad.
    
    Args:
        sigma_b_val (float): Value of sigma_b.
        mu_target, variance_target, autocorr_target, cv_target, fano_factor_target: Target values.
        
    Returns:
        float: The difference between numerical and analytical Jacobians.
    """
    # Define a scalar objective function (sum of squared residuals)
    def objective(vars):
        residuals = equations(vars, sigma_b_val, mu_target, variance_target, 
                             autocorr_target, cv_target, fano_factor_target)
        return np.sum(np.square(residuals))
    
    # Define gradient of the objective function using our analytical Jacobian
    def gradient(vars):
        J = jacobian(vars, sigma_b_val, mu_target, variance_target, 
                    autocorr_target, cv_target, fano_factor_target)
        residuals = equations(vars, sigma_b_val, mu_target, variance_target, 
                             autocorr_target, cv_target, fano_factor_target)
        # Gradient of sum of squared residuals is 2 * J^T * residuals
        return 2 * np.dot(J.T, residuals)
    
    # Test point - reasonable values for rho, sigma_u, d
    test_point = np.array([50.0, 0.5, 1.0])
    
    # Check the gradient
    diff = check_grad(objective, gradient, test_point)
    
    # A small difference indicates that the Jacobian is accurate
    threshold = 1e-4
    if diff < threshold:
        print(f"✅ Jacobian validation passed: difference = {diff:.6e}")
    else:
        print(f"⚠️ Jacobian validation warning: difference = {diff:.6e} > {threshold}")
    
    return diff

def find_parameters(parameter_set, mu_target=None, variance_target=None, autocorr_target=None, cv_target=None, fano_factor_target=None,
                    rho_range=(1, 1000), sigma_u_range=(0.1, 1000), d_range=(0.1, 5), num_guesses=1000, 
                    check_biological=True, max_fano_factor=20, max_cv=5.0):
    """
    Find parameters rho, sigma_u, and d that satisfy the equations for given target statistical properties.

    You can specify at most 3 of these statistical properties: mean, variance, autocorrelation, CV, and Fano factor.
    The function will solve for the corresponding telegraph model parameters (rho, sigma_u, d).

    Note: You cannot specify both variance and CV (they are related by CV = sqrt(variance)/mean).
    Note: You cannot specify both autocorrelation and Fano factor (would over-constrain the system).
    
    Statistical properties map to model parameters as follows:
    - Mean (mu_target) → rho
    - Variance (variance_target) or CV (cv_target) → sigma_u
    - Autocorrelation (autocorr_target) or Fano factor (fano_factor_target) → d
    
    Args:
        parameter_set (dict): Dictionary with at least a 'sigma_u' key.
        mu_target (float, optional): Target mean.
        variance_target (float, optional): Target variance.
        autocorr_target (float, optional): Target autocorrelation at t=1.
        cv_target (float, optional): Target coefficient of variation.
        fano_factor_target (float, optional): Target Fano factor.
        rho_range (tuple): Range for rho initial guesses.
        sigma_u_range (tuple): Range for sigma_u initial guesses.
        d_range (tuple): Range for d initial guesses.
        num_guesses (int): Number of random initial guesses to try.
        check_biological (bool): Whether to check if the solution is biologically appropriate.
        max_fano_factor (float): Maximum allowed Fano factor for a biologically appropriate solution.
        max_cv (float): Maximum allowed coefficient of variation for a biologically appropriate solution.
    
    Returns:
        tuple: A tuple containing the found parameters (rho, sigma_u, d).
    """

    sigma_b_val = parameter_set.get("sigma_b")
    if sigma_b_val is None:
        raise ValueError("parameter_set must include a 'sigma_b' key.")

    # TODO: Get rid of this
    # Check if we're using the specialized case (mean, CV, and autocorrelation)
    # use_specialized_jacobian = (mu_target is not None and cv_target is not None and 
    #                            autocorr_target is not None and
    #                            variance_target is None and fano_factor_target is None)
    
    # if use_specialized_jacobian:
    #     print("Using specialized Jacobian for CV, mean, and autocorrelation")
    # else:
    #     # Validate the general Jacobian before using it in fsolve
    #     validate_jacobian(sigma_b_val, mu_target, variance_target, autocorr_target, cv_target, fano_factor_target)

    to_solve = []
    fixed = {}

    # Count how many statistical properties we're trying to fix
    # We can only solve for 3 parameters (rho, sigma_u, d), so we can only fix 3 properties
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

    
    # Cannot specify both variance and CV (would be redundant/potentially conflicting)
    if variance_target is not None and cv_target is not None:
        raise ValueError("Cannot specify both variance_target and cv_target (they are related by CV = sqrt(variance)/mean)")
    
    # Variance or CV -> sigma_u
    if variance_target is not None or cv_target is not None:
        to_solve.append("sigma_u")
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
            current_sigma_u_range = sigma_u_range
            current_d_range = d_range
            current_num_guesses = num_guesses
        else:
            # Expand ranges by 30% each failed attempt
            factor = min(1 + attempt * 0.3, max_factor)
            current_rho_range = (rho_range[0], rho_range[1] * factor)
            current_sigma_u_range = (sigma_u_range[0], sigma_u_range[1] * factor)
            current_d_range = (d_range[0], d_range[1] * factor)
            current_num_guesses = min(int(num_guesses * (1 + attempt * 0.2)), max_guesses)

        # Generate guesses
        if "rho" in to_solve:
            rho_guesses = np.random.uniform(*current_rho_range, current_num_guesses)
        else:
            rho_guesses = [fixed["rho"]] * current_num_guesses
        if "sigma_u" in to_solve:
            sigma_u_guesses = np.random.uniform(*current_sigma_u_range, current_num_guesses)
        else:
            sigma_u_guesses = [fixed["sigma_u"]] * current_num_guesses
        if "d" in to_solve:
            d_guesses = np.random.uniform(*current_d_range, current_num_guesses)
        else:
            d_guesses = [fixed["d"]] * current_num_guesses

        initial_guesses = list(zip(rho_guesses, sigma_u_guesses, d_guesses))

        # print(f"Initial guesses: {initial_guesses}")

        for initial_guess in initial_guesses:
            try:
                solution = fsolve(
                    equations, initial_guess,
                    args=(sigma_b_val, mu_target, variance_target, autocorr_target, cv_target, fano_factor_target),
                    fprime=jacobian,  # Always use the main jacobian function
                    xtol=1e-8
                )

                residuals = equations(solution, sigma_b_val, mu_target, variance_target, autocorr_target, cv_target, fano_factor_target)
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


# TODO: a quick version of find_parameters
def quick_find_parameters(sigma_b, mu_target=None, autocorr_target=None, cv_target=None):
    ''' Providing we are only fixing CV, mean and autocorrelation.
    Args:
        sigma_b (float): Value of sigma_b.
        mu_target (float, optional): Target mean.
        autocorr_target (float, optional): Target autocorrelation at t=1.
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
        
        # return -0.5 + (-(d_val*(1 + (-1 + cv_target)*np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + 2*cv_target + mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_target*sigma_b)))*mu_target) + 
        #   (-1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + 2*cv_target + mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_target*sigma_b)) + 
        #      cv_target**2*(1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + 2*cv_target + mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_target*sigma_b))) + 
        #      cv_target*(-2*np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + 2*cv_target + mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_target*sigma_b)) + mu_target))*sigma_b)/(cv_target*np.exp(autocorr_target*d_val)*(-(d_val*mu_target) + (-2 + 2*cv_target + mu_target)*sigma_b))
        
        #TODO: The right equation
        cv_sq_target = cv_target ** 2
        return -0.5 + (-(d_val*mu_target*(1 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target))) + (-1 + cv_sq_target*mu_target**2 + cv_sq_target**2*mu_target**2 + np.exp((autocorr_target*d_val*(d_val*mu_target - (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))/(d_val*mu_target + sigma_b - cv_sq_target*mu_target*sigma_b))*(-1 + cv_sq_target*mu_target)**2)*sigma_b)/(cv_sq_target*np.exp(autocorr_target*d_val)*mu_target*(-(d_val*mu_target) + (-2 + mu_target + 2*cv_sq_target*mu_target)*sigma_b))

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
            #     sigma_u = -(((-1 + cv_target) * sigma_b * (d_value + sigma_b)) / 
            #    (-d_value * mu_target + (-1 + cv_target) * sigma_b))
            
                #TODO: This is the right equation, need to use cv_sq_target so it's easier to solve
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