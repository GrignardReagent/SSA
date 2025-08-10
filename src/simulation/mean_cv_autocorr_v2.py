#!/usr/bin/python

import sympy as sp
from sympy import init_printing
import numpy as np
from scipy.optimize import fsolve, check_grad, minimize_scalar, root_scalar
from stats.fano_factor import calculate_fano_factor, calculate_fano_factor_from_params
from stats.cv import calculate_cv, calculate_cv_from_params
from stats.mean import calculate_mean_from_params
from stats.variance import calculate_variance_from_params
from stats.autocorrelation import calculate_ac_from_params

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

# # Rescale parameters
# rho_tilda = rho / (sigma_b + sigma_u)
# d_tilda = d / (sigma_b + sigma_u)
# sigma_b_tilda = sigma_b / (sigma_b + sigma_u)
# sigma_u_tilda = sigma_u / (sigma_b + sigma_u)
# # t_tilda is the only one that's multiplied by the scaling factor 
# t_tilda = t * (sigma_b + sigma_u)

############## EXPERIMENTAL #####################
# Compute rescaled parameters for a fixed ``sigma_sum``.
#
# This helper solves the mean, CV and autocorrelation equations in the
# rescaled space assuming a known sum of switching rates ``sigma_sum``.
# It is intentionally kept private; the public :func:`find_tilda_parameters`
# routine wraps this function and automatically searches for an appropriate
# ``sigma_sum``.
def _solve_tilda_parameters(
    sigma_sum: float,
    mu_target: float,
    t_ac_target: float,
    cv_target: float,
    ac_target: float = np.exp(-1),
):
    """Solve for ``rho``, ``d``, ``sigma_b`` and ``sigma_u`` given ``sigma_sum``.

    The equations are expressed in a rescaled space where all parameters are
    divided by ``sigma_sum`` to simplify the algebra.  ``_solve_tilda_parameters``
    inverts these relationships to recover the original parameters.

    Parameters
    ----------
    sigma_sum : float
        The sum ``sigma_b + sigma_u`` used to scale the system.
    mu_target : float
        Target mean of the system.
    t_ac_target : float
        Target autocorrelation time.
    cv_target : float
        Desired coefficient of variation.
    ac_target : float, optional
        Target autocorrelation value at ``t_ac_target``. Defaults to ``exp(-1)``.

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
    
    # t_tilda = (sigma_b + sigma_u) * t; 
    t_tilda = t_ac_target * sigma_sum
    
    # find d_tilda from equation C, via t_tilda, v
    def scaled_ac_equation(d_tilda):
        '''
        AC(t_tilda), rescaled equation of AC(t) to find d_tilda:
        '''
        
        scaled_ACmRNA_eq = ((d_tilda * v * np.exp(- t_tilda) + (d_tilda - v - 1) * np.exp(- d_tilda * t_tilda)) / ((d_tilda -1) * (v + 1)))

        
        return float(scaled_ACmRNA_eq - ac_target)
    
    ################ Use root scalar to find d_tilda, AC(t_tilda) - AC_target = 0  is a root-finding problem, so root_scalar is more appropriate ##################
    
    # the AC(t_tilda) formula has the denominator (d_tilda -1)(v + 1), and d_tilda = 1 is undefined, so the search space needs to exclude 1.0
    lower, upper = 1e-3, 1e3
    candidates = [(lower, 0.999), (1.001, upper)] if (lower < 1.0 < upper) else [(lower, upper)]
    
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
                if abs(residue) > 1e-6 or d_tilda_solution is None or np.isclose(d_tilda_solution, 1e3, atol=1e-4):
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


def find_sigma_sum(
    mu_target: float,
    t_ac_target: float,
    cv_target: float,
    ac_target: float = np.exp(-1),
    tolerance: float = 1e-3,
    initial_guess: float = 1.0,
    max_iter: int = 100,
    max_sigma_sum: float = 1e6,
):
    """Find a suitable ``sigma_sum`` for the given targets.

    ``_solve_tilda_parameters`` requires the sum of the switching rates
    (``sigma_b + sigma_u``) as an input. For some parameter combinations
    only sufficiently large values of ``sigma_sum`` yield parameters that
    reproduce the desired mean, CV and autocorrelation when plugged back
    into the original equations. This helper searches for the smallest
    ``sigma_sum`` that achieves the requested accuracy.

    The search proceeds in two phases:

    1. **Expansion phase** – repeatedly double ``sigma_sum`` until the
       parameters returned by :func:`_solve_tilda_parameters` reproduce the
       targets within the specified tolerance.
    2. **Refinement phase** – perform a binary search between the last
       invalid and valid ``sigma_sum`` to locate the minimal value that
       still satisfies all constraints.

    Parameters
    ----------
    mu_target : float
        Target mean of the system.
    t_ac_target : float
        Desired autocorrelation time.
    cv_target : float
        Target coefficient of variation.
    ac_target : float, optional
        Target autocorrelation value at ``t_ac_target``. Defaults to
        ``exp(-1)``.
    tolerance : float, optional
        Acceptable absolute error between the obtained mean, CV and
        autocorrelation and their targets. Defaults to ``1e-3``.
    initial_guess : float, optional
        Starting value for ``sigma_sum``. Defaults to ``1.0``.
    max_iter : int, optional
        Maximum number of refinement iterations. Defaults to ``100``.
    max_sigma_sum : float, optional
        Upper bound while searching. Defaults to ``1e6``.

    Returns
    -------
    tuple
        ``(sigma_sum, rho, d, sigma_b, sigma_u)`` where ``sigma_sum``
        is the found switching-rate sum and the remaining values are the
        parameters returned by :func:`_solve_tilda_parameters`.

    Raises
    ------
    ValueError
        If a suitable ``sigma_sum`` cannot be found within ``max_sigma_sum``
        or if ``_solve_tilda_parameters`` fails for all tested values.
    """

    def _errors(sigma_sum: float):
        """Return mean, CV and autocorrelation errors for ``sigma_sum``."""

        # Attempt to derive parameters for the current ``sigma_sum``. Any
        # failure (e.g., no solution) is treated as an infinite error so the
        # caller knows this value is invalid.
        try:
            params = _solve_tilda_parameters(
                sigma_sum, mu_target, t_ac_target, cv_target, ac_target
            )
        except ValueError:
            return np.inf, np.inf, np.inf, None

        # Evaluate how far the resulting parameters deviate from the targets.
        rho, d, sigma_b, sigma_u = params
        mu_val = calculate_mean_from_params(rho, d, sigma_b, sigma_u)
        cv_val = calculate_cv_from_params(rho, d, sigma_b, sigma_u)
        ac_val = calculate_ac_from_params(
            rho, d, sigma_b, sigma_u, t_ac_target, ac_target
        )
        return (
            mu_val - mu_target,
            cv_val - cv_target,
            ac_val - ac_target,
            params,
        )

    # ----------
    # Phase 1: find an upper bound where all three metrics are within
    # tolerance by repeatedly doubling ``sigma_sum``.
    # ----------
    sigma_high = initial_guess
    mu_err_high, cv_err_high, ac_err_high, params_high = _errors(sigma_high)
    iterations = 0
    def _within_tol(m_err, c_err, a_err):
        """Check if all errors are finite and within the requested tolerance."""
        return (
            np.isfinite(m_err)
            and np.isfinite(c_err)
            and np.isfinite(a_err)
            and abs(m_err) <= tolerance
            and abs(c_err) <= tolerance
            and abs(a_err) <= tolerance
        )
    # while the guess doesn't produce finite and within-tolerance errors, expand guess value for sigma_sum, do this till we reach max_iter, else raise error. 
    while not _within_tol(mu_err_high, cv_err_high, ac_err_high) and iterations < max_iter:
        sigma_high *= 2
        if sigma_high > max_sigma_sum:
            raise ValueError("Could not find a valid sigma_sum within bounds.")
        mu_err_high, cv_err_high, ac_err_high, params_high = _errors(sigma_high)
        iterations += 1

    if not _within_tol(mu_err_high, cv_err_high, ac_err_high):
        raise ValueError(
            "Unable to determine sigma_sum that satisfies mean, CV and autocorrelation tolerances."
        )

    # ----------
    # Phase 2: refine using a binary search between the last failing value and
    # the successful ``sigma_high`` to find the minimal acceptable ``sigma_sum``.
    # ----------
    sigma_low = sigma_high / 2
    for _ in range(max_iter):
        if sigma_high - sigma_low <= 1e-6:
            break
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        mu_err_mid, cv_err_mid, ac_err_mid, params_mid = _errors(sigma_mid)

        # if sigma_mid is within tolerance, we have found the minimal ascceptable sigma_sum!
        if _within_tol(mu_err_mid, cv_err_mid, ac_err_mid):
            sigma_high = sigma_mid
            mu_err_high, cv_err_high, ac_err_high, params_high = (
                mu_err_mid,
                cv_err_mid,
                ac_err_mid,
                params_mid,
            )
        else:
            sigma_low = sigma_mid

    return sigma_high, *params_high


def find_tilda_parameters(
    mu_target: float,
    t_ac_target: float,
    cv_target: float,
    ac_target: float = np.exp(-1),
    sigma_sum_seed: float = 1.0,
    tolerance: float = 1e-3,
    check_biological: bool = True,
    max_fano_factor: float = 20,
    min_fano_factor: float = 1,
    max_cv: float = 5.0,
    max_iter: int = 100,
    max_sigma_sum: float = 1e6,
):
    """Find parameters matching ``mu``, ``CV`` and autocorrelation targets.

    This is the primary entry point for users.  The function searches for
    a suitable switching-rate sum ``sigma_sum`` starting from
    ``sigma_sum_seed`` and refines it until the resulting parameters
    reproduce the requested mean, CV and autocorrelation within
    ``tolerance``.

    Parameters
    ----------
    mu_target : float
        Desired mean of the system.
    t_ac_target : float
        Target autocorrelation time.
    cv_target : float
        Desired coefficient of variation.
    ac_target : float, optional
        Target autocorrelation value at ``t_ac_target``. Defaults to
        ``exp(-1)``.
    sigma_sum_seed : float, optional
        Initial guess for ``sigma_sum``. Defaults to ``1.0``.
    tolerance : float, optional
        Acceptable absolute error for mean, CV and autocorrelation.
    check_biological : bool, optional
        Whether to verify that the solution is biologically plausible via
        :func:`check_biological_appropriateness`.
    max_fano_factor, min_fano_factor, max_cv : float, optional
        Thresholds passed to :func:`check_biological_appropriateness` when
        ``check_biological`` is ``True``.
    max_iter : int, optional
        Maximum refinement iterations for the ``sigma_sum`` search.
    max_sigma_sum : float, optional
        Upper bound while searching for ``sigma_sum``.

    Returns
    -------
    tuple
        ``(rho, d, sigma_b, sigma_u)`` satisfying the targets. The optimal
        switching-rate sum is searched internally and not returned.
    """

    _sigma_sum, rho, d, sigma_b, sigma_u = find_sigma_sum(
        mu_target,
        t_ac_target,
        cv_target,
        ac_target=ac_target,
        tolerance=tolerance,
        initial_guess=sigma_sum_seed,
        max_iter=max_iter,
        max_sigma_sum=max_sigma_sum,
    )

    if check_biological:
        variance_target = (cv_target * mu_target) ** 2
        is_appropriate = check_biological_appropriateness(
            variance_target, mu_target, max_fano_factor, min_fano_factor, max_cv
        )
        if not is_appropriate:
            raise ValueError(
                "Solution is not biologically appropriate. "
                f"Consider adjusting target parameters: mu={mu_target}, cv={cv_target}, "
                f"which gives variance={variance_target:.2f}"
            )

    return rho, d, sigma_b, sigma_u
