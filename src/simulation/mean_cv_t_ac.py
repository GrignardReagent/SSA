#!/usr/bin/python

import numpy as np
from scipy.optimize import root_scalar
from utils.biological import check_biological_appropriateness
from stats.mean import calculate_mean_from_params
from stats.variance import calculate_variance_from_params
from stats.autocorrelation import calculate_ac_from_params
import warnings


def ac_limit_at_one(t_ac_tilda: float, v: float) -> float:
    """Autocorrelation value as ``d`` approaches one."""

    return float(np.exp(-t_ac_tilda) * (1.0 + v + t_ac_tilda * v) / (1.0 + v))


def d_ac_dd_at_one(t_ac_tilda: float, v: float) -> float:
    """Derivative of the autocorrelation with respect to ``d`` at ``d=1``."""

    return float(
        -np.exp(-t_ac_tilda)
        * t_ac_tilda
        * (t_ac_tilda * v + 2.0)
        / (2.0 * (1.0 + v))
    )


def _solve_d_tilda_limit(
    ac_target: float,
    t_ac_tilda: float,
    v: float,
    a: float,
    b: float,
    res_limit: float,
    scaled_ac_equation,
) -> float:
    """Linearised recovery of ``d_tilda`` when Brent's root hits a boundary.

    Parameters
    ----------
    ac_target, t_ac_tilda, v
        Problem parameters used for the Taylor expansion around ``d=1``.
    a, b
        Original bracket edges.
    res_limit
        Maximum allowed absolute residual of the autocorrelation equation.
    scaled_ac_equation
        Callable evaluating the full scaled autocorrelation residual.

    Returns
    -------
    float
        A valid ``d_tilda`` estimate inside ``(a, b)``.

    Raises
    ------
    ValueError
        If the linearised path or optional Newton refinement fails to
        produce a valid solution.
    """

    print("[d_tilda] Boundary hit → using linearised l’Hôpital/Taylor solve around d=1")

    AC1 = ac_limit_at_one(t_ac_tilda, v)
    dAC_dd_1 = d_ac_dd_at_one(t_ac_tilda, v)
    if np.isclose(dAC_dd_1, 0.0, atol=1e-14):
        raise ValueError("Slope at d=1 is ~0; cannot linearise to recover d_tilda.")

    edge_tol = 1e-4
    inner_a = a + 2 * edge_tol
    inner_b = b - 2 * edge_tol

    d_est = 1.0 + (ac_target - AC1) / dAC_dd_1
    d_est = float(np.clip(d_est, inner_a, inner_b))
    res_est = scaled_ac_equation(d_est)
    if abs(res_est) <= res_limit:
        return d_est

    def num_deriv(x: float) -> float:
        h = 1e-6
        return (scaled_ac_equation(x + h) - scaled_ac_equation(x - h)) / (2 * h)

    df_est = num_deriv(d_est)
    if np.isclose(df_est, 0.0, atol=1e-14):
        raise ValueError("Slope at Newton start is ~0; cannot refine d_tilda.")

    d_new = d_est - res_est / df_est
    d_new = float(np.clip(d_new, inner_a, inner_b))
    res_new = scaled_ac_equation(d_new)
    if abs(res_new) <= res_limit:
        return d_new

    raise ValueError(
        f"⚠️No valid solution found for parameter d_tilda: residual={res_new:.2e}, d_tilda={d_new:.4f}"
    )

# # Rescale parameters
# rho_tilda = rho / (sigma_b + sigma_u)
# d_tilda = d / (sigma_b + sigma_u)
# sigma_b_tilda = sigma_b / (sigma_b + sigma_u)
# sigma_u_tilda = sigma_u / (sigma_b + sigma_u)
# # t_ac_tilda is the only one that's multiplied by the scaling factor 
# t_ac_tilda = t_ac * (sigma_b + sigma_u)

# Compute rescaled parameters for a fixed ``sigma_sum``.
#
# This helper solves the mean, CV and autocorrelation time equations in the
# rescaled space assuming a known sum of switching rates ``sigma_sum`` (default = 1).
# It is intentionally kept private; the public :func:`find_tilda_parameters`
# routine wraps this function and automatically searches for an appropriate
# ``sigma_sum``.
def find_tilda_parameters(
    mu_target: float,
    t_ac_target: float,
    cv_target: float,
    sigma_sum: float = 1.0,
    ac_target: float = np.exp(-1),
    res_limit: float = 1e-3,
    lower: float = 1e-3,
    upper: float = 1e3,
    check_biological: bool = False,
    max_fano_factor: float = 20,
    min_fano_factor: float = 1.0,
    max_cv: float = 5,
):
    """Solve for ``rho``, ``d``, ``sigma_b`` and ``sigma_u`` given ``sigma_sum``.

    The equations are expressed in a rescaled space where all parameters are divided by ``sigma_sum`` to simplify the algebra.  

    Parameters
    ----------
    mu_target : float
        Target mean of the system.
    t_ac_target : float
        Target autocorrelation time.
    cv_target : float
        Desired coefficient of variation.
    sigma_sum : float, optional
        The sum ``sigma_b + sigma_u`` used to scale the system. Default to be 1.0. Changing this might lead to unexpected bahaviours!
    ac_target : float, optional
        Target autocorrelation value at ``t_ac_target``. Defaults to ``exp(-1)``.
    res_limit : float, optional
        Residual limit for root-finding. Defaults to ``1e-3``. This is used to check if the solution found is valid or not.
    lower : float, optional
        Lower bound for the root-finding search space for d_tilda. Defaults to ``1e-3``
    upper : float, optional
        Upper bound for the root-finding search space for d_tilda. Defaults to ``1e3``.
    check_biological : bool, optional
        If True, checks if the solution is biologically appropriate. Defaults to False.
    max_fano_factor : float, optional
        Maximum allowed Fano factor for the solution. Defaults to 20.
    min_fano_factor : float, optional
        Minimum allowed Fano factor for the solution. Defaults to 1.0.
    max_cv : float, optional
        Maximum allowed coefficient of variation for the solution. Defaults to 5.

    Returns
    -------
    tuple
        ``(rho, d, sigma_b, sigma_u)`` that reproduce the requested targets.

    Raises
    ------
    ValueError
        If no valid solution for ``d`` can be found.
    """
    
    # Basic feasibility checks
    if mu_target <= 0:
        raise ValueError("mu_target must be > 0.")
    if cv_target <= 0:
        raise ValueError("cv_target must be > 0.")
    if t_ac_target <= 0:
        raise ValueError("t_ac_target must be > 0 (AC(0)=1 carries no info).")
    if sigma_sum <= 0:
        raise ValueError("sigma_sum (Σ) must be > 0.")

    
    # using equation B, we find v via mu_target & cv_target^2
    v = (mu_target * (cv_target ** 2)) - 1
    
    # check if v > 0, if not, there's no solution
    if v <= 0:
        raise ValueError(f"Invalid parameters: v = (mu_target * cv_target ** 2) - 1 must be positive, got v = {v}. Reconsider mu_target and cv_target choices.")
    
    # t_ac_tilda = (sigma_b + sigma_u) * t; 
    t_ac_tilda = t_ac_target * sigma_sum
    
    # scaled lag: warn if far from 1 (conditioning)
    if t_ac_tilda < 1e-3 or t_ac_tilda > 1e3:
        warnings.warn(
            f"Scaled lag t̃ = Σ·t_ac = {t_ac_tilda:.3g} is ill-conditioned; "
            f"consider Σ≈1/t_ac_target so t̃≈1."
        )
    
    # find d_tilda from equation C, via t_ac_tilda, v
    def scaled_ac_equation(d_tilda):
        '''
        AC(t_ac_tilda), rescaled equation of AC(t) to find d_tilda:
        '''
        
        scaled_ACmRNA_eq = ((d_tilda * v * np.exp(- t_ac_tilda) + (d_tilda - v - 1) * np.exp(- d_tilda * t_ac_tilda)) / ((d_tilda -1) * (v + 1)))

        
        return float(scaled_ACmRNA_eq - ac_target)
    
    ################ Use root scalar to find d_tilda, AC(t_ac_tilda) - AC_target = 0  is a root-finding problem, so root_scalar is more appropriate ##################
    
    # the AC(t_ac_tilda) formula has the denominator (d_tilda -1)(v + 1), and d_tilda = 1 is undefined, so the search space needs to exclude 1.0; multiple decimal places to bracket away from 1.0
    candidates = [(lower, 0.999), (1.001, upper)] if (lower < 1.0 < upper) else [(lower, upper)]  # several decimal places are intentionally used here to avoid values close to 1.0
    
    d_tilda = None
    for a, b in candidates:
        try:
            result = root_scalar(scaled_ac_equation, bracket=[a, b],method='brentq', # Brent's method for root finding, generally considered the best for this. 
                                 xtol=1e-10, rtol=1e-10, maxiter=200)
            
            if result.converged:
                d_tilda_solution = result.root

                # error trap for no solution found for d_tilda_solution
                # 1. checks that absolute value of residue is less than res_limit
                # 2. checks that d_tilda_solution is not None
                residue = scaled_ac_equation(d_tilda_solution)
                if abs(residue) > res_limit or d_tilda_solution is None:
                    raise ValueError(
                        f"⚠️No valid solution found for parameter d_tilda: residual={residue:.2e}, d_tilda={d_tilda_solution:.4f}"
                    )

                # If the root lies extremely close to a bracket edge, use a
                # linearised l'Hôpital/Taylor expansion around d=1. This avoids
                # the singular behaviour of the original autocorrelation
                # equation.
                if np.isclose(d_tilda_solution, a, atol=1e-4) or np.isclose(
                    d_tilda_solution, b, atol=1e-4
                ):
                    d_tilda = _solve_d_tilda_limit(
                        ac_target,
                        t_ac_tilda,
                        v,
                        a,
                        b,
                        res_limit,
                        scaled_ac_equation,
                    )
                else:
                    d_tilda = d_tilda_solution

                break
                
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
    
    # scaling back to original param space
    sigma_b = sigma_b_tilda * sigma_sum
    sigma_u = sigma_u_tilda * sigma_sum
    rho = rho_tilda * sigma_sum
    d = d_tilda * sigma_sum
    
    # optional biological check
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
        
    # back-substitution verification against original formulas
    mu_analytical = calculate_mean_from_params(rho, d, sigma_b, sigma_u)
    var_analytical = calculate_variance_from_params(rho, d, sigma_b, sigma_u)
    cv_analytical = np.sqrt(var_analytical) / mu_analytical
    ac_analytical = calculate_ac_from_params(rho, d, sigma_b, sigma_u, t_ac_target)
    
    # relative/absolute residuals
    rel = lambda x, y: abs(x - y) / max(1.0, abs(y)) # behaves like a relative error when abs(y) >= 1 and like an absolute error when abs(y) < 1. In contrastm, using just abs(y) in the denominator blows up (or becomes overly strict) when y is tiny or zero.
    mu_err = rel(mu_analytical, mu_target)
    cv_err = rel(cv_analytical, cv_target)
    ac_err = abs(ac_analytical - ac_target)
    if mu_err > 0.2 or cv_err > 0.2 or ac_err > 0.2:
        raise ValueError(
            "Back-substitution check failed: "
            f"μ_err={mu_err:.2e}, CV_err={cv_err:.2e}, AC_err={ac_err:.2e}. "
            "Try adjusting Σ so that t̃≈1, or relax targets."
        )
    return rho, d, sigma_b, sigma_u
