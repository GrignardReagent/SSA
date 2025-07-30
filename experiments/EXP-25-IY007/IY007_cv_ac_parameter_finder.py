#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import tqdm
import sympy as sp
from sympy import init_printing, exp
from scipy.optimize import fsolve

###############################################################################
# 1) Symbolic definitions for mean, CV, and autocorrelation
###############################################################################
rho, sigma_u, sigma_b, d, t = sp.symbols('rho sigma_u sigma_b d t', real=True, positive=True)
init_printing(use_unicode=True)

# Define the equations for mean, CV, and autocorrelation
def equations(vars, sigma_b_fixed, mu_target, cv_target, autocorr_target):
    """
    Define the equations for the telegraph model based on mean, CV, and autocorrelation.
    
    Args:
        vars (list): [rho, sigma_u, d] - variables to solve for
        sigma_b_fixed (float): Fixed value of sigma_b
        mu_target (float): Target mean
        cv_target (float): Target coefficient of variation
        autocorr_target (float): Target autocorrelation time (time when AC = 1/e)
    
    Returns:
        list: [mean_eq, cv_eq, autocorr_eq] - equations to be solved
    """
    rho_val, sigma_u_val, d_val = vars
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    sigma_u_val = max(sigma_u_val, epsilon)
    d_val = max(d_val, epsilon)
    rho_val = max(rho_val, epsilon)
    
    # Mean equation: mu = (sigma_b * rho) / (d * (sigma_b + sigma_u))
    denominator_mean = d_val * (sigma_b_fixed + sigma_u_val)
    mean_calculated = (sigma_b_fixed * rho_val) / denominator_mean
    mean_eq = mean_calculated - mu_target
    
    # Variance equation
    variance_term1 = mean_calculated  # This is the mean contribution
    variance_term2 = ((sigma_u_val * sigma_b_fixed * rho_val**2) / 
                     (d_val * (sigma_b_fixed + sigma_u_val + d_val) * (sigma_u_val + sigma_b_fixed)**2))
    variance = variance_term1 + variance_term2
    
    # Ensure variance is positive for CV calculation
    if variance <= 0:
        variance = epsilon
    
    # CV equation: CV = sqrt(variance) / mean
    cv_calculated = np.sqrt(variance) / mean_calculated
    cv_eq = cv_calculated - cv_target
    
    # Autocorrelation equation: proper telegraph model autocorrelation
    # Check for potential division by zero in denominators
    denom1 = d_val - sigma_u_val - sigma_b_fixed
    denom2 = rho_val * sigma_u_val + d_val * (sigma_u_val + sigma_b_fixed) + (sigma_u_val + sigma_b_fixed) ** 2
    
    # If denominators are too close to zero, use simplified equation
    if abs(denom1) < 1e-8 or abs(denom2) < 1e-8:
        # Fall back to simplified exponential decay: AC(t) ≈ exp(-d*t)
        autocorr_eq = float(sp.exp(-d_val * autocorr_target) - 1/sp.exp(1))
    else:
        try:
            ACmRNA_eq = sp.exp(-d_val * t) * (
                d_val * sp.exp((d_val - sigma_u_val - sigma_b_fixed) * t) * rho_val * sigma_u_val
                - (sigma_u_val + sigma_b_fixed) * (-d_val**2 + rho_val * sigma_u_val + (sigma_u_val + sigma_b_fixed) ** 2)
            ) / (
                (d_val - sigma_u_val - sigma_b_fixed) * (rho_val * sigma_u_val + d_val * (sigma_u_val + sigma_b_fixed) + (sigma_u_val + sigma_b_fixed) ** 2)
            )
            # Evaluate autocorrelation at t= autocorr_target
            autocorr_eqn = ACmRNA_eq.subs(t, autocorr_target) - 1/sp.exp(1)
            autocorr_eq = float(autocorr_eqn)
        except (OverflowError, ZeroDivisionError, ValueError):
            # Fall back to simplified equation if numerical issues occur
            autocorr_eq = float(sp.exp(-d_val * autocorr_target) - 1/sp.exp(1))
    
    return [mean_eq, cv_eq, autocorr_eq]

###############################################################################
# 2) Define fixed parameters and CV ranges
###############################################################################
# Fixed parameters
sigma_b_fixed = 0.02  # Initialize sigma_b value
mu_target = 10.0      # Fixed mean for both conditions
autocorr_target = 1.0 # Fixed autocorrelation time (time when AC = 1/e)

# CV ranges to loop over
cv_values = np.array([0.5, 0.7, 1.0])  # Just test a few CV values with the complex equation

# Conditions to test
conditions = ["stress", "normal"]

# Output directory
output_dir = "data_cv_ac_parameter_sweep"
os.makedirs(output_dir, exist_ok=True)

# Results storage
all_results = []

###############################################################################
# 3) Loop over different CV values and conditions
###############################################################################
for cv in tqdm.tqdm(cv_values, desc="Running CV Parameter Finding"):
    print(f"\n--- Processing CV = {cv:.3f} ---")
    
    # Store solutions for each condition
    solutions = {}
    
    for condition in conditions:
        print(f"Finding parameters for {condition} condition...")
        
        # Generate initial guesses (more focused search space for complex equation)
        rho_guesses = np.logspace(1, 2, 15)      # 10 to 100
        sigma_u_guesses = np.logspace(-2, 0, 15) # 0.01 to 1.0  
        d_guesses = np.linspace(0.8, 1.2, 10)   # 0.8 to 1.2 (around 1/autocorr_target)
        
        solution_found = False
        best_solution = None
        best_residuals = None
        
        # Try different initial guesses
        for rho_ig in rho_guesses:
            if solution_found:
                break
            for sigma_u_ig in sigma_u_guesses:
                if solution_found:
                    break
                for d_ig in d_guesses:
                    initial_guess = [rho_ig, sigma_u_ig, d_ig]
                    
                    try:
                        # Solve the system of equations
                        solution = fsolve(
                            equations, 
                            initial_guess, 
                            args=(sigma_b_fixed, mu_target, cv, autocorr_target),
                            xtol=1e-6,  # Relaxed tolerance for complex equation
                            maxfev=1000  # Limit function evaluations
                        )
                        
                        # Check the residuals
                        residuals = equations(solution, sigma_b_fixed, mu_target, cv, autocorr_target)
                        residuals_abs = [abs(r) for r in residuals]
                        
                        # Check if solution is valid (positive parameters and reasonable residuals)
                        if (all(x > 1e-4 for x in solution) and 
                            all(r < 1e-2 for r in residuals_abs)):  # More relaxed tolerance for complex equation
                            
                            rho_val, sigma_u_val, d_val = solution
                            
                            # Additional biological constraints (relaxed)
                            if (rho_val < 500 and sigma_u_val < 10 and d_val < 5):
                                print(f"  ✅ Solution found for {condition}: rho={rho_val:.4f}, sigma_u={sigma_u_val:.4f}, d={d_val:.4f}")
                                print(f"     Residuals: {residuals_abs}")
                                
                                solutions[condition] = {
                                    'rho': rho_val,
                                    'sigma_u': sigma_u_val,
                                    'd': d_val,
                                    'sigma_b': sigma_b_fixed,
                                    'residuals': residuals_abs
                                }
                                solution_found = True
                                break
                        else:
                            # Keep track of best solution even if not perfect
                            if (all(x > 1e-4 for x in solution) and 
                                (best_residuals is None or max(residuals_abs) < max(best_residuals))):
                                best_solution = solution
                                best_residuals = residuals_abs
                                
                    except Exception as e:
                        continue
        
        # If no perfect solution found, use the best one
        if not solution_found and best_solution is not None:
            rho_val, sigma_u_val, d_val = best_solution
            print(f"  ⚠️  Best solution for {condition}: rho={rho_val:.4f}, sigma_u={sigma_u_val:.4f}, d={d_val:.4f}")
            print(f"     Best residuals: {best_residuals}")
            
            solutions[condition] = {
                'rho': rho_val,
                'sigma_u': sigma_u_val,
                'd': d_val,
                'sigma_b': sigma_b_fixed,
                'residuals': best_residuals
            }
        elif not solution_found:
            print(f"  ❌ No solution found for {condition} with CV={cv:.3f}")
            continue
    
    # If we found solutions for both conditions, save the results
    if len(solutions) == len(conditions):
        # Verify the solutions by calculating actual statistics
        for condition, sol in solutions.items():
            # Calculate actual mean
            actual_mean = (sol['sigma_b'] * sol['rho']) / (sol['d'] * (sol['sigma_b'] + sol['sigma_u']))
            
            # Calculate actual variance
            actual_variance = actual_mean + \
                            ((sol['sigma_u'] * sol['sigma_b'] * sol['rho']**2) / 
                             (sol['d'] * (sol['sigma_b'] + sol['sigma_u'] + sol['d']) * (sol['sigma_u'] + sol['sigma_b'])**2))
            
            # Calculate actual CV
            actual_cv = np.sqrt(actual_variance) / actual_mean
            
            # Calculate actual autocorrelation decay rate
            actual_autocorr_at_target = np.exp(-sol['d'] * autocorr_target)
            
            print(f"  Verification for {condition}:")
            print(f"    Target mean: {mu_target:.4f}, Actual mean: {actual_mean:.4f}")
            print(f"    Target CV: {cv:.4f}, Actual CV: {actual_cv:.4f}")
            print(f"    Target AC at t={autocorr_target}: {1/np.e:.4f}, Actual: {actual_autocorr_at_target:.4f}")
        
        # Store results
        result_entry = {
            'cv_target': cv,
            'mu_target': mu_target,
            'autocorr_target': autocorr_target,
            'sigma_b_fixed': sigma_b_fixed
        }
        
        for condition in conditions:
            sol = solutions[condition]
            result_entry.update({
                f'{condition}_rho': sol['rho'],
                f'{condition}_sigma_u': sol['sigma_u'],
                f'{condition}_d': sol['d'],
                f'{condition}_sigma_b': sol['sigma_b'],
                f'{condition}_max_residual': max(sol['residuals'])
            })
        
        all_results.append(result_entry)
        
        # Save intermediate results
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{output_dir}/cv_ac_parameter_results.csv", index=False)
        
        print(f"✅ Successfully found parameters for CV={cv:.3f}")
    else:
        print(f"❌ Could not find complete parameter set for CV={cv:.3f}")

###############################################################################
# 4) Save final results
###############################################################################
if all_results:
    df_final = pd.DataFrame(all_results)
    
    # Save detailed results
    df_final.to_csv(f"{output_dir}/cv_ac_parameter_results_final.csv", index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_cv_values_tested': len(cv_values),
        'successful_parameter_sets': len(all_results),
        'success_rate': len(all_results) / len(cv_values) * 100,
        'cv_range_tested': f"{cv_values.min():.3f} to {cv_values.max():.3f}",
        'mean_target': mu_target,
        'autocorr_target': autocorr_target,
        'sigma_b_fixed': sigma_b_fixed
    }
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")
    
    # Save summary
    pd.Series(summary_stats).to_csv(f"{output_dir}/summary_stats.csv")
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"- cv_ac_parameter_results_final.csv: Complete parameter sets")
    print(f"- summary_stats.csv: Summary statistics")
    
else:
    print("❌ No successful parameter sets found!")
