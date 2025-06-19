#!/usr/bin/python

import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from simulation.mean_var_autocorr import check_biological_appropriateness, find_parameters
from simulation.mean_cv_autocorr import quick_find_parameters

def compare_parameter_finding_methods():
    """
    Compare the efficiency and parameter values computed by find_parameters() and quick_find_parameters()
    """
    ###############################################################################
    # Define target parameters for comparison
    ###############################################################################
    parameters = {
        "stress": {"sigma_b": 20.0},
        "normal": {"sigma_b": 10.0}
    }
    target_cv_normal = 0.25  # initial CV for normal condition
    mu_target = 50   # Mean
    autocorr_target = 5    # Autocorrelation time
    
    # Use a smaller set of CV ratios for the comparison
    cv_ratios = np.arange(0.5, 2.0, 0.1)  # Fewer points for the comparison
    
    print(f"Starting comparison with:")
    print(f"Normal CV: {target_cv_normal:.2f}, Mean: {mu_target:.2f}, Autocorrelation Time: {autocorr_target:.2f}")
    
    ###############################################################################
    # Run comparison between methods
    ###############################################################################
    comparison_results = []
    
    for ratio in tqdm.tqdm(cv_ratios, desc="Comparing methods across CV ratios"):
        # For the stress condition, we define CV by ratio
        target_cv_stress = ratio * target_cv_normal
        
        # Calculate corresponding variance values
        variance_stress = (target_cv_stress * mu_target)**2
        variance_normal = (target_cv_normal * mu_target)**2
        
        print(f"\nCV ratio: {ratio:.2f}, Stress CV: {target_cv_stress:.2f}, Normal CV: {target_cv_normal:.2f}")
        
        # Check if stress parameters are biologically appropriate
        is_stress_appropriate = check_biological_appropriateness(variance_stress, mu_target)
        if not is_stress_appropriate:
            print(f"⚠️ Stress parameters not biologically appropriate for ratio {ratio:.2f}. Skipping.")
            continue
        
        for condition, param in parameters.items():
            # Decide which CV to use for this condition
            cv_for_condition = target_cv_normal if condition == "normal" else target_cv_stress
            
            result_entry = {
                "ratio": ratio,
                "condition": condition,
                "cv_target": cv_for_condition,
                "sigma_b": param["sigma_b"]
            }
            
            # Test find_parameters
            try:
                start_time = time.time()
                standard_result = find_parameters(
                    param, 
                    mu_target=mu_target, 
                    autocorr_target=autocorr_target, 
                    cv_target=cv_for_condition
                )
                standard_time = time.time() - start_time
                
                result_entry["standard_time"] = standard_time
                result_entry["standard_rho"] = standard_result[0]
                result_entry["standard_sigma_u"] = standard_result[1]
                result_entry["standard_d"] = standard_result[2]
                result_entry["standard_success"] = True
                
                print(f"[{condition}] Standard method: {standard_time:.4f}s, parameters: {standard_result}")
            except Exception as e:
                print(f"[{condition}] Standard method failed: {str(e)}")
                result_entry["standard_time"] = None
                result_entry["standard_success"] = False
            
            # Test quick_find_parameters
            try:
                start_time = time.time()
                quick_result = quick_find_parameters(
                    param["sigma_b"],
                    mu_target=mu_target, 
                    autocorr_target=autocorr_target, 
                    cv_target=cv_for_condition
                )
                quick_time = time.time() - start_time
                
                result_entry["quick_time"] = quick_time
                result_entry["quick_rho"] = quick_result[0]
                result_entry["quick_sigma_u"] = quick_result[1]
                result_entry["quick_d"] = quick_result[2]
                result_entry["quick_success"] = True
                
                print(f"[{condition}] Quick method: {quick_time:.4f}s, parameters: {quick_result}")
                
                # Calculate percentage differences
                if result_entry.get("standard_success"):
                    result_entry["speedup"] = standard_time / quick_time if quick_time > 0 else float('inf')
                    result_entry["rho_diff_pct"] = 100 * abs(quick_result[0] - standard_result[0]) / standard_result[0]
                    result_entry["sigma_u_diff_pct"] = 100 * abs(quick_result[1] - standard_result[1]) / standard_result[1]
                    result_entry["d_diff_pct"] = 100 * abs(quick_result[2] - standard_result[2]) / standard_result[2]
            except Exception as e:
                print(f"[{condition}] Quick method failed: {str(e)}")
                result_entry["quick_time"] = None
                result_entry["quick_success"] = False
            
            comparison_results.append(result_entry)
    
    return comparison_results

def display_results(results):
    """
    Display and visualize the comparison results
    """
    # Calculate summary statistics
    successful_comparisons = [r for r in results if r.get("standard_success") and r.get("quick_success")]
    
    if not successful_comparisons:
        print("No successful comparisons to analyze.")
        return
    
    # Calculate average speedup
    avg_speedup = np.mean([r["speedup"] for r in successful_comparisons])
    
    # Calculate average parameter differences
    avg_rho_diff = np.mean([r["rho_diff_pct"] for r in successful_comparisons])
    avg_sigma_u_diff = np.mean([r["sigma_u_diff_pct"] for r in successful_comparisons])
    avg_d_diff = np.mean([r["d_diff_pct"] for r in successful_comparisons])
    
    # Count successes
    standard_successes = sum(1 for r in results if r.get("standard_success"))
    quick_successes = sum(1 for r in results if r.get("quick_success"))
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(f"Total test cases: {len(results)}")
    print(f"Standard method successes: {standard_successes} ({standard_successes/len(results)*100:.1f}%)")
    print(f"Quick method successes: {quick_successes} ({quick_successes/len(results)*100:.1f}%)")
    print(f"Average speedup (standard/quick): {avg_speedup:.2f}x")
    print("\nAverage parameter differences:")
    print(f"  rho: {avg_rho_diff:.2f}%")
    print(f"  sigma_u: {avg_sigma_u_diff:.2f}%")
    print(f"  d: {avg_d_diff:.2f}%")
    
    # Create table of results
    table_data = []
    for r in successful_comparisons:
        table_data.append([
            f"{r['ratio']:.1f}",
            r['condition'],
            f"{r['standard_time']:.4f}s",
            f"{r['quick_time']:.4f}s",
            f"{r['speedup']:.2f}x",
            f"{r['rho_diff_pct']:.2f}%",
            f"{r['sigma_u_diff_pct']:.2f}%",
            f"{r['d_diff_pct']:.2f}%"
        ])
    
    headers = ["CV Ratio", "Condition", "Standard Time", "Quick Time", "Speedup", 
               "ρ Diff %", "σᵤ Diff %", "d Diff %"]
    
    print("\nDetailed Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Computation time comparison
    plt.subplot(2, 2, 1)
    ratios = [r["ratio"] for r in successful_comparisons]
    standard_times = [r["standard_time"] for r in successful_comparisons]
    quick_times = [r["quick_time"] for r in successful_comparisons]
    
    plt.bar(np.arange(len(ratios))-0.2, standard_times, width=0.4, label='Standard Method')
    plt.bar(np.arange(len(ratios))+0.2, quick_times, width=0.4, label='Quick Method')
    plt.xticks(np.arange(len(ratios)), [f"{r:.1f}" for r in ratios])
    plt.xlabel('CV Ratio')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Speedup by CV ratio
    plt.subplot(2, 2, 2)
    speedups = [r["speedup"] for r in successful_comparisons]
    plt.plot(ratios, speedups, 'o-', markersize=8)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('CV Ratio')
    plt.ylabel('Speedup Factor (standard/quick)')
    plt.title('Performance Speedup by CV Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameter difference percentages
    plt.subplot(2, 2, 3)
    rho_diffs = [r["rho_diff_pct"] for r in successful_comparisons]
    sigma_u_diffs = [r["sigma_u_diff_pct"] for r in successful_comparisons]
    d_diffs = [r["d_diff_pct"] for r in successful_comparisons]
    
    plt.plot(ratios, rho_diffs, 'o-', label='ρ difference %', markersize=8)
    plt.plot(ratios, sigma_u_diffs, 's-', label='σᵤ difference %', markersize=8)
    plt.plot(ratios, d_diffs, '^-', label='d difference %', markersize=8)
    plt.xlabel('CV Ratio')
    plt.ylabel('Difference (%)')
    plt.title('Parameter Difference by CV Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Calculation success rate
    plt.subplot(2, 2, 4)
    
    # Group by CV ratio
    unique_ratios = sorted(set(r["ratio"] for r in results))
    standard_success_rate = []
    quick_success_rate = []
    
    for ratio in unique_ratios:
        ratio_results = [r for r in results if r["ratio"] == ratio]
        standard_success_rate.append(sum(1 for r in ratio_results if r.get("standard_success")) / len(ratio_results) * 100)
        quick_success_rate.append(sum(1 for r in ratio_results if r.get("quick_success")) / len(ratio_results) * 100)
    
    plt.bar(np.arange(len(unique_ratios))-0.2, standard_success_rate, width=0.4, label='Standard Method')
    plt.bar(np.arange(len(unique_ratios))+0.2, quick_success_rate, width=0.4, label='Quick Method')
    plt.xticks(np.arange(len(unique_ratios)), [f"{r:.1f}" for r in unique_ratios])
    plt.xlabel('CV Ratio')
    plt.ylabel('Success Rate (%)')
    plt.title('Calculation Success Rate by CV Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_methods_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Starting comparison of parameter finding methods...")
    results = compare_parameter_finding_methods()
    display_results(results)
