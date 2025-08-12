#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import tqdm

# Import local modules/functions
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems
from stats.report import statistical_report
from utils.load_data import load_and_split_data
from utils.steady_state import save_steady_state
from stats.cv import calculate_cv
from stats.fano_factor import calculate_fano_factor, calculate_fano_factor_from_params
from simulation.mean_var_autocorr import find_parameters
from utils.biological import find_biological_variance_mean, check_biological_appropriateness
from classifiers.lstm_classifier import lstm_classifier
from classifiers.svm_classifier import svm_classifier
from classifiers.random_forest_classifier import random_forest_classifier
from classifiers.logistic_regression_classifier import logistic_regression_classifier
from classifiers.mlp_classifier import mlp_classifier
from classifiers.random_classifier import random_classifier
from classifiers.transformer_classifier import transformer_classifier


#############################################################################
# Benchmarking
# 1) Use the best hyperparameters from IY001
#############################################################################
# read in the finetuning results
df_finetune_results = pd.read_csv("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY001/data/IY001A.csv")

df_finetune_results = df_finetune_results.sort_values(by='test_acc', ascending=False)

# get the best parameters
best_params = df_finetune_results.iloc[0][
    ['architecture', 'hidden_size', 'num_layers', 'dropout_rate', 'learning_rate', 'batch_size', 'epochs']
    ]

architectures = [
    {"name": "Vanilla LSTM", "conv1d": False, "attention": False, "multihead": False, "aux": False},
    {"name": "Conv1D Only", "conv1d": True,  "attention": False, "multihead": False, "aux": False},
    {"name": "Conv1D + Attention", "conv1d": True,  "attention": True, "multihead": False, "aux": False},
    {"name": "Conv1D + MultiHead", "conv1d": True,  "attention": True, "multihead": True,  "aux": False},
    {"name": "Full Model", "conv1d": True,  "attention": True, "multihead": True,  "aux": True},
]

# map the best architecture name to its config
best_arch_name = best_params['architecture']
best_arch_config = next((arch for arch in architectures if arch['name'] == best_arch_name), None)

if best_arch_config is None:
    raise ValueError(f"Architecture '{best_arch_name}' not found in predefined architectures.")

###############################################################################
# Generate Synthetic Data
# 1) Define target mean, variance and autocorrelations, and some parameters to start with
###############################################################################

# Define a function to try different values
def find_biologically_appropriate_parameters(target_cv_initial=1.7, mu_target_initial=5, max_attempts=10):
    # Try different combinations of CV and mean until we find biologically appropriate values
    for attempt in range(max_attempts):
        # On first attempt, use the initial values
        if attempt == 0:
            target_cv = target_cv_initial
            mu = mu_target_initial
        # On subsequent attempts, adjust the values
        else:
            # Try reducing CV or increasing mean to improve Fano factor and CV
            target_cv = target_cv_initial * (0.9 - attempt * 0.05)  # Gradually reduce CV
            mu = mu_target_initial * (1.1 + attempt * 0.1)         # Gradually increase mean
        
        # Calculate corresponding variance
        variance = (target_cv * mu)**2
        
        print(f"\nAttempt {attempt+1}/{max_attempts}")
        print(f"Testing: CV = {target_cv:.2f}, Mean = {mu:.2f}, Variance = {variance:.2f}")
        
        # Check if these values are biologically appropriate
        is_appropriate = check_biological_appropriateness(variance, mu)
        
        if is_appropriate:
            print(f"✅ Found biologically appropriate values: CV = {target_cv:.2f}, Mean = {mu:.2f}, Variance = {variance:.2f}")
            return target_cv, mu, variance
    
    # If we reach here, we couldn't find appropriate parameters
    print("❌ Could not find biologically appropriate parameters")
    return None, None, None

# Function to pre-check all CV ratios for biological appropriateness
def find_parameters_without_skipping(initial_cv_normal=1.7, initial_mu_target=5, cv_ratios=np.arange(0.1, 3.0, 0.01), max_attempts=5):
    for attempt in range(max_attempts):
        # On first attempt, use the initial values
        if attempt == 0:
            cv_normal = initial_cv_normal
            mu = initial_mu_target
        # On subsequent attempts, adjust the values
        else:
            # Try decreasing CV and increasing mean to improve all conditions
            cv_normal = initial_cv_normal * (0.85 - attempt * 0.05)  # More aggressive CV reduction
            mu = initial_mu_target * (1.3 + attempt * 0.2)          # More aggressive mean increase
        
        # Calculate normal variance
        variance_normal = (cv_normal * mu)**2
        
        print(f"\nTesting parameters: CV normal = {cv_normal:.2f}, Mean = {mu:.2f}")
        
        # Check if normal condition is biologically appropriate
        is_normal_appropriate = check_biological_appropriateness(variance_normal, mu)
        if not is_normal_appropriate:
            print(f"⚠️ Normal condition not biologically appropriate. Trying different parameters...")
            continue
        
        # Check all CV ratios for the stress condition
        all_appropriate = True
        inappropriate_ratios = []
        
        for ratio in tqdm.tqdm(cv_ratios, desc="Checking all CV ratios"):
            cv_stress = ratio * cv_normal
            variance_stress = (cv_stress * mu)**2
            
            is_stress_appropriate = check_biological_appropriateness(variance_stress, mu)
            if not is_stress_appropriate:
                all_appropriate = False
                inappropriate_ratios.append(ratio)
        
        if all_appropriate:
            print(f"✅ Found parameters where all CV ratios are biologically appropriate!")
            print(f"CV normal = {cv_normal:.2f}, Mean = {mu:.2f}, Variance normal = {variance_normal:.2f}")
            return cv_normal, mu, variance_normal
        else:
            print(f"⚠️ {len(inappropriate_ratios)} out of {len(cv_ratios)} CV ratios are not biologically appropriate")
            print(f"First few inappropriate ratios: {inappropriate_ratios[:5]}")
    
    print("❌ Could not find parameters where all CV ratios are biologically appropriate")
    print("You may need to adjust the range of CV ratios or try more extreme parameter values")
    return None, None, None

# Set initial parameters based on our testing in the notebook
initial_cv_normal = 0.8  # Try a lower initial CV for normal condition
initial_mu_target = 15   # Try a higher initial mean
autocorr_target = 0.5    # Autocorrelation (same for both conditions)

# Try to find parameters that work for all CV ratios
print("Finding parameters where all CV ratios are biologically appropriate...")
# Define a narrower CV ratio range based on testing
cv_ratio = np.arange(0.5, 2.0, 0.01)  # Narrower range that's more likely to be biologically appropriate

target_cv_normal, mu_target, variance_normal = find_parameters_without_skipping(
    initial_cv_normal=initial_cv_normal, 
    initial_mu_target=initial_mu_target,
    cv_ratios=cv_ratio,
    max_attempts=10  # Increase max attempts to try more parameter combinations
)

# If we couldn't find parameters that work for all ratios, use the original function
if target_cv_normal is None:
    print("\nFalling back to finding biologically appropriate parameters for normal condition only...")
    target_cv_normal, mu_target, variance_normal = find_biologically_appropriate_parameters(
        target_cv_initial=initial_cv_normal, 
        mu_target_initial=initial_mu_target
    )  

# Initial parameters
parameters = {
    "stress": {"sigma_u": 18.0},
    "normal": {"sigma_u": 9.0}
}

print(f"\nStarting simulations with:")
print(f"Normal CV: {target_cv_normal:.2f}, Mean: {mu_target:.2f}, Variance: {variance_normal:.2f}")

###############################################################################
# 2) Loop over different CV ratios
###############################################################################

# Track inappropriate ratios for reporting
inappropriate_ratios = []

for ratio in tqdm.tqdm(cv_ratio, desc="Running CV Ratio Simulations"):
    # For the stress condition, we define CV by ratio
    target_cv_stress = ratio * target_cv_normal
    
    # Calculate corresponding variance values for reporting only
    variance_stress = (target_cv_stress * mu_target)**2
    
    print(f"\nTesting CV ratio: {ratio:.2f}, Stress CV: {target_cv_stress:.2f}, Normal CV: {target_cv_normal:.2f}")
    print(f"Corresponding variances - Stress: {variance_stress:.2f}, Normal: {variance_normal:.2f}")
    
    # Double-check if stress parameters are biologically appropriate
    is_stress_appropriate = check_biological_appropriateness(variance_stress, mu_target)
    
    # If stress parameters are not biologically appropriate, track it and skip
    if not is_stress_appropriate:
        print(f"⚠️ Stress parameters are not biologically appropriate for ratio {ratio:.2f}. Skipping.")
        inappropriate_ratios.append(ratio)
        continue
    
    # Store results for each condition
    results = {}
    success = True  # Track if both conditions succeed
    
    for condition, param in parameters.items():
        # Decide which CV to use for this condition
        if condition == "normal":
            cv_for_condition = target_cv_normal
        else:  # condition == "stress"
            cv_for_condition = target_cv_stress

        try:
            # Fix mean, CV, and autocorrelation
            rho, sigma_b, d = find_parameters(
                param, mu_target=mu_target, variance_target=None, autocorr_target=autocorr_target, 
                cv_target=cv_for_condition, check_biological=True
            )

            results[condition] = {"rho": rho, "sigma_b": sigma_b, "d": d}
            print(f"[{condition.upper()}] ✅ Found: {results[condition]}")
        except ValueError as e:
            print(f'{e}')
            print(f"[{condition.upper()}] ❌ No suitable solution found.")
            success = False
    
    # If any condition failed, skip to the next ratio
    if not success:
        print(f"Skipping ratio {ratio:.2f} due to parameter finding failure")
        continue


    # update parameter sets 
    parameter_sets = [
        {"sigma_u": parameters["stress"]["sigma_u"], 
        "sigma_b": results["stress"]['sigma_b'], 
        "rho": results["stress"]['rho'], 
        "d": results["stress"]['d'], 
        "label": 0},
        
        {"sigma_u": parameters["normal"]["sigma_u"], 
        "sigma_b": results["normal"]['sigma_b'], 
        "rho": results["normal"]['rho'], 
        "d": results["normal"]['d'], 
        "label": 1}
    ]

    # Output the results
    print("Updated Parameter Sets:", parameter_sets)

    # Simulation parameters
    min_d = min(pset['d'] for pset in parameter_sets)
    steady_state_time = int(10 / min_d)
    time_points = np.arange(0, 144.0, 1.0)
    extended_time_points = np.arange(
        time_points[0],
        len(time_points) + steady_state_time,
        time_points[1] - time_points[0]
    )
    size = 200
    num_iterations = 10

    ###########################################################################
    # 3) Simulate & Save data
    ###########################################################################
    for i in range(num_iterations):
        df_results = simulate_two_telegraph_model_systems(parameter_sets, extended_time_points, size)

        # Create output directory with CV values
        output_dir = f"/home/ianyang/stochastic_simulations/experiments/EXP-25-IY007/data/mRNA_trajectories_cv_{target_cv_stress:.2f}_{target_cv_normal:.2f}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full time series
        output_file = f"{output_dir}/m_traj_cv_{target_cv_stress:.2f}_{target_cv_normal:.2f}_{i}.csv"
        df_results.to_csv(output_file, index=False)
        
        # Create steady state directory
        save_path = f'{output_dir}/steady_state_trajectories/'
        os.makedirs(save_path, exist_ok=True)
        
        # Get only the steady state part of the data
        remaining_time_points, steady_state_series = save_steady_state(output_file, parameter_sets, time_points,
                                                                       save_path=save_path,)

        # Read in the steady state data
        steady_state_file = os.path.join(save_path, f"{os.path.splitext(os.path.basename(output_file))[0]}_SS.csv")

        #######################################################################
        # 4) Analysis & classification (updated to reuse IY002A model)
        #######################################################################
        stress_trajectories = steady_state_series[steady_state_series['label'] == 0].iloc[:, 1:].values
        normal_trajectories = steady_state_series[steady_state_series['label'] == 1].iloc[:, 1:].values
        stats = statistical_report(parameter_sets, stress_trajectories, normal_trajectories)

        # classifiers
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(steady_state_file, split_val_size=0.2)

        # Baseline classifiers
        svm_rbf_accuracy = svm_classifier(X_train, X_test, y_train, y_test)
        svm_linear_accuracy = svm_classifier(X_train, X_test, y_train, y_test, svm_kernel='linear')
        rf_accuracy = random_forest_classifier(X_train, X_test, y_train, y_test)
        log_reg_accuracy = logistic_regression_classifier(X_train, X_test, y_train, y_test)
        mlp_accuracy = mlp_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=100)
        # vanilla LSTM, not finetuned
        lstm_accuracy = lstm_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=50,
                                        use_conv1d=False, use_attention=False, use_auxiliary=False)
        # Train and evaluate model using best hyperparams and architecture from IY001
        iy001a_lstm_accuracy = lstm_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test,
            epochs=int(best_params['epochs']),
            hidden_size=int(best_params['hidden_size']),
            num_layers=int(best_params['num_layers']),
            dropout_rate=float(best_params['dropout_rate']),
            learning_rate=float(best_params['learning_rate']),
            batch_size=int(best_params['batch_size']),
            use_conv1d=best_arch_config['conv1d'],
            use_attention=best_arch_config['attention'],
            num_attention_heads=4 if best_arch_config['attention'] else 0,
            use_auxiliary=best_arch_config['aux'],
        )
        # Vanilla Transformer, not finetuned
        transformer_accuracy = transformer_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            d_model=64, nhead=4, num_layers=2, epochs=50,
            use_conv1d=False, use_auxiliary=False
        )
        # Transformer with Conv1D and auxiliary task, not finetuned
        transformer_full_accuracy = transformer_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            d_model=128, nhead=8, num_layers=4, epochs=50,
            use_conv1d=True, use_auxiliary=True
        )
        random_accuracy = random_classifier(y_test)

        # Record results
        df_acc_results = pd.DataFrame({
            "Parameter Sets": [parameter_sets],
            "Stats": [stats],
            "CV Ratio": [ratio],
            "Normal CV": [target_cv_normal],
            "Stress CV": [target_cv_stress],
            "SVM (rbf) Accuracy": [svm_rbf_accuracy],
            "SVM (linear) Accuracy": [svm_linear_accuracy],
            "Random Forest Accuracy": [rf_accuracy],
            "Logistic Regression Accuracy": [log_reg_accuracy],
            "MLP Accuracy": [mlp_accuracy],
            "Random Classifier Accuracy": [random_accuracy],
            "Vanilla LSTM Accuracy": [lstm_accuracy],
            "IY001A Accuracy": [iy001a_lstm_accuracy],
            "Vanilla Transformer Accuracy": [transformer_accuracy],
            "Full Transformer Accuracy": [transformer_full_accuracy],
        })

        # Save results
        results_file = "data/IY007A.csv"
        if not os.path.isfile(results_file):
            df_acc_results.to_csv(results_file, index=False)
        else:
            df_acc_results.to_csv(results_file, mode='a', header=False, index=False)

# Summarize results after all simulations
if len(inappropriate_ratios) == 0:
    print("\n✅ All CV ratios were biologically appropriate! No iterations were skipped.")
else:
    print(f"\n⚠️ {len(inappropriate_ratios)} out of {len(cv_ratio)} CV ratios were not biologically appropriate")
    print(f"Inappropriate ratios: {inappropriate_ratios[:10]}{'...' if len(inappropriate_ratios) > 10 else ''}")
    
    # Log inappropriate ratios for future reference
    inappropriate_file = "data/inappropriate_ratios.csv"
    pd.DataFrame({
        "Inappropriate Ratios": inappropriate_ratios,
        "CV Normal": target_cv_normal,
        "Mean": mu_target
    }).to_csv(inappropriate_file, index=False)
    
    print(f"Inappropriate ratios saved to {inappropriate_file}")
