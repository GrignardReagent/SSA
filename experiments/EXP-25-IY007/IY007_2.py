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
from utils.cv import calculate_cv
from utils.fano_factor import calculate_fano_factor, calculate_fano_factor_from_params
from simulation.mean_var_autocorr import find_biological_variance_mean, check_biological_appropriateness
from simulation.mean_cv_autocorr import quick_find_parameters
from classifiers.lstm_classifier import lstm_classifier
from classifiers.svm_classifier import svm_classifier
from classifiers.random_forest_classifier import random_forest_classifier
from classifiers.logistic_regression_classifier import logistic_regression_classifier
from classifiers.mlp_classifier import mlp_classifier
from classifiers.random_classifier import random_classifier
from classifiers.transformer_classifier import transformer_classifier

###############################################################################
# Generate Synthetic Data
# 1) Define target mean, CV and autocorrelations, and some parameters to start with
###############################################################################

# Set initial parameters based on our testing in the notebook
target_cv_normal = 0.5  # initial CV for normal condition
mu_target = 20
autocorr_target = 1     # Autocorrelation time (same for both conditions)

# Try to find parameters that work for all CV ratios
print("Finding parameters where all CV ratios are biologically appropriate...")
# Define a narrower CV ratio range based on testing
cv_ratio = np.arange(0.5, 2.0, 0.01)  # Narrower range that's more likely to be biologically appropriate

# Initial parameters
parameters = {
    "stress": {"sigma_b": 2.0},
    "normal": {"sigma_b": 1.0}
}

print(f"\nStarting simulations with:")
print(f"Normal CV: {target_cv_normal:.2f}, Mean: {mu_target:.2f}, Autocorrelation Time: {autocorr_target:.2f}")

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
    variance_normal = (target_cv_normal * mu_target)**2
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
            rho, sigma_u, d = quick_find_parameters(
                param.get('sigma_b'), # Pass the sigma_b value from the param dictionary, not the whole dictionary 
                mu_target=mu_target, autocorr_target=autocorr_target, 
                cv_target=cv_for_condition
            )

            results[condition] = {"rho": rho, "sigma_u": sigma_u, "d": d}
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
        {"sigma_u": results["stress"]["sigma_u"], 
        "sigma_b": parameters["stress"]['sigma_b'], 
        "rho": results["stress"]['rho'], 
        "d": results["stress"]['d'], 
        "label": 0},
        
        {"sigma_u": results["normal"]["sigma_u"], 
        "sigma_b": parameters["normal"]['sigma_b'], 
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
        # Load the steady state data with explicit check for duplicate columns
        steady_state_series = pd.read_csv(steady_state_file)
        
        # Check for duplicate column names and handle them
        if len(steady_state_series.columns) != len(set(steady_state_series.columns)):
            print("Warning: Duplicate column names detected in steady state data")
            # Create a dictionary mapping original column names to unique ones
            col_map = {}
            for col in steady_state_series.columns:
                if col in col_map:
                    count = col_map[col]["count"] + 1
                    col_map[col]["count"] = count
                    col_map[col]["names"].append(f"{col}_{count}")
                else:
                    col_map[col] = {"count": 0, "names": [col]}
            
            # Create a flat list of new column names
            new_columns = []
            for col in steady_state_series.columns:
                new_columns.append(col_map[col]["names"].pop(0))
            
            # Rename columns
            steady_state_series.columns = new_columns
        
        # Make sure 'label' is correctly formatted
        if 'label' in steady_state_series.columns:
            stress_trajectories = steady_state_series[steady_state_series['label'] == 0].iloc[:, 1:].values
            normal_trajectories = steady_state_series[steady_state_series['label'] == 1].iloc[:, 1:].values
        else:
            # If 'label' is missing, look for a column that might contain label information
            # This assumes labels are in the first column if not explicitly named 'label'
            label_col = steady_state_series.columns[0]
            print(f"'label' column not found, using '{label_col}' as label column")
            stress_trajectories = steady_state_series[steady_state_series[label_col] == 0].iloc[:, 1:].values
            normal_trajectories = steady_state_series[steady_state_series[label_col] == 1].iloc[:, 1:].values
        
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
        # Train and evaluate model using best hyperparams from IY001A
        iy001a_lstm_accuracy = lstm_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test,
            epochs=100,
            hidden_size=128,
            num_layers=4,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=64,
            use_conv1d=True,
            use_attention=True,
            num_attention_heads=4,
            use_auxiliary=True,
        )
        # IY006C-Transformer benchmark - using best configuration from IY006C
        iy006c_transformer_accuracy = transformer_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            d_model=128, nhead=4, num_layers=2, 
            dropout_rate=0.1, learning_rate=0.01, batch_size=64,
            epochs=50, pooling_strategy='last', 
            use_conv1d=True, use_auxiliary=False,
            gradient_clip=1.0
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
            "IY006C-Transformer Accuracy": [iy006c_transformer_accuracy],
            "Vanilla Transformer Accuracy": [transformer_accuracy],
            "Full Transformer Accuracy": [transformer_full_accuracy],
        })

        # Save results
        results_file = "data/IY007D.csv"
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
