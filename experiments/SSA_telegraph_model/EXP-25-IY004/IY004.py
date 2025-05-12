#!/usr/bin/env python3

import pandas as pd
from utils.load_data import load_and_split_data
from classifiers.lstm_classifier import lstm_classifier

#############################################################################
# 1) Use the best hyperparameters from IY001
#############################################################################

# read in the finetuning results
df_finetune_results = pd.read_csv("data/IY001A.csv")
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

# -----------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import tqdm

# Import your own local modules/functions
from stats.report import statistical_report
from utils.load_data import load_and_split_data
from simulation.mean_var_autocorr import find_parameters
from classifiers.lstm_classifier import lstm_classifier
from classifiers.svm_classifier import svm_classifier
from classifiers.random_forest_classifier import random_forest_classifier
from classifiers.logistic_regression_classifier import logistic_regression_classifier
from classifiers.mlp_classifier import mlp_classifier
from classifiers.random_classifier import random_classifier

###############################################################################
# 1) Define target mean, variance and autocorrelations, and some parameters to start with
###############################################################################
variance_target_normal = 1200.0  # Fixed normal variance
mu_target = 10.0                 # Mean (same for both)
variance_ratios = np.arange(0.1, 3.0, 0.01)
autocorr_target = 0.5

parameters = {
    "stress": {"sigma_u": 18.0},
    "normal": {"sigma_u": 9.0}
}

###############################################################################
# 2) Loop over different variance ratios
###############################################################################
for ratio in tqdm.tqdm(variance_ratios, desc="Running Variance Ratio Simulations"):
    # For the stress condition, we define variance_target_stress by ratio
    variance_target_stress = ratio * variance_target_normal

    # store results for each condition
    results = {}
    for condition, param in parameters.items():
        # Decide which variance to use for this condition
        if condition == "normal":
            var_for_condition = variance_target_normal
        else:  # condition == "stress"
            var_for_condition = variance_target_stress

        try:
            # Fix all three statistical properties
            rho, sigma_b, d = find_parameters(
                param, mu_target, var_for_condition, autocorr_target
            )

            results[condition] = {"rho": rho, "sigma_b": sigma_b, "d": d}
            print(f"[{condition.upper()}] ✅ Found: {results[condition]}")
        except ValueError as e:
            print(f'{e}')
            print(f"[{condition.upper()}] ❌ No suitable solution found.")


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

        output_dir = f"/home/ianyang/stochastic_simulations/experiments/SSA_telegraph_model/var_v_accuracy_plot/data_12_04_2025/mRNA_trajectories_variance_{int(variance_target_stress)}_{int(variance_target_normal)}"
        os.makedirs(output_dir, exist_ok=True)
        # save full time series
        output_file = f"{output_dir}/m_traj_{variance_target_stress}_{variance_target_normal}_{i}.csv"
        
        # get only the steady state part of the data
        steady_state_path = f'{output_dir}/steady_state_trajectories/'
        
        # read in the steady state data
        steady_state_file = os.path.join(steady_state_path, f"{os.path.splitext(os.path.basename(output_file))[0]}_SS.csv")
        steady_state_series = pd.read_csv(steady_state_file)

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
        # vinilla LSTM, not finetuned
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
        random_accuracy = random_classifier(y_test)

        # Record results
        df_acc_results = pd.DataFrame({
            "Parameter Sets": [parameter_sets],
            "Stats": [stats],
            "Variance Ratio": [ratio],
            "SVM (rbf) Accuracy": [svm_rbf_accuracy],
            "SVM (linear) Accuracy": [svm_linear_accuracy],
            "Random Forest Accuracy": [rf_accuracy],
            "Logistic Regression Accuracy": [log_reg_accuracy],
            "MLP Accuracy": [mlp_accuracy],
            "Random Classifier Accuracy": [random_accuracy],
            "Vanilla LSTM Accuracy": [lstm_accuracy],
            "IY001A Accuracy": [iy001a_lstm_accuracy],
        })

        # Save results
        results_file = "data/IY004A.csv"
        if not os.path.isfile(results_file):
            df_acc_results.to_csv(results_file, index=False)
        else:
            df_acc_results.to_csv(results_file, mode='a', header=False, index=False)
