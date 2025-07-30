#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import tqdm
import ast
from collections import defaultdict

# Import local modules/functions
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems
from stats.report import statistical_report
from utils.load_data import load_and_split_data
from utils.steady_state import save_steady_state
from stats.cv import calculate_cv
from stats.fano_factor import calculate_fano_factor, calculate_fano_factor_from_params
from classifiers.lstm_classifier import lstm_classifier
from classifiers.svm_classifier import svm_classifier
from classifiers.random_forest_classifier import random_forest_classifier
from classifiers.logistic_regression_classifier import logistic_regression_classifier
from classifiers.mlp_classifier import mlp_classifier
from classifiers.random_classifier import random_classifier
from classifiers.transformer_classifier import transformer_classifier


#############################################################################
# Load Parameters from IY007A.csv
#############################################################################

# Create data directory if it doesn't exist
os.makedirs('/home/ianyang/stochastic_simulations/experiments/EXP-25-IY007/data', exist_ok=True)

# Read the IY007A.csv file which contains the parameters
params_file = '/home/ianyang/stochastic_simulations/experiments/EXP-25-IY007/data/IY007A.csv'
df_params = pd.read_csv(params_file)

# Get unique CV ratios and their parameters
# Group by CV Ratio and get the first entry (assuming parameters are consistent for each ratio)
grouped_params = df_params.groupby('CV Ratio').first().reset_index()

print(f"Found {len(grouped_params)} unique CV ratios in IY007A.csv")

#############################################################################
# Benchmarking setup
# Reuse the best hyperparameters from IY001
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

#############################################################################
# Load the best transformer configuration from IY006B
#############################################################################
# Read in the transformer finetuning results
df_transformer_results = pd.read_csv("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY006/data/IY006B.csv")

# Sort by test accuracy
df_transformer_results = df_transformer_results.sort_values(by='test_acc', ascending=False)

# Get the best transformer parameters
best_transformer_params = df_transformer_results.iloc[0]
print("Best transformer configuration from IY006B:")
print(f"d_model: {best_transformer_params['d_model']}, nhead: {best_transformer_params['nhead']}, "
      f"num_layers: {best_transformer_params['num_layers']}, dropout: {best_transformer_params['dropout_rate']}, "
      f"learning_rate: {best_transformer_params['learning_rate']}, batch_size: {best_transformer_params['batch_size']}, "
      f"use_conv1d: {best_transformer_params['use_conv1d']}, use_auxiliary: {best_transformer_params['use_auxiliary']}")


###############################################################################
# Simulation and Classification
###############################################################################

# Define simulation parameters (consistent with IY007_1.py)
time_points = np.arange(0, 144.0, 1.0)
size = 200
num_iterations = 10

# Create results file
results_file = "/home/ianyang/stochastic_simulations/experiments/EXP-25-IY007/data/IY007C.csv"
if os.path.exists(results_file):
    print(f"Results file {results_file} already exists. Will append to it.")

# Iterate through each CV ratio and its parameters
for _, row in tqdm.tqdm(grouped_params.iterrows(), total=len(grouped_params), desc="Processing CV ratios"):
    cv_ratio = row['CV Ratio']
    normal_cv = row['Normal CV']
    stress_cv = row['Stress CV']
    
    # Parse the parameter sets from the string representation
    parameter_sets = ast.literal_eval(row['Parameter Sets'])
    
    print(f"\nProcessing CV ratio: {cv_ratio:.2f}, Stress CV: {stress_cv:.2f}, Normal CV: {normal_cv:.2f}")
    
    # Find the minimum d value for steady state calculation
    min_d = min(pset['d'] for pset in parameter_sets)
    steady_state_time = int(10 / min_d)
    
    # Define extended time points for simulation
    extended_time_points = np.arange(
        time_points[0],
        len(time_points) + steady_state_time,
        time_points[1] - time_points[0]
    )
    
    ###########################################################################
    # Simulate & Save data
    ###########################################################################
    for i in range(num_iterations):
        df_results = simulate_two_telegraph_model_systems(parameter_sets, extended_time_points, size)

        # Create output directory with CV values
        output_dir = f"/home/ianyang/stochastic_simulations/experiments/EXP-25-IY007/data/mRNA_trajectories_cv_{stress_cv:.2f}_{normal_cv:.2f}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full time series
        output_file = f"{output_dir}/m_traj_cv_{stress_cv:.2f}_{normal_cv:.2f}_{i}.csv"
        df_results.to_csv(output_file, index=False)
        
        # Create steady state directory
        save_path = f'{output_dir}/steady_state_trajectories/'
        os.makedirs(save_path, exist_ok=True)
        
        # Get only the steady state part of the data
        remaining_time_points, steady_state_series = save_steady_state(output_file, parameter_sets, time_points,
                                                                       save_path=save_path)

        # Read in the steady state data
        steady_state_file = os.path.join(save_path, f"{os.path.splitext(os.path.basename(output_file))[0]}_SS.csv")

        #######################################################################
        # Analysis & classification
        #######################################################################
        stress_trajectories = steady_state_series[steady_state_series['label'] == 0].iloc[:, 1:].values
        normal_trajectories = steady_state_series[steady_state_series['label'] == 1].iloc[:, 1:].values
        stats = statistical_report(parameter_sets, stress_trajectories, normal_trajectories)

        # Load and split data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(steady_state_file, split_val_size=0.2)

        # Run all classifiers (identical to IY007_1.py)
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
        # Transformer with best config from IY006B
        transformer_iy006b_accuracy = transformer_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            d_model=int(best_transformer_params['d_model']), 
            nhead=int(best_transformer_params['nhead']), 
            num_layers=int(best_transformer_params['num_layers']), 
            epochs=50,
            dropout_rate=float(best_transformer_params['dropout_rate']),
            learning_rate=float(best_transformer_params['learning_rate']),
            batch_size=int(best_transformer_params['batch_size']),
            use_conv1d=bool(best_transformer_params['use_conv1d']),
            use_auxiliary=bool(best_transformer_params['use_auxiliary']),
            gradient_clip=float(best_transformer_params['gradient_clip']),
        )
        random_accuracy = random_classifier(y_test)

        # Record results
        df_acc_results = pd.DataFrame({
            "Parameter Sets": [parameter_sets],
            "Stats": [stats],
            "CV Ratio": [cv_ratio],
            "Normal CV": [normal_cv],
            "Stress CV": [stress_cv],
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
            "IY006B Transformer Accuracy": [transformer_iy006b_accuracy],
        })

        # Save results
        if not os.path.isfile(results_file):
            df_acc_results.to_csv(results_file, index=False)
        else:
            df_acc_results.to_csv(results_file, mode='a', header=False, index=False)

        print(f"Completed iteration {i+1}/{num_iterations} for CV ratio {cv_ratio:.2f}")

print("\nSimulation and benchmarking complete. Results saved to:", results_file)
