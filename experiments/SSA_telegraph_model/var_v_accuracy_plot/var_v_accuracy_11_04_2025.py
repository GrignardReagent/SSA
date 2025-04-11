#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import tqdm
# from sympy import sqrt
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import sympy as sp
# from sympy import init_printing, solve
# from scipy.optimize import fsolve

# Import your own local modules/functions
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems
from stats.report import statistical_report

from utils.load_data import load_and_split_data
from utils.steady_state import save_steady_state
from simulation.mean_var_autocorr import find_parameters
from classifiers.lstm_classifier import lstm_classifier
from classifiers.lstm_conv1d_classifier import lstm_conv1d_classifier
from classifiers.svm_classifier import svm_classifier
from classifiers.random_forest_classifier import random_forest_classifier
from classifiers.logistic_regression_classifier import logistic_regression_classifier
from classifiers.mlp_classifier import mlp_classifier
from classifiers.random_classifier import random_classifier

############# READ THIS ################
# 1. After finetuning the LSTM architercture and hyperparameters, I am now running some classification benchmarks using the new model (saved in the folder 'src/classifiers/lstm_conv1d_classifier.py').: 
# hidden_size      64.000
# num_layers        2.000
# dropout_rate      0.010
# learning_rate     0.001
# batch_size       32.000
# 2. We want ONLY the steady state part of the data for the classification task, we use save_steady_state() to get the steady state part of the data.

############# READ THIS ################


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
    # extended_time_points = np.arange(
    #     time_points[0],
    #     len(time_points) + steady_state_time,
    #     time_points[1] - time_points[0]
    # )
    size = 200
    num_iterations = 10

    ###########################################################################
    # 3) Simulate & Save data
    ###########################################################################
    for i in range(num_iterations):
        # df_results = simulate_two_telegraph_model_systems(parameter_sets, time_points, size)

        output_dir = f"data_03_04_2025/mRNA_trajectories_variance_{int(variance_target_stress)}_{int(variance_target_normal)}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/m_traj_{variance_target_stress}_{variance_target_normal}_{i}.csv"
        df_results = pd.read_csv(output_file)

        # get only the steady state part of the data
        save_path = 'data_03_04_2025/steady_state_trajectories/'
        remaining_time_points, steady_state_series = save_steady_state(output_file, parameter_sets, time_points,
                                                                       save_path=save_path,)

        #######################################################################
        # 4) Analysis & classification (optional, can be done post-hoc)
        #######################################################################
        stress_trajectories = steady_state_series[steady_state_series['label'] == 0].iloc[:, 1:].values
        normal_trajectories = steady_state_series[steady_state_series['label'] == 1].iloc[:, 1:].values
        stats = statistical_report(parameter_sets, stress_trajectories, normal_trajectories)

        # read in the steady state data
        steady_state_file = os.path.join(save_path, f"{os.path.splitext(os.path.basename(output_file))[0]}_SS.csv") # this is the file name we saved it as

        # classifiers
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(steady_state_file, split_val_size=0.2) # we must define split_val_size here to get a validation set
        svm_rbf_accuracy = svm_classifier(X_train, X_test, y_train, y_test)
        svm_linear_accuracy = svm_classifier(X_train, X_test, y_train, y_test, svm_kernel='linear')
        rf_accuracy = random_forest_classifier(X_train, X_test, y_train, y_test)
        log_reg_accuracy = logistic_regression_classifier(X_train, X_test, y_train, y_test)
        mlp_accuracy = mlp_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=100)
        random_accuracy = random_classifier(y_test)
        lstm_accuracy = lstm_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=50)
        lstm_conv1d_accuracy = lstm_conv1d_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=50)

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
            "LSTM Accuracy": [lstm_accuracy],
            "LSTM Conv1D Accuracy": [lstm_conv1d_accuracy]
        })

        # Save the accuracy results to a CSV file
        if not os.path.isfile("data_03_04_2025/accuracy_results_11_04_2025.csv"):
            df_acc_results.to_csv("data_03_04_2025/accuracy_results_11_04_2025.csv", index=False)
        else:
            df_acc_results.to_csv("data_03_04_2025/accuracy_results_11_04_2025.csv", mode='a', header=False, index=False)