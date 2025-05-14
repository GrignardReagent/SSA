#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm
from sympy import sqrt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sympy as sp
from sympy import init_printing, solve

# Import your own local modules/functions
from ssa_simulation import simulate_two_telegraph_model_systems
from ssa_analysis import statistical_report
from ssa_classification import *
from models.MLP import MLP


###############################################################################
# 1) Symbolic definitions for mean & variance
###############################################################################
rho, mu, sigma_sq, d, sigma_u, sigma_b = sp.symbols(
    'rho mu sigma_sq d sigma_u sigma_b', real=True, positive=True
)
init_printing(use_unicode=True)

# mean_eq = mu - (sigma_b * rho / [d*(sigma_b + sigma_u)]) = 0
mean_eq = mu - (
    sigma_b * rho / (d * (sigma_b + sigma_u))
)

# variance_eq = sigma_sq - [ M1 + M2 ] = 0
# M1 = sigma_b*rho / [d*(sigma_b + sigma_u)]
# M2 = (sigma_u*sigma_b)*rho^2 / [d*(sigma_b + sigma_u + d)*(sigma_b + sigma_u)^2]
variance_eq = sigma_sq - (
    (sigma_b * rho) / (d * (sigma_b + sigma_u))
    + (sigma_u * sigma_b) * rho**2 / (
        d * (sigma_b + sigma_u + d) * (sigma_b + sigma_u)**2
    )
)

# Solve the single equation [variance_eq - mean_eq=0]
solutions = solve([variance_eq - mean_eq], rho, dict=True)

# print("Symbolic solutions for rho (variance - mean = 0):")
# for s in solutions:
#     print("  rho =", s[rho])


###############################################################################
# 2) Define parameters
###############################################################################
variance_target_normal = 1200.0  # Fixed normal variance
mu_target = 10.0                 # Mean (same for both)
variance_ratios = np.arange(0.1, 3.0, 0.01)

parameters = {
    "stress": {"sigma_u": 18.0, "sigma_b": 0.01, "d": 1.0},
    "normal": {"sigma_u": 9.0,  "sigma_b": 0.02, "d": 1.0}
}


###############################################################################
# 3) Loop over different variance ratios
###############################################################################
for ratio in tqdm.tqdm(variance_ratios, desc="Running Variance Ratio Simulations"):
    # For the stress condition, we define variance_target_stress by ratio
    variance_target_stress = ratio * variance_target_normal

    # We'll solve for rho in each condition separately,
    # picking the first positive root from 'solutions'.
    rho_values = {}
    for condition, param_set in parameters.items():

        # Decide which variance to use for this condition
        if condition == "normal":
            var_for_condition = variance_target_normal
        else:  # condition == "stress"
            var_for_condition = variance_target_stress

        # Attempt each symbolic solution in 'solutions'
        rho_value_positive = None
        for sol_dict in solutions:
            candidate_rho = sol_dict[rho].subs({
                sigma_u: param_set["sigma_u"],
                sigma_b: param_set["sigma_b"],
                d      : param_set["d"],
                mu     : mu_target,
                sigma_sq: var_for_condition
            })
            if candidate_rho > 0:
                rho_value_positive = candidate_rho
                break

        if rho_value_positive is None:
            raise ValueError(
                f"No positive rho found for condition={condition} "
                f"with variance={var_for_condition}"
            )

        rho_values[condition] = rho_value_positive

    # Now build the final parameter_sets with solved rho
    parameter_sets = [
        {
            "sigma_u": parameters["stress"]["sigma_u"],
            "sigma_b": parameters["stress"]["sigma_b"],
            "rho":     rho_values["stress"],
            "d":       parameters["stress"]["d"],
            "label":   0
        },
        {
            "sigma_u": parameters["normal"]["sigma_u"],
            "sigma_b": parameters["normal"]["sigma_b"],
            "rho":     rho_values["normal"],
            "d":       parameters["normal"]["d"],
            "label":   1
        }
    ]

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
    # 4) Simulate & Save data
    ###########################################################################
    for i in range(num_iterations):
        df_results = simulate_two_telegraph_model_systems(parameter_sets, time_points, size)

        output_dir = f"data_22_03_2025/mRNA_trajectories_variance_{int(variance_target_stress)}_{int(variance_target_normal)}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/m_traj_{variance_target_stress}_{variance_target_normal}_{i}.csv"
        df_results.to_csv(output_file, index=False)

        #######################################################################
        # 5) Analysis & classification (optional, can be done post-hoc)
        #######################################################################
        stress_trajectories = df_results[df_results['label'] == 0].iloc[:, 1:].values
        normal_trajectories = df_results[df_results['label'] == 1].iloc[:, 1:].values
        stats = statistical_report(parameter_sets, stress_trajectories, normal_trajectories)

        # classifiers
        X_train, X_test, y_train, y_test = load_and_split_data(output_file)
        svm_rbf_accuracy = svm_classifier(X_train, X_test, y_train, y_test)
        svm_linear_accuracy = svm_classifier(X_train, X_test, y_train, y_test, svm_kernel='linear')
        rf_accuracy = random_forest_classifier(X_train, X_test, y_train, y_test)
        log_reg_accuracy = logistic_regression_classifier(X_train, X_test, y_train, y_test)
        mlp_accuracy = mlp_classifier(X_train, X_test, y_train, y_test, epochs=100)
        random_accuracy = random_classifier(y_test)

        df_acc_results = pd.DataFrame({
            "Parameter Sets": [parameter_sets],
            "Stats": [stats],
            "Variance Ratio": [ratio],
            "SVM (rbf) Accuracy": [svm_rbf_accuracy],
            "SVM (linear) Accuracy": [svm_linear_accuracy],
            "Random Forest Accuracy": [rf_accuracy],
            "Logistic Regression Accuracy": [log_reg_accuracy],
            "MLP Accuracy": [mlp_accuracy],
            "Random Classifier Accuracy": [random_accuracy]
        })

        # Save the accuracy results to a CSV file
        if not os.path.isfile("data_22_03_2025/accuracy_results_22_03_2025.csv"):
            df_acc_results.to_csv("data_22_03_2025/accuracy_results_22_03_2025.csv", index=False)
        else:
            df_acc_results.to_csv("data_22_03_2025/accuracy_results_22_03_2025.csv", mode='a', header=False, index=False)
