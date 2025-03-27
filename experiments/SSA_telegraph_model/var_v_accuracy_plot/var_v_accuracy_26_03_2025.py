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
from scipy.optimize import fsolve

# Import your own local modules/functions
from ssa_simulation import simulate_two_telegraph_model_systems
from ssa_analysis import statistical_report
from ssa_classification import *
from models.MLP import MLP
from utils.load_data import load_and_split_data


###############################################################################
# 1) Symbolic definitions for mean & variance
###############################################################################
rho, mu, sigma_sq, d, sigma_u, sigma_b = sp.symbols(
    'rho mu sigma_sq d sigma_u sigma_b', real=True, positive=True
)
init_printing(use_unicode=True)

# Define the variance and mean equations
def equations(vars, sigma_u, sigma_b, d, mu_target, variance_target):
    rho = vars[0]
    sigma_b = vars[1]

    # Mean equation
    mean_eqn = sigma_b * rho / (d * (sigma_b + sigma_u))

    # Variance equation
    variance_eqn = (sigma_b * rho / (d * (sigma_b + sigma_u))) + \
                     ((sigma_u * sigma_b) * rho**2 / (d * (sigma_b + sigma_u + d) * (sigma_u + sigma_b)**2))

    # Define the equations to be solved
    eq1 = mean_eqn - mu_target
    eq2 = variance_eqn - variance_target

    return [eq1, eq2]

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

    # Store positive rho values for both conditions
    rho_values = {}
    sigma_b_values = {}

    for condition, param_set in parameters.items():
        # Decide which variance to use for this condition
        if condition == "normal":
            var_for_condition = variance_target_normal
        else:  # condition == "stress"
            var_for_condition = variance_target_stress
    
        # Initial guess for rho and sigma_b, this is important to finding a solution that satisfies the equations
        rho_ig = np.arange(1, 10000, 10)
        sigma_b_ig = np.arange(1, 10000, 10)
        initial_guesses = [[rho, sigma_b] for rho in rho_ig for sigma_b in sigma_b_ig]

        for initial_guess in initial_guesses:
            solution = fsolve(equations, initial_guess, args=(param_set['sigma_u'], param_set['sigma_b'], param_set['d'], mu_target, var_for_condition))
            solved_equations = equations(solution, param_set['sigma_u'], param_set['sigma_b'], param_set['d'], mu_target, var_for_condition)

            # Check if the solved equation is close to zero
            if -1e-6 < solved_equations[0] < 1e-6 and -1e-6 < solved_equations[1] < 1e-6:
                # Check if the solution is positive
                if solution[0] > 0 and solution[1] > 0:
                    print('Positive solution found with different initial guesses.')
                    print(f"Solution for {condition} condition: {solution}")
                    rho_values[condition] = solution[0]
                    sigma_b_values[condition] = solution[1]
                    #DEBUG
                    print(f'Solution: {solution}, Solved equation: {solved_equations}')

                    break

    # Updated Parameter Sets using the calculated rho values
    parameter_sets = [
        {"sigma_u": parameters["stress"]["sigma_u"],
        "sigma_b": sigma_b_values["stress"],
        "rho": rho_values["stress"],
        "d": parameters["stress"]["d"],
        "label": 0},

        {"sigma_u": parameters["normal"]["sigma_u"],
        "sigma_b": sigma_b_values["normal"],
        "rho": rho_values["normal"],
        "d": parameters["normal"]["d"],
        "label": 1}
    ]
    # Output the results
    # print("Calculated rho values:", rho_values)
    # print("Calculated sigma_b values:", sigma_b_values)
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
    # 4) Simulate & Save data
    ###########################################################################
    for i in range(num_iterations):
        df_results = simulate_two_telegraph_model_systems(parameter_sets, time_points, size)

        output_dir = f"data_26_03_2025/mRNA_trajectories_variance_{int(variance_target_stress)}_{int(variance_target_normal)}"
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
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(output_file, split_val_size=0.2) # we must define split_val_size here to get a validation set
        svm_rbf_accuracy = svm_classifier(X_train, X_test, y_train, y_test)
        svm_linear_accuracy = svm_classifier(X_train, X_test, y_train, y_test, svm_kernel='linear')
        rf_accuracy = random_forest_classifier(X_train, X_test, y_train, y_test)
        log_reg_accuracy = logistic_regression_classifier(X_train, X_test, y_train, y_test)
        mlp_accuracy = mlp_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=100)
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
        if not os.path.isfile("data_26_03_2025/accuracy_results_26_03_2025.csv"):
            df_acc_results.to_csv("data_26_03_2025/accuracy_results_26_03_2025.csv", index=False)
        else:
            df_acc_results.to_csv("data_26_03_2025/accuracy_results_26_03_2025.csv", mode='a', header=False, index=False)
