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
# Import all the functions from the 'src' directory, we import all the functions from each module so we can use them straight away
from ssa_simulation import *
from ssa_analysis import *
from ssa_classification import *
from models.MLP import MLP 

############# READ THIS ################
# I already ran var_v_accuracy_plot.py previously, this script is only for running extra classifier benchmarking experiments 
# I saved the SSA time series in data/mRNA_trajectories_variance_{int(variance_target_stress)}_{int(variance_target_normal)}/m_traj_{variance_target_stress}_{variance_target_normal}_{i}.csv 
# Previous results were saved in data/accuracy_results.csv
############# READ THIS ################

# Define system parameters
variance_target_normal = 1200.0
variance_ratios = np.arange(0.1, 3.0, 0.01)

# Define activation/deactivation rates for two conditions
sigma_u_stress, sigma_b_stress = 18.0, 0.01  # Stressed Condition
sigma_u_normal, sigma_b_normal = 9.0, 0.02   # Normal Condition
d_stress = d_normal = 1 # Define degradation rates

# Iterate over different variance ratios
for ratio in tqdm.tqdm(variance_ratios, desc="Running Classifier Benchmarks"):
    variance_target_stress = ratio * variance_target_normal  # Adjust stress variance based on ratio

    # Compute transcription rates (rho) for both conditions
    rho_stress = - (sigma_b_stress + sigma_u_stress) * np.sqrt(d_stress + sigma_b_stress + sigma_u_stress) * (
        np.sqrt(sigma_b_stress) * np.sqrt(d_stress + sigma_b_stress + sigma_u_stress) -
        np.sqrt(sigma_b_stress * (sigma_b_stress + sigma_u_stress) + d_stress * (sigma_b_stress + 4 * variance_target_stress * sigma_u_stress))
    ) / (2 * np.sqrt(sigma_b_stress) * sigma_u_stress)

    rho_normal = - (sigma_b_normal + sigma_u_normal) * np.sqrt(d_normal + sigma_b_normal + sigma_u_normal) * (
        np.sqrt(sigma_b_normal) * np.sqrt(d_normal + sigma_b_normal + sigma_u_normal) -
        np.sqrt(sigma_b_normal * (sigma_b_normal + sigma_u_normal) + d_normal * (sigma_b_normal + 4 * variance_target_normal * sigma_u_normal))
    ) / (2 * np.sqrt(sigma_b_normal) * sigma_u_normal)

    # Update parameter sets
    parameter_sets = [
        {"sigma_u": sigma_u_stress, "sigma_b": sigma_b_stress, "rho": rho_stress, "d": d_stress, "label": 0},
        {"sigma_u": sigma_u_normal, "sigma_b": sigma_b_normal, "rho": rho_normal, "d": d_normal, "label": 1},
    ]

    # set the number of iterations to run for the simulation
    num_iterations = 10

    for i in range(num_iterations):
        # Previously generated SSA dataset were saved here, so all we need to do is load the dataset and run the classifiers
        output_dir = f"data/mRNA_trajectories_variance_{int(variance_target_stress)}_{int(variance_target_normal)}"
        output_file = f"{output_dir}/m_traj_{variance_target_stress}_{variance_target_normal}_{i}.csv"
        df_results = pd.read_csv(output_file)

        # Extract mRNA trajectories
        stress_trajectories = df_results[df_results['label'] == 0].iloc[:, 1:].values  
        normal_trajectories = df_results[df_results['label'] == 1].iloc[:, 1:].values  

        # Statistical report
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
            "Parameter Sets": parameter_sets,
            "Stats": stats,
            "Variance Ratio": ratio,
            "SVM (rbf) Accuracy": svm_rbf_accuracy,
            "SVM (linear) Accuracy": svm_linear_accuracy,
            "Random Forest Accuracy": rf_accuracy,
            "Logistic Regression Accuracy": log_reg_accuracy,
            "MLP Accuracy": mlp_accuracy,
            "Random Classifier Accuracy": random_accuracy
        })

        # Save the accuracy results to a csv file
        if not os.path.isfile("data/accuracy_results_05_03_2025.csv"):
            df_acc_results.to_csv("data/accuracy_results_05_03_2025.csv", index=False)
        else:
            df_acc_results.to_csv("data/accuracy_results_05_03_2025.csv", mode='a', header=False, index=False)
