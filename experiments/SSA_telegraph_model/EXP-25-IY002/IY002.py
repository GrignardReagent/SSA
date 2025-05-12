#!/usr/bin/env python3

import pandas as pd
import glob
from utils.load_data import load_and_split_data
from models.lstm import LSTMClassifier
from classifiers.lstm_classifier import lstm_classifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils.set_seed import set_seed


##########################################################################
# 1) Combine all steady state data from IY004, 1 sample from each variance ratio
##########################################################################

# path to all *steady state* CSV files, for simplicity we only take the first set of steady state data ending with 0_SS.csv
file_paths = sorted(glob.glob('/home/ianyang/stochastic_simulations/experiments/SSA_telegraph_model/var_v_accuracy_plot/data_12_04_2025/mRNA_trajectories_variance_*/steady_state_trajectories/m_traj_*_0_SS.csv')) 
# len(file_paths)

# Read and combine
dfs = [pd.read_csv(f) for f in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: shuffle the rows
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
input_file_path = 'data/IY002_input_A.csv'
combined_df.to_csv(input_file_path, index=False)
print(f"✅ Combined {len(file_paths)} files into {combined_df.shape[0]} rows.")

#############################################################################
# 2) Load the combined data and split into train, validation, and test sets, and use the best hyperparameters from IY001
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

# Train LSTM model using SSA data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(input_file_path, split_val_size=0.2) # we must define split_val_size here to get a validation set

############################################################
# 3) Train and evaluate the model using the best hyperparameters and architecture, then save the model as IY002A.pth
############################################################

# Train and evaluate model using best hyperparams and architecture
lstm_accuracy = lstm_classifier(
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
    save_path='IY002A.pth'
)

#####################################################################
# 4) Load the IY002A model and prepare the data for evaluation (much later on)
#####################################################################

set_seed(42)  # reproducibility
input_size = 1  # each time step is a single value
output_size = len(set(y_train))
    
# === Standardize ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# === Reshape for LSTM: (batch, seq_len, features) ===
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], input_size))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], input_size))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], input_size))

# === Convert to tensors and loaders ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

# recreate the exact model architecture that was used when saving IY002A.pth
IY002A_model = LSTMClassifier(
    input_size=1,    
    hidden_size=int(best_params['hidden_size']),
    num_layers=int(best_params['num_layers']),
    output_size=len(set(y_train)),             
    dropout_rate=float(best_params['dropout_rate']),
    learning_rate=float(best_params['learning_rate']),
    use_conv1d=best_arch_config['conv1d'],
    use_attention=best_arch_config['attention'],
    num_attention_heads=4 if best_arch_config['attention'] else 0,
    use_auxiliary=best_arch_config['aux']
)
IY002A_model.load_model('IY002A.pth')

# -----------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------

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
        random_accuracy = random_classifier(y_test)

        # === Preprocess for IY002A ===
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        input_size = 1
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], input_size))
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        from utils.evaluate import evaluate_model 
        IY002A_model.eval()
        IY002A_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
        iy002a_accuracy = evaluate_model(IY002A_model, test_loader, output_size=len(set(y_train)))

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
            "IY002A Accuracy": [iy002a_accuracy],
        })

        # Save results
        results_file = "data/IY002A.csv"
        if not os.path.isfile(results_file):
            df_acc_results.to_csv(results_file, index=False)
        else:
            df_acc_results.to_csv(results_file, mode='a', header=False, index=False)
