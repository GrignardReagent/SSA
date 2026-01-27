from pathlib import Path
import numpy as np
import pandas as pd

# ml
import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerClassifier
from training.eval import evaluate_model
from training.train import train_model 
from classifiers.svm_classifier import svm_classifier
from sklearn.svm import SVC

# data handling
from sklearn.preprocessing import StandardScaler
from dataloaders import baseline_data_prep, save_loader_to_disk, load_loader_from_disk

# simulation
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model


DATA_ROOT = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data")
RESULTS_PATH = DATA_ROOT / "IY011_simulation_parameters_sobol.csv" #  this csv file stores all the simulation parameters used
df_params = pd.read_csv(RESULTS_PATH) 
TRAJ_PATH = [DATA_ROOT / df_params['trajectory_filename'].values[i] for i in range(len(df_params))]
TRAJ_NPZ_PATH = [traj_file.with_suffix('.npz') for traj_file in TRAJ_PATH]


# === Dataloader hyperparams & data prep ===
batch_size = 32
num_groups_train=100
num_groups_val=int(num_groups_train * 0.2)
num_groups_test=int(num_groups_train * 0.2)
num_traj=2 # provide more trajectories per group for easier task
param_dist_threshold=0.7 # 2-fold change
sample_len=None # randomly crop each time series sample to this length
train_loader, val_loader, test_loader, scaler = baseline_data_prep(
    TRAJ_NPZ_PATH,
    batch_size=batch_size,
    num_groups_train=num_groups_train,
    num_groups_val=num_groups_val,
    num_groups_test=num_groups_test,
    num_traj=num_traj,
    sample_len=sample_len,
    param_dist_threshold=param_dist_threshold,
    verbose=True,
)
# === Dataloader hyperparams & data prep ===

# === Save data for debugging later === 
# 1. Define paths
train_save_path = DATA_ROOT / "IY018_static_train.pt"
val_save_path   = DATA_ROOT / "IY018_static_val.pt"
test_save_path  = DATA_ROOT / "IY018_static_test.pt"

# 2. Check if static data already exists
if not train_save_path.exists():
    print("Static data not found. Saving...")
    
    # Save them to disk
    save_loader_to_disk(train_loader, train_save_path)
    save_loader_to_disk(val_loader, val_save_path)
    save_loader_to_disk(test_loader, test_save_path)
else:
    print("Found existing static data on disk, the simulated data will not be saved, to prevent overwriting existing data.")
# === Save data for debugging later === 