#!/usr/bin/env python3
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
from utils.data_loader import baseline_data_prep, save_loader_to_disk

# simulation
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
from simulation.unseen import generate_unseen_classes
from training.diagnostics import run_permutation_test
from training.few_shot import (
    prepare_group_input,
    test_baseline_performance,
    test_svm_few_shot,
)
from utils.processing.normalisation import instance_norm_np
from utils.svm import extract_data_for_svm

'''
Train a baseline transformer model on IY011 data with varying cv parameters.
'''

# Get the absolute path of the directory containing this script
script_dir = Path(__file__).resolve().parent
DATA_ROOT = script_dir / "temp_data_cv_variation"
RESULTS_PATH = DATA_ROOT / "IY011_simulation_cv_parameters_sobol.csv" #  this csv file stores all the simulation parameters used
df_params = pd.read_csv(RESULTS_PATH) 
# filter out only successful simulations with no error_message, and mean_rel_error_pct < 10, cv_rel_error_pct  < 10, & t_ac_rel_error_pct < 10
df_params = df_params[(df_params['success'] == True) & 
                      (df_params['error_message'].isna()) &
                      (df_params['mean_rel_error_pct'] < 10) & 
                      (df_params['cv_rel_error_pct'] < 10) & 
                      (df_params['t_ac_rel_error_pct'] < 10)]
TRAJ_PATH = [DATA_ROOT / df_params['trajectory_filename'].values[i] for i in range(len(df_params))]
TRAJ_NPZ_PATH = [traj_file.with_suffix('.npz') for traj_file in TRAJ_PATH]

# === Dataloader hyperparams & data prep ===
batch_size = 64
num_groups_train=3000 
num_groups_val=int(num_groups_train * 0.2)
num_groups_test=int(num_groups_train * 0.2)
num_traj=2
train_loader, val_loader, test_loader, scaler = baseline_data_prep(
    TRAJ_NPZ_PATH,
    batch_size=batch_size,
    num_groups_train=num_groups_train,
    num_groups_val=num_groups_val,
    num_groups_test=num_groups_test,
    num_traj=num_traj,
)
# === Dataloader hyperparams & data prep ===

# === Save data for debugging later === 
# 1. Define paths
train_save_path = DATA_ROOT / "static_train.pt"
val_save_path   = DATA_ROOT / "static_val.pt"
test_save_path  = DATA_ROOT / "static_test.pt"

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

X_b, y_b = next(iter(train_loader))
print(X_b.shape, y_b.shape) # (Batch, Seq_Len, num_traj), (Batch, 1)

# === Model hyperparams ===
input_size = X_b.shape[2]  # number of input channels/features
num_classes = 2
d_model=64
nhead=4
num_layers=2
dropout=0.001
use_conv1d=False 
max_seq_length = X_b.shape[1] + 100  # e.g., 5020 + 100 = 5120

model = TransformerClassifier(
    input_size=input_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout, 
    use_conv1d=use_conv1d,
    max_seq_length=max_seq_length,
)

model_path = 'IY011_baseline_transformer_model_7_cv.pth'
# === Model hyperparams ===

# === Training hyperparams ===
epochs = 100
patience = 10
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)

### schedulers ### 
# simple scheduler choice
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5) 

loss_fn = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_clip = 1.0
save_path = None
verbose = True

model.to(device)
# === Training hyperparams ===

# === wandb config (required for tracking within train_model) ===
wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY011-baseline-model",
    "name": f"num_groups_train_{num_groups_train}_traj_{num_traj}_batch_size_{batch_size} (cv variation)", # change this to what you want
    "dataset": DATA_ROOT.name,
    "batch_size": batch_size,
    "input_size": input_size,
    "d_model": d_model,
    "nhead": nhead,
    "num_layers": num_layers,
    "num_classes": num_classes,
    "dropout": dropout,
    "use_conv1d": use_conv1d,
    "epochs": epochs,
    "patience": patience,
    "lr": lr,
    "optimizer": type(optimizer).__name__,
    "scheduler": type(scheduler).__name__,
    "loss_fn": type(loss_fn).__name__,
    "model": type(model).__name__,
    "batch_size": train_loader.batch_size,
    "num_traj_per_group": num_traj,
    "num_groups_train": num_groups_train,
    "num_groups_val": num_groups_val,
    "num_groups_test": num_groups_test,
    "model_path": model_path,
}
# === wandb config === 

history = train_model(
    model,
    train_loader,
    val_loader,
    epochs=epochs,
    patience=patience,
    lr=lr,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    grad_clip=grad_clip,
    save_path=save_path,
    verbose=verbose,
    wandb_logging=True, # this enables wandb logging within train_model
    wandb_config=wandb_config, # pass the config dictionary
)

# save the trained model
torch.save(model.state_dict(), model_path)
print('Model saved to', model_path)


###########################################################################################
# EVALUATION
###########################################################################################

print("\n=== Evaluating on Test Set ===")
# evaluate on test set
test_loss, test_acc = evaluate_model(
    model,
    test_loader,
    loss_fn,
    device,
)

# ==========================================
# SVM BENCHMARK
# ==========================================



# 1. Extract Data dynamically from your loaders
X_train_svm, y_train_svm = extract_data_for_svm(train_loader)
X_test_svm, y_test_svm   = extract_data_for_svm(test_loader)

print(f"SVM Train Shape: {X_train_svm.shape}")
print(f"SVM Test Shape:  {X_test_svm.shape}")

# 2. Run the SVM Classifier
svm_accuracy = svm_classifier(
    X_train_svm,
    X_test_svm,
    y_train_svm,
    y_test_svm,
)

# ==========================================
# 1. GENERATE UNSEEN DATA
# ==========================================
# === RUN GENERATION ===
unseen_datasets = generate_unseen_classes(n_classes=5, n_trajs_per_class=20)

# ==========================================
# 2. EVALUATE BASELINE MODEL on UNSEEN CLASSES
# ==========================================
print("\n=== Evaluating Baseline Model on Unseen Classes ===")
acc = test_baseline_performance(model, unseen_datasets, scaler, num_traj=num_traj)

# ==========================================
# SVM ON UNSEEN DATA BENCHMARK
# ==========================================

print("\n=== Evaluating SVM Few-Shot on Unseen Classes ===")
acc_svm_few_shot = test_svm_few_shot(unseen_datasets)

# ==========================================
# PERMUTATION TESTS
# ==========================================

print("\n=== Running Permutation Test ===")
acc_orig, acc_shuff = run_permutation_test(model, test_loader, device=device)

# Test baseline performance on unseen data
acc = test_baseline_performance(model, unseen_datasets, scaler, num_traj=num_traj)