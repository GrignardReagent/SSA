#!/usr/bin/env python3
'''
2-fold difference 
'''

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
from simulation.unseen import generate_unseen_classes
from training.diagnostics import run_permutation_test
from training.few_shot import (
    prepare_group_input,
    test_baseline_performance,
    test_svm_few_shot,
)
from utils.processing.normalisation import instance_norm_np
from utils.svm import extract_data_for_svm


# ==========================================
# CONFIGURATION
# ==========================================
# 2-fold change experiments (from IY011)
EXPERIMENTS = [
    {
        "name": "Baseline",
        "data_dir": "../EXP-25-IY011/data",
        "csv_name": "IY011_simulation_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "black"
    },
    {
        "name": "Mu-Variation",
        "data_dir": "../EXP-25-IY011/data_mu_variation",
        "csv_name": "IY011_simulation_mu_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "blue"
    },
    {
        "name": "CV-Variation",
        "data_dir": "../EXP-25-IY011/data_cv_variation",
        "csv_name": "IY011_simulation_cv_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "green"
    },
    {
        "name": "T_ac-Variation",
        "data_dir": "../EXP-25-IY011/data_t_ac_variation",
        "csv_name": "IY011_simulation_t_ac_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "red"
    }
]

# ==========================================
BASELINE_DATA_ROOT = Path(EXPERIMENTS[0]["data_dir"])
MU_DATA_ROOT = Path(EXPERIMENTS[1]["data_dir"])
CV_DATA_ROOT = Path(EXPERIMENTS[2]["data_dir"]) 
TAC_DATA_ROOT = Path(EXPERIMENTS[3]["data_dir"])

baseline_train_save_path = BASELINE_DATA_ROOT / EXPERIMENTS[0]["train_save_path"]
baseline_val_save_path   = BASELINE_DATA_ROOT / EXPERIMENTS[0]["val_save_path"]
baseline_test_save_path  = BASELINE_DATA_ROOT / EXPERIMENTS[0]["test_save_path"]
baseline_train_loader = load_loader_from_disk(baseline_train_save_path)
baseline_val_loader = load_loader_from_disk(baseline_val_save_path)
baseline_test_loader = load_loader_from_disk(baseline_test_save_path)

cv_train_save_path = CV_DATA_ROOT / EXPERIMENTS[2]["train_save_path"]
cv_val_save_path   = CV_DATA_ROOT / EXPERIMENTS[2]["val_save_path"]
cv_test_save_path  = CV_DATA_ROOT / EXPERIMENTS[2]["test_save_path"]
cv_train_loader = load_loader_from_disk(cv_train_save_path)
cv_val_loader = load_loader_from_disk(cv_val_save_path)
cv_test_loader = load_loader_from_disk(cv_test_save_path)   

tac_train_save_path = TAC_DATA_ROOT / EXPERIMENTS[3]["train_save_path"]
tac_val_save_path   = TAC_DATA_ROOT / EXPERIMENTS[3]["val_save_path"]
tac_test_save_path  = TAC_DATA_ROOT / EXPERIMENTS[3]["test_save_path"]
tac_train_loader = load_loader_from_disk(tac_train_save_path)
tac_val_loader = load_loader_from_disk(tac_val_save_path)
tac_test_loader = load_loader_from_disk(tac_test_save_path)

mu_train_save_path = MU_DATA_ROOT / EXPERIMENTS[1]["train_save_path"]
mu_val_save_path   = MU_DATA_ROOT / EXPERIMENTS[1]["val_save_path"]
mu_test_save_path  = MU_DATA_ROOT / EXPERIMENTS[1]["test_save_path"]
mu_train_loader = load_loader_from_disk(mu_train_save_path)
mu_val_loader = load_loader_from_disk(mu_val_save_path)
mu_test_loader = load_loader_from_disk(mu_test_save_path)   

def training_pipeline(DATA_ROOT, train_loader, val_loader, test_loader, exp_name):

    # === Dataloader hyperparams & data prep ===
    batch_size = 64
    num_groups_train=3000 
    num_groups_val=int(num_groups_train * 0.2)
    num_groups_test=int(num_groups_train * 0.2)
    num_traj=2
    param_dist_threshold=0.7 #2-fold 
    # === Dataloader hyperparams & data prep ===

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

    model_path = f'IY011_baseline_transformer_model_{exp_name}.pth'
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
        "project": "IY016-baseline-model",
        "name": f"num_groups_train_3000_traj_2 ({exp_name})", # change this to what you want
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
        "param_dist_threshold": param_dist_threshold,
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
    
    
training_pipeline(BASELINE_DATA_ROOT, baseline_train_loader, baseline_val_loader, baseline_test_loader, EXPERIMENTS[0]["name"].lower())
training_pipeline(MU_DATA_ROOT, mu_train_loader, mu_val_loader, mu_test_loader, EXPERIMENTS[1]["name"].lower())
training_pipeline(CV_DATA_ROOT, cv_train_loader, cv_val_loader, cv_test_loader, EXPERIMENTS[2]["name"].lower())
training_pipeline(TAC_DATA_ROOT, tac_train_loader, tac_val_loader, tac_test_loader, EXPERIMENTS[3]["name"].lower())