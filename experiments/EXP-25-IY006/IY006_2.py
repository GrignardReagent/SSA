#!/usr/bin/python
"""Extended grid search hyperparameter tuning for TransformerClassifier.

This script extends the grid search approach in ``IY006.py`` by exploring
additional hyperparameters in the ``TransformerClassifier``. It examines advanced
options such as optimizer choice, pooling strategies, auxiliary tasks, and more.
The search space is kept small as a proof of concept.
"""

import itertools
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.transformer import TransformerClassifier
from utils.load_data import load_and_split_data

# ---------------------------------------------------------------------------
# Data preparation (same dataset and preprocessing as IY006.py)
# ---------------------------------------------------------------------------
# USING CV FIXED DATASET
DATA_FILE = "/exports/eddie/scratch/s1732775/SSA/experiments/EXP-25-IY006/data/combined_traj_cv_0.32_0.32_SS.csv"

# Load data with an explicit validation split
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    DATA_FILE, split_val_size=0.2
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape to [samples, seq_len, 1] for the transformer
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


def to_tensor(data, labels):
    """Helper to convert arrays to a TensorDataset."""
    return TensorDataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )

# ---------------------------------------------------------------------------
# Grid search configuration with extended hyperparameters
# ---------------------------------------------------------------------------
# Basic hyperparameters - using small search space
d_models = [128, 256]
nheads = [4, 8]
num_layers_list = [2, 3]
dropout_rates = [0.1, 0.01, 0.001]
learning_rates = [0.01, 1e-3]
batch_sizes = [64, 128]

# Extended hyperparameters
# optimizers = ["Adam", "AdamW"]
pooling_strategies = ["last", "mean"]
use_conv1d_options = [True, False]
use_auxiliary_options = [False, True]
gradient_clip_options = [1.0, 5.0]

# Create parameter grid
param_grid = list(
    itertools.product(
        d_models,
        nheads,
        num_layers_list,
        dropout_rates,
        learning_rates,
        batch_sizes,
        # optimizers,
        pooling_strategies,
        use_conv1d_options,
        use_auxiliary_options,
        gradient_clip_options,
    )
)

print(f"Total configurations to test: {len(param_grid)}")

# Define columns for results file
csv_columns = [
    "d_model",
    "nhead",
    "num_layers",
    "dropout_rate",
    "learning_rate",
    "batch_size",
    # "optimizer",
    "pooling_strategy",
    "use_conv1d",
    "use_auxiliary",
    "gradient_clip",
    "train_acc",
    "val_acc",
    "test_acc",
    "time",
]
results_file = "IY006C.csv"

# Initialize CSV with header
with open(results_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

# ---------------------------------------------------------------------------
# Grid search loop
# ---------------------------------------------------------------------------
results = []
for (
    d_model,
    nhead,
    num_layers,
    dropout_rate,
    lr,
    batch_size,
    # optimizer,
    pooling_strategy,
    use_conv1d,
    use_auxiliary,
    gradient_clip,
) in tqdm(param_grid, desc="IY006_1: Extended Transformer Grid Search"):
    train_loader = DataLoader(
        to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=batch_size)

    start = time.time()
    
    # Set auxiliary weight if using auxiliary task
    aux_weight = 0.1 if use_auxiliary else None
    
    # Create model with extended hyperparameters
    model = TransformerClassifier(
        input_size=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_size=len(np.unique(y_train)),
        dropout_rate=dropout_rate,
        learning_rate=lr,
        # optimizer=optimizer,
        use_conv1d=use_conv1d,
        use_auxiliary=use_auxiliary,
        aux_weight=aux_weight,
        pooling_strategy=pooling_strategy,
        gradient_clip=gradient_clip,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Train with the same epochs and patience as the original script
    history = model.train_model(train_loader, val_loader, epochs=50, patience=10)
    test_acc = model.evaluate(test_loader)
    duration = time.time() - start

    # Record results with extended hyperparameters
    result = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": lr,
        "batch_size": batch_size,
        # "optimizer": optimizer,
        "pooling_strategy": pooling_strategy,
        "use_conv1d": use_conv1d,
        "use_auxiliary": use_auxiliary,
        "gradient_clip": gradient_clip,
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1]
        if "val_acc" in history and history["val_acc"]
        else None,
        "test_acc": test_acc,
        "time": duration,
    }

    results.append(result)

    # Save results incrementally
    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(result)

# ---------------------------------------------------------------------------
# Report top configurations
# ---------------------------------------------------------------------------
results = sorted(results, key=lambda x: x["test_acc"], reverse=True)

print("\n=== Top Configs ===")
for i, r in enumerate(results[:10]):
    print(f"Rank {i+1}:")
    print(
        f"Test Acc: {r['test_acc']:.4f}, Val Acc: {r['val_acc']}, "
        f"Train Acc: {r['train_acc']:.4f}, Time: {r['time']:.2f}s"
    )
    print(
        f"  Basic Params: d_model={r['d_model']}, nhead={r['nhead']}, "
        f"Layers={r['num_layers']}, Dropout={r['dropout_rate']}, "
        f"LR={r['learning_rate']}, Batch={r['batch_size']}"
    )
    print(
        # f"  Extended Params: Optimizer={r['optimizer']}, "
        f"Pooling={r['pooling_strategy']}, Conv1D={r['use_conv1d']}, "
        f"Auxiliary={r['use_auxiliary']}, GradClip={r['gradient_clip']}"
    )
    print()

# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------
if results:
    best_config = results[0]
    print(f"Best configuration: Test Accuracy = {best_config['test_acc']:.4f}")
    print("Recreating and saving the best model...")
    
    # Recreate the best model
    best_model = TransformerClassifier(
        input_size=1,
        d_model=best_config['d_model'],
        nhead=best_config['nhead'],
        num_layers=best_config['num_layers'],
        output_size=len(np.unique(y_train)),
        dropout_rate=best_config['dropout_rate'],
        learning_rate=best_config['learning_rate'],
        # optimizer=best_config['optimizer'],
        use_conv1d=best_config['use_conv1d'],
        use_auxiliary=best_config['use_auxiliary'],
        aux_weight=0.1 if best_config['use_auxiliary'] else None,
        pooling_strategy=best_config['pooling_strategy'],
        gradient_clip=best_config['gradient_clip'],
    )
    
    # Train the best model
    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=best_config['batch_size'], shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=best_config['batch_size'])
    
    history = best_model.train_model(
        train_loader, 
        val_loader, 
        epochs=50, 
        patience=10,
        save_path="IY006C.pt"
    )

    print("Best model saved as 'IY006C.pt'")
