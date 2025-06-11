#!/usr/bin/python
"""Grid search hyperparameter tuning for TransformerClassifier.

This script mirrors the grid search approach used in ``IY001.py`` but applies
it to the ``TransformerClassifier``. It loads the same dataset,
performs standard preprocessing, and evaluates different model
hyperparameter combinations.
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
# Data preparation (same dataset and preprocessing as IY001.py)
# ---------------------------------------------------------------------------
DATA_FILE = "~/stochastic_simulations/experiments/EXP-25-IY001/data/combined_traj_1199_1200_SS.csv"

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
# Grid search configuration
# ---------------------------------------------------------------------------
# Hyperparameter ranges (kept fairly small for demonstration)
d_models = [32, 64, 128]
nheads = [2, 4]
num_layers_list = [1, 2, 3]
dropout_rates = [0.1, 0.2, 0.3]
learning_rates = [1e-3, 1e-4]
batch_sizes = [32, 64]

param_grid = list(
    itertools.product(
        d_models,
        nheads,
        num_layers_list,
        dropout_rates,
        learning_rates,
        batch_sizes,
    )
)

csv_columns = [
    "d_model",
    "nhead",
    "num_layers",
    "dropout_rate",
    "learning_rate",
    "batch_size",
    "train_acc",
    "val_acc",
    "test_acc",
    "time",
]
results_file = "IY006A.csv"

# Initialise CSV with header
with open(results_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

# ---------------------------------------------------------------------------
# Grid search loop
# ---------------------------------------------------------------------------
results = []
for d_model, nhead, num_layers, dropout_rate, lr, batch_size in tqdm(
    param_grid, desc="IY006: Transformer Grid Search"
):
    train_loader = DataLoader(
        to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=batch_size)

    start = time.time()
    model = TransformerClassifier(
        input_size=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_size=len(np.unique(y_train)),
        dropout_rate=dropout_rate,
        learning_rate=lr,
        use_conv1d=True,  # conv1d preprocessing like in the original LSTM grid
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    history = model.train_model(train_loader, val_loader, epochs=50, patience=10)
    test_acc = model.evaluate(test_loader)
    duration = time.time() - start

    result = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": lr,
        "batch_size": batch_size,
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1]
        if "val_acc" in history and history["val_acc"]
        else None,
        "test_acc": test_acc,
        "time": duration,
    }

    results.append(result)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(result)

# ---------------------------------------------------------------------------
# Report top configurations
# ---------------------------------------------------------------------------
results = sorted(results, key=lambda x: x["test_acc"], reverse=True)

print("\n=== Top Configs ===")
for r in results[:10]:
    print(
        f"Test Acc: {r['test_acc']:.4f}, Val Acc: {r['val_acc']}, "
        f"Train Acc: {r['train_acc']:.4f}, Time: {r['time']:.2f}s"
    )
    print(
        f"  Params: d_model={r['d_model']}, nhead={r['nhead']}, "
        f"Layers={r['num_layers']}, Dropout={r['dropout_rate']}, "
        f"LR={r['learning_rate']}, Batch={r['batch_size']}"
    )
