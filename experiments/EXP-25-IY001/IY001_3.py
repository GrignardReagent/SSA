import itertools
import torch
import torch.nn as nn
import numpy as np
import time
import csv
import os
import pandas as pd
from models.lstm import LSTMClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.load_data import load_and_split_data
from tqdm import tqdm

# Load and preprocess data
output_file = 'data/combined_traj_1199_1200_SS.csv'
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(output_file, split_val_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

def to_tensor(data, labels):
    return TensorDataset(torch.tensor(data, dtype=torch.float32),
                         torch.tensor(labels, dtype=torch.long))

# Architecture configs to sweep
architectures = [
    {"name": "Vanilla LSTM", "conv1d": False, "attention": False, "multihead": False, "aux": False},
    {"name": "Conv1D Only", "conv1d": True,  "attention": False, "multihead": False, "aux": False},
    {"name": "Conv1D + Attention", "conv1d": True,  "attention": True, "multihead": False, "aux": False},
    {"name": "Conv1D + MultiHead", "conv1d": True,  "attention": True, "multihead": True,  "aux": False},
    {"name": "Full Model", "conv1d": True,  "attention": True, "multihead": True,  "aux": True},
]

# Grid search space
hidden_sizes = [64, 128, 256]
num_layers_list = [2, 3, 4]
dropout_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
learning_rates = [0.01, 1e-3, 1e-4, 5e-4]
batch_sizes = [32, 64, 128]
epochs_list = [50, 100, 150]

param_grid = list(itertools.product(hidden_sizes,
                                    num_layers_list,
                                    dropout_rates,
                                    learning_rates,
                                    batch_sizes,
                                    epochs_list))

csv_columns = ["architecture", "hidden_size", "num_layers", "dropout_rate", "learning_rate", "batch_size", "epochs", "train_acc", "val_acc", "test_acc", "test_acc_std", "time"]
results_file = "IY001A.csv"

# Initialize CSV file if it doesn't exist
if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
    with open(results_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

# Claim a config for exclusive access
def claim_config(config_key):
    """Claim a config by writing a placeholder row with 'IN_PROGRESS' if not yet taken."""
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        keys = set((
            row["architecture"],
            row["hidden_size"],
            row["num_layers"],
            row["dropout_rate"],
            row["learning_rate"],
            row["batch_size"],
            row["epochs"]
        ) for _, row in df.iterrows())

        if config_key in keys:
            return False

    # Claim it with a placeholder
    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow({
            "architecture": config_key[0],
            "hidden_size": config_key[1],
            "num_layers": config_key[2],
            "dropout_rate": config_key[3],
            "learning_rate": config_key[4],
            "batch_size": config_key[5],
            "epochs": config_key[6],
            "train_acc": "IN_PROGRESS",
            "val_acc": None,
            "test_acc": None,
            "test_acc_std": None,
            "time": None
        })

    return True

num_runs = 10
results = []

# Grid search loop
for cfg in tqdm(architectures):
    print(f"\n=== Sweeping Architecture: {cfg['name']} ===")

    for hidden_size, num_layers, dropout_rate, lr, batch_size, epochs in tqdm(param_grid, desc="Grid Search"):
        config_key = (cfg['name'], hidden_size, num_layers, dropout_rate, lr, batch_size, epochs)

        if not claim_config(config_key):
            continue  # Already completed or in progress

        print(f"Testing: Hidden={hidden_size}, Layers={num_layers}, Dropout={dropout_rate}, LR={lr}, Batch={batch_size}, Epochs={epochs}")
        acc_runs = []
        run_stats = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
            test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=batch_size)

            model = LSTMClassifier(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=len(np.unique(y_train)),
                dropout_rate=dropout_rate,
                learning_rate=lr,
                use_conv1d=cfg['conv1d'],
                use_attention=cfg['attention'],
                num_attention_heads=4 if cfg['multihead'] else 1,
                use_auxiliary=cfg['aux']
            )

            try:
                start = time.time()
                history = model.train_model(train_loader, val_loader, epochs=epochs, patience=10)
                test_acc = model.evaluate(test_loader)
                duration = time.time() - start

                acc_runs.append(test_acc)
                run_stats.append({
                    "architecture": cfg['name'],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout_rate": dropout_rate,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "train_acc": history['train_acc'][-1],
                    "val_acc": history['val_acc'][-1] if 'val_acc' in history and history['val_acc'] else None,
                    "test_acc": test_acc,
                    "test_acc_std": None,
                    "time": duration
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA OOM error caught. Skipping this config.")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise

        if acc_runs:
            mean_acc = np.mean(acc_runs)
            std_acc = np.std(acc_runs)

            avg_result = run_stats[0].copy()
            avg_result["test_acc"] = mean_acc
            avg_result["test_acc_std"] = std_acc
            results.append(avg_result)

            # Update the placeholder entry
            df = pd.read_csv(results_file)
            mask = (
                (df["architecture"] == config_key[0]) &
                (df["hidden_size"] == config_key[1]) &
                (df["num_layers"] == config_key[2]) &
                (df["dropout_rate"] == config_key[3]) &
                (df["learning_rate"] == config_key[4]) &
                (df["batch_size"] == config_key[5]) &
                (df["epochs"] == config_key[6])
            )
            df.loc[mask, avg_result.keys()] = list(avg_result.values())
            df.to_csv(results_file, index=False)

# Report top results
results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

print("\n=== Top Configs ===")
for r in results[:10]:
    print(f"Test Acc: {r['test_acc']:.4f} ± {r['test_acc_std']:.4f}, Val Acc: {r['val_acc']}, Train Acc: {r['train_acc']:.4f}, Time: {r['time']:.2f}s")
    print(f"  Params: Hidden={r['hidden_size']}, Layers={r['num_layers']}, Dropout={r['dropout_rate']}, LR={r['learning_rate']}, Batch={r['batch_size']}, Epochs={r['epochs']}")
