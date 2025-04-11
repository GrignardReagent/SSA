import itertools
import torch
import torch.nn as nn
import numpy as np
import time
import csv
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

# Grid search space
hidden_sizes = [64, 128, 256]
num_layers_list = [2, 3, 4]
dropout_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
learning_rates = [0.01, 1e-3, 1e-4, 5e-4]
batch_sizes = [32, 64, 128]

param_grid = list(itertools.product(hidden_sizes, 
                                    num_layers_list, 
                                    dropout_rates, 
                                    learning_rates, 
                                    batch_sizes))

csv_columns = ["hidden_size", "num_layers", "dropout_rate", "learning_rate", "batch_size", "train_acc", "val_acc", "test_acc", "time"]
results_file = "lstm_conv1d_gridsearch_results_10_04_2025.csv"

# Initialize CSV file and write header
with open(results_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

# Grid search with tqdm
results = []

for i, (hidden_size, num_layers, dropout_rate, lr, batch_size) in enumerate(tqdm(param_grid, desc="Grid Search")):
    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=batch_size)

    start = time.time()
    model = LSTMClassifier(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=len(np.unique(y_train)),
        dropout_rate=dropout_rate,
        learning_rate=lr,
        use_conv1d=True,
        use_attention=False,
        use_auxiliary=False
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    history = model.train_model(train_loader, val_loader, epochs=50, patience=10)
    test_acc = model.evaluate(test_loader)
    duration = time.time() - start

    result = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": lr,
        "batch_size": batch_size,
        "train_acc": history['train_acc'][-1],
        "val_acc": history['val_acc'][-1] if 'val_acc' in history and history['val_acc'] else None,
        "test_acc": test_acc,
        "time": duration
    }

    results.append(result)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(result)

# Sort results by test accuracy
results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

print("\n=== Top Configs ===")
for r in results[:10]:
    print(f"Test Acc: {r['test_acc']:.4f}, Val Acc: {r['val_acc']}, Train Acc: {r['train_acc']:.4f}, Time: {r['time']:.2f}s")
    print(f"  Params: Hidden={r['hidden_size']}, Layers={r['num_layers']}, Dropout={r['dropout_rate']}, LR={r['learning_rate']}, Batch={r['batch_size']}")
