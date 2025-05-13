import torch
import numpy as np
import time
from models.lstm import LSTMClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.load_data import load_and_split_data  

output_file = 'data/combined_traj_1199_1200.csv'
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

train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=64)
test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=64)

configs = [
    {"name": "Vanilla LSTM", "conv1d": False, "attention": False, "multihead": False, "aux": False},
    {"name": "Conv1D Only", "conv1d": True,  "attention": False, "multihead": False, "aux": False},
    {"name": "Conv1D + Attention", "conv1d": True,  "attention": True, "multihead": False, "aux": False},
    {"name": "Conv1D + MultiHead", "conv1d": True,  "attention": True, "multihead": True,  "aux": False},
    {"name": "Full Model", "conv1d": True,  "attention": True, "multihead": True,  "aux": True},
]

results = []
num_runs = 10

for cfg in configs:
    print(f"\n=== Repeated Training for {cfg['name']} ===")
    test_accs, val_accs, train_accs, train_losses, times = [], [], [], [], []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        start = time.time()

        model = LSTMClassifier(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=len(np.unique(y_train)),
            dropout_rate=0.2,
            use_attention=cfg['attention'],
            num_attention_heads=4 if cfg['multihead'] else 1,
            use_auxiliary=cfg['aux'],
            use_conv1d=cfg['conv1d']
        )

        history = model.train_model(train_loader, val_loader, epochs=50, patience=10)
        test_acc = model.evaluate(test_loader)
        duration = time.time() - start

        test_accs.append(test_acc)
        val_accs.append(history['val_acc'][-1] if 'val_acc' in history and history['val_acc'] else None)
        train_accs.append(history['train_acc'][-1])
        train_losses.append(history['train_loss'][-1])
        times.append(duration)

    results.append({
        "model": cfg['name'],
        "test_acc_mean": np.mean(test_accs),
        "test_acc_std": np.std(test_accs),
        "val_acc_mean": np.mean(val_accs),
        "train_acc_mean": np.mean(train_accs),
        "train_loss_mean": np.mean(train_losses),
        "time_mean": np.mean(times)
    })

print("\n=== Averaged Experiment Summary ===")
for r in results:
    print(f"{r['model']}: Test Acc = {r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}, Val Acc = {r['val_acc_mean']:.4f}, Train Acc = {r['train_acc_mean']:.4f}, Time = {r['time_mean']:.2f}s")

