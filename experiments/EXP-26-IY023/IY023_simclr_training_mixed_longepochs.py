import torch
import torch.optim as optim
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from dataloaders.simclr import ssl_data_prep
from dataloaders import  save_loader_to_disk
from models.ssl_transformer import SSL_Transformer
from training.train import train_ssl_model
from info_nce import InfoNCE
import wandb

# Setup Configuration
EXP_ROOT = Path("/home/s1732775/scratch_s1732775/SSA/experiments/EXP-26-IY020")

DATA_SOURCES = [
    (EXP_ROOT / "data",              "IY020_simulation_parameters_sobol.csv"),
    (EXP_ROOT / "data_mu_variation", "IY020_simulation_mu_parameters_sobol.csv"),
    (EXP_ROOT / "data_cv_variation", "IY020_simulation_cv_parameters_sobol.csv"),
    (EXP_ROOT / "data_t_ac_variation", "IY020_simulation_t_ac_parameters_sobol.csv"),
]

# Collect trajectory paths from all 4 data sources, applying the same quality filter
TRAJ_PATH = []
for data_root, results_csv in DATA_SOURCES:
    df = pd.read_csv(data_root / results_csv)
    df = df[(df['success'] == True) &
            (df['error_message'].isna()) &
            (df['mean_rel_error_pct'] < 10) &
            (df['cv_rel_error_pct'] < 10) &
            (df['t_ac_rel_error_pct'] < 10)]
    paths = [data_root / df['trajectory_filename'].values[i] for i in range(len(df))]
    TRAJ_PATH.extend(paths)
    print(f"  {data_root.name}: {len(paths)} trajectories")

print(f"Total trajectories: {len(TRAJ_PATH)}")

# === Dataloader hyperparams & data prep ===
batch_size = 1024
num_traj = 1  # number of trajectories per view
sample_len = 500
log_scale = False
normalisation = 'batch-wise'

train_loader, val_loader, test_loader = ssl_data_prep(
    TRAJ_PATH,
    batch_size=batch_size,
    sample_len=sample_len,
    log_scale=log_scale,
    normalisation=normalisation,
    num_traj=num_traj,
)
# === Dataloader hyperparams & data prep ===

# === Model hyperparams ===
X1_b, X2_b, y_b = next(iter(train_loader))
input_size = X1_b.shape[2]
num_classes = 2
d_model = 16
nhead = 4
num_layers = 2
dropout = 0.01
use_conv1d = False

model = SSL_Transformer(
    input_size=input_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout,
    use_conv1d=use_conv1d,
)
# === Model hyperparams ===

# === Training hyperparams ===
epochs = 500
patience = epochs // 3
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)

warmup_steps = int(0.1 * epochs)
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=epochs,
)

nce_temp = 0.2
loss_fn = InfoNCE(negative_mode='unpaired', temperature=nce_temp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_clip = None
verbose = True

model.to(device)
# === Training hyperparams ===

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f'IY023_simCLR_mixed_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_{normalisation}_{timestamp}_model.pth'

# === wandb config ===
wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY023-SSL-model",
    "name": f"simCLR_mixed_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_trj{num_traj}_len{sample_len}_{normalisation}_{timestamp}",
    "dataset": [str(data_root) for data_root, _ in DATA_SOURCES],
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
    "num_traj_per_view": num_traj,
    "sample_len": sample_len,
    "log_scale": log_scale,
    "normalisation": normalisation,
    "save_path": save_path,
    "nce_temp": nce_temp,
    "grad_clip": grad_clip,
    "total_trajectories": len(TRAJ_PATH),
}
# === wandb config ===

from training.train import train_ssl_model
history = train_ssl_model(
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
    wandb_logging=True,
    wandb_config=wandb_config,
)

torch.cuda.empty_cache()
