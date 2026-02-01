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


'''
No scheduler, no early stopping

'''


# Setup Configuration
DATA_ROOT = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data_mu_variation")
RESULTS_PATH = DATA_ROOT / "IY011_simulation_mu_parameters_sobol.csv" #  this csv file stores all the simulation parameters used
df_params = pd.read_csv(RESULTS_PATH) 
TRAJ_PATH = [DATA_ROOT / df_params['trajectory_filename'].values[i] for i in range(len(df_params))]
TRAJ_NPZ_PATH = [traj_file.with_suffix('.npz') for traj_file in TRAJ_PATH]

# === Dataloader hyperparams & data prep ===
batch_size = 32 
num_traj=10 # number of trajectories per view
sample_len=400
log_scale = False 
instance_norm = False

train_loader, val_loader, test_loader, scaler = ssl_data_prep(
    TRAJ_NPZ_PATH,
    batch_size=batch_size,
    sample_len=sample_len,
    log_scale=log_scale,
    instance_norm=instance_norm,
    num_traj=num_traj,
)
# === Dataloader hyperparams & data prep ===

# === Model hyperparams ===
X1_b, X2_b, y_b = next(iter(train_loader))
input_size = X1_b.shape[2] 
num_classes = 2
d_model=128
nhead=4
num_layers=2
dropout=0.001
use_conv1d=False 

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
epochs = 200
patience = epochs # SSL may benefit from high patience
lr = 1e-2  # Reduced from 1e-3 to prevent divergence with unnormalized data
optimizer = optim.Adam(model.parameters(), lr=lr)

# simple scheduler choice
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience= patience // 3, factor=0.5) 
scheduler = None
loss_fn = InfoNCE(negative_mode='unpaired')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_clip = 1.0
save_path = None
verbose = True

model.to(device)
# === Training hyperparams ===

model_path = f'IY017_simCLR_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_model.pth'

# === wandb config (required for tracking within train_model) ===
wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY017-SSL-model",
    "name": f"simCLR_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_trj{num_traj} (mu) - no scheduler, no early stopping", # change this to what you want
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
    "num_traj_per_view": num_traj,
    "sample_len": sample_len,
    "log_scale" : log_scale,
    "instance_norm": instance_norm,
    "model_path": model_path,
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
    wandb_logging=True, # this enables wandb logging within train_model
    wandb_config=wandb_config, # pass the config dictionary
)

# save the trained model
torch.save(model.state_dict(), model_path)

# clear cuda cache
torch.cuda.empty_cache()