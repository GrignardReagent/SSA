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
DATA_ROOT = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY020/data")
RESULTS_PATH = DATA_ROOT / "IY020_simulation_parameters_sobol.csv" #  this csv file stores all the simulation parameters used
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
batch_size = 4096
num_traj=1 # number of trajectories per view
sample_len=500
log_scale = False 
instance_norm = True

train_loader, val_loader, test_loader = ssl_data_prep(
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
d_model=16
nhead=4
num_layers=2
dropout=0.01
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
epochs = 500
patience = epochs // 3
lr = 4.8 # same as in SimCLR paper

# LARS optimizer to be used for SSL
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    """
    LARS Optimizer (Layer-wise Adaptive Rate Scaling)
    Standard implementation for SimCLR / Self-Supervised Learning.
    
    Args:
        params: Model parameters.
        lr: Base learning rate (SimCLR uses high LR, e.g., 1.0 - 5.0).
        momentum: Momentum factor (default: 0.9).
        weight_decay: Weight decay (L2 penalty) (default: 1e-6).
        trust_coefficient: Trust coefficient for the ratio computation (eta) (default: 0.001).
        eps: Epsilon to avoid division by zero.
    """
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-6,
                 trust_coefficient=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            trust_coeff = group['trust_coefficient']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 1. Compute Weight Norm and Grad Norm
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(grad.data)

                # 2. Compute Local Learning Rate (Trust Ratio)
                if p_norm != 0 and g_norm != 0:
                    # LARS Trust Ratio: eta * ||w|| / (||g|| + wd * ||w||)
                    denominator = g_norm + weight_decay * p_norm
                    local_lr = trust_coeff * p_norm / (denominator + eps)
                    
                    # Scale the standard LR by the local trust ratio
                    actual_lr = lr * local_lr
                else:
                    actual_lr = lr

                # 3. Apply Weight Decay to Gradient (Standard SGD-W style)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # 4. Apply Momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1) # buf = momentum * buf + grad

                # 5. Update Weights
                # p = p - actual_lr * buf
                p.data.add_(buf, alpha=-actual_lr)

        return loss

# 2. Separate Parameters (SimCLR Best Practice)
# We exclude Bias and BatchNorm from LARS adaptation/Weight Decay for stability
param_weights = []
param_biases = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'bias' in name or 'norm' in name or 'bn' in name:
            param_biases.append(param)
        else:
            param_weights.append(param)

# 3. Initialize LARS Optimizer OUTSIDE the function
# SimCLR typically uses a very high LR (e.g., 0.2 to 4.0) with LARS
optimizer = LARS(
    [
        {'params': param_weights, 'weight_decay': 1e-6},
        {'params': param_biases, 'weight_decay': 0.0, 'trust_coefficient': 1.0} # No decay/adapt for biases
    ],
    lr=lr, # LARS allows higher LRs
    momentum=0.9,
    trust_coefficient=0.001
)

# cosine scheduler with warmup, most commonly used for transformer
warmup_steps = int(0.1 * epochs)   # 10% warmup (good default) - LR will increase linearly for the first 10% of training, then follow cosine decay for the remaining 90% of training. This helps stabilize training in the early stages and can lead to better convergence.
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
save_path = f'IY017_simCLR_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_{timestamp}_model.pth'
# # checkpoint save path
# save_path = model_path

# === wandb config (required for tracking within train_model) ===
wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY017-SSL-model",
    "name": f"simCLR_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_trj{num_traj}_len{sample_len}_{timestamp}", # change this to what you want
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
    "save_path": save_path,
    "nce_temp": nce_temp,
    "grad_clip": grad_clip,
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
# torch.save(model.state_dict(), model_path)

# clear cuda cache
torch.cuda.empty_cache()