"""
SimCLR pretraining on EXPERIMENTAL (not synthetic) time-lapse data, with
noise augmentation added on top of IY035_simclr_training_cross_view.py.

Everything else (dense pairing via ssl_exp_data_prep, model architecture,
optimizer, scheduler, loss) is identical to that script -- see its docstring
for the rationale behind ssl_exp_data_prep and sample_len=200. The only
addition here is augment_batch(), applying independent Gaussian noise to
each of the two contrastive views. Starting with just noise -- amplitude
scaling and temporal shift (both present in IY034_supcon_backbone_finetune's
augment_batch, which this was adapted from) are left out for now; add them
back in once noise alone has been evaluated.

Why: create_view's random-crop (sample_len=200 vs raw 494/540 tp) is
currently the ONLY source of view-to-view diversity. SimCLR's own ablations
found augmentation composition/strength is the single biggest lever for
representation quality -- bigger than architecture or batch size -- so
adding noise on top of the crop should push further in the same direction.

Implementation: rather than touching dataloaders.simclr (which builds and
returns already-constructed DataLoaders), this rewraps the SAME underlying
SimCLR_ExpDataset objects (via train_loader.dataset etc.) in new DataLoaders
that use a custom collate_fn -- default collation followed by augment_batch()
on each view -- keeping the augmentation logic local to this script. DataLoader
settings (batch_size, shuffle, num_workers, drop_last) are copied from
ssl_exp_data_prep's own construction for consistency.
"""

import re
import tempfile
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataloaders.simclr import ssl_exp_data_prep
from models.ssl_transformer import SSL_Transformer
from training.train import train_ssl_model, cross_view_info_nce
from utils.processing.imputation import fill_nans

IY035_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY035")
IY008_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY008")
FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series"

META_COLS = ["id", "group", "experiment"]
# Same convention as IY031/IY034: filename -> (exp_id, group_id, channel), and
# exp 18446 excluded project-wide (not properly recorded).
FILE_RE = re.compile(r"^(\d+)_.*_group_(.+?)_(GFP|mCherry)_time_series$")
EXCLUDED_EXPS = {"18446"}


def augment_batch(X: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """Gaussian noise only, to start with -- amplitude scaling and temporal
    shift (both present in IY034_supcon_backbone_finetune's augment_batch,
    which this was ported from) are left out for now. Operates on a batch of
    shape (B, T, C): adds independent N(0, noise_std^2) to every timepoint.
    """
    return X + torch.randn_like(X) * noise_std


def augmented_collate_fn(batch):
    """Default collation, then augment_batch() applied independently to each
    of the two views -- independent noise draws so v1/v2 differ, maximising
    the contrastive task's view diversity.
    """
    z1_list, z2_list, y_list = zip(*batch)
    z1 = torch.stack(z1_list, dim=0)
    z2 = torch.stack(z2_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return augment_batch(z1), augment_batch(z2), y


with tempfile.TemporaryDirectory(prefix="iy035_experimental_trajectories_") as tmp_dir:
    tmp_dir = Path(tmp_dir)

    # === Convert labelled experimental CSVs -> clean per-file .npz (scratch dir) ===
    n_converted = 0
    for csv_path in sorted(FULL_DATA_DIR.glob("*.csv")):
        match = FILE_RE.match(csv_path.stem)
        if match is None or match.group(1) in EXCLUDED_EXPS:
            continue
        df = pd.read_csv(csv_path)
        time_cols = [c for c in df.columns if c not in META_COLS]
        trajectories = fill_nans(df[time_cols].to_numpy(dtype=np.float64)).astype(np.float32)
        np.savez_compressed(tmp_dir / f"{csv_path.stem}.npz", trajectories=trajectories)
        n_converted += 1
    print(f"Converted {n_converted} experimental CSVs -> {tmp_dir} (scratch, deleted on exit)")
    # === Convert labelled experimental CSVs -> clean per-file .npz (scratch dir) ===

    TRAJ_PATH = sorted(tmp_dir.glob("*.npz"))
    print(f"Total trajectory files (experimental, TF@condition contexts): {len(TRAJ_PATH)}")

    # === Dataloader hyperparams & data prep ===
    batch_size = 1024
    num_traj = 1  # number of trajectories (cells) per view
    sample_len = 250  # well below raw trace length (494/540 tp) -- see module docstring
    log_scale = False  # values are already bounded fluorescence ratios (~0.01-0.4), not counts
    normalisation = 'instance'

    # num_groups_{train,val,test}: pairs/epoch, stated directly rather than
    # derived from file/cell counts. Lazily drawn -- no upfront generation
    # pass, and a fresh set of draws every epoch.
    total_num_groups = 50000
    num_groups_train = int(0.8 * total_num_groups)
    num_groups_val = int(0.1 * total_num_groups)
    num_groups_test = int(0.1 * total_num_groups)

    train_loader, val_loader, test_loader = ssl_exp_data_prep(
        TRAJ_PATH,
        batch_size=batch_size,
        sample_len=sample_len,
        log_scale=log_scale,
        normalisation=normalisation,
        num_traj=num_traj,
        num_groups_train=num_groups_train,
        num_groups_val=num_groups_val,
        num_groups_test=num_groups_test,
        verbose=True,
    )

    # Rewrap the same underlying datasets with augmented_collate_fn -- same
    # DataLoader settings ssl_exp_data_prep itself uses (batch_size, shuffle,
    # num_workers, drop_last), just a different collate_fn.
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                               num_workers=4, drop_last=True, collate_fn=augmented_collate_fn)
    val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, collate_fn=augmented_collate_fn)
    test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, collate_fn=augmented_collate_fn)
    # === Dataloader hyperparams & data prep ===

    # === Model hyperparams ===
    # Same architecture as the synthetic-data checkpoints (IY017/022/023/024) for
    # direct embedding comparability downstream.
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
    patience = epochs // 10
    lr = 3e-3
    weight_decay = 1e-4  # AdamW regularization on backbone + projection head
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_steps = int(0.1 * epochs)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs,
    )

    nce_temp = 0.2
    # Cross-view InfoNCE: pools both view banks into a single 2B negative set,
    # exposing same-side cross-file pairs (A↔C, B↔D) that standard InfoNCE ignores.
    # Self-comparisons (A↔A, B↔B) are masked to -inf.
    loss_fn = lambda q, k: cross_view_info_nce(q, k, temperature=nce_temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_clip = None
    verbose = True

    model.to(device)
    # === Training hyperparams ===

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = str(IY035_DIR / f'IY035_simCLR_experimental_augmented_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_{normalisation}_len{sample_len}_{timestamp}_model.pth')

    # === wandb config ===
    wandb_config = {
        "entity": "grignard-reagent",
        "project": "IY035-SSL-model",
        "name": f"simCLR_experimental_augmented_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_trj{num_traj}_len{sample_len}_{normalisation}_{timestamp}",
        "dataset": str(FULL_DATA_DIR),
        "augmentation": "noise",
        "num_groups_train": num_groups_train,
        "num_groups_val": num_groups_val,
        "num_groups_test": num_groups_test,
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
        "weight_decay": weight_decay,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
        "loss_fn": "cross_view_info_nce",
        "model": type(model).__name__,
        "num_traj_per_view": num_traj,
        "sample_len": sample_len,
        "log_scale": log_scale,
        "normalisation": normalisation,
        "save_path": save_path,
        "nce_temp": nce_temp,
        "grad_clip": grad_clip,
        "total_trajectory_files": len(TRAJ_PATH),
    }
    # === wandb config ===

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

# === Loss / accuracy curves ===
plt.rcParams.update({
    "font.family": "sans-serif", "axes.titlesize": 14,
    "axes.labelsize": 12, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10,
})
palette = sns.color_palette("colorblind")
epochs_ran = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].plot(epochs_ran, history["train_loss"], label="Train", color=palette[0])
axes[0].plot(epochs_ran, history["val_loss"], label="Val", color=palette[1])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-view InfoNCE loss")
axes[0].legend(loc="best")
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, history["train_acc"], label="Train", color=palette[0])
axes[1].plot(epochs_ran, history["val_acc"], label="Val", color=palette[1])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Contrastive batch accuracy")
axes[1].legend(loc="best")
axes[1].grid(alpha=0.3)

fig.suptitle("IY035 SimCLR pretraining on experimental data (augmented)", fontsize=14)
fig_path = IY035_DIR / f"IY035_simclr_experimental_augmented_training_curves_{timestamp}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")
