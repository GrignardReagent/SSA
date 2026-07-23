"""
SimCLR pretraining on EXPERIMENTAL (not synthetic) time-lapse data.

Positive-pair definition: two randomly drawn cell traces from the same raw
CSV (same exp_id/group_id/channel -> same TF at the same condition) are
treated as a positive pair, mirroring how the synthetic-data version treats
two trajectories from the same parameter-set file as a positive pair.

Only 66 raw files are available (vs. thousands of synthetic trajectory
files), but each holds ~180 cells on average, so a scheme that draws just one
pair per file per epoch wastes most of those cells every epoch.
dataloaders.simclr.ssl_exp_data_prep (SimCLR_ExpDataset) fixes this: you
state how many training/val/test pairs you want directly
(num_groups_train/val/test), same ergonomics as dataloaders.ssl.ssl_data_prep
(make_groups), but stays lazy -- nothing is pre-generated up front, each
__getitem__ call independently draws a fresh random file + pair on demand
(a fresh set of draws every epoch too), so raising num_groups_train doesn't
cost an upfront generation pass or a resident in-memory copy of the whole
dataset the way make_groups' pre-materialized group list did.

The labelled FULL_DATA_DIR CSVs carry non-numeric id/group/experiment
metadata columns that the dataloader doesn't know about, so they're first
converted to per-file .npz bundles in a scratch tempfile.TemporaryDirectory()
-- not saved into the experiment folder -- that is deleted automatically
once training finishes.

sample_len is set explicitly to 200, well below the raw trace length
(494/540 tp): create_view's random-crop augmentation only does anything
meaningful when sample_len is well below the trace length it's cropping
from -- at sample_len=None (auto-detecting to ~raw length) the "random crop"
degenerates to a ~1-sample jitter, which is close to no augmentation at all.
"""

import re
import tempfile
import torch
import torch.optim as optim
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
    sample_len = 200  # 200 - subsampling
    log_scale = False  # values are already bounded fluorescence ratios (~0.01-0.4), not counts
    normalisation = 'instance'

    # num_groups_{train,val,test}: pairs/epoch, stated directly rather than
    # derived from file/cell counts. Lazily drawn -- no upfront generation
    # pass, and a fresh set of draws every epoch (see module docstring).
    num_groups_train = 20000
    num_groups_val = 5000
    num_groups_test = 5000

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
    save_path = str(IY035_DIR / f'IY035_simCLR_experimental_cross_view_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_{normalisation}_len{sample_len}_{timestamp}_model.pth')

    # === wandb config ===
    wandb_config = {
        "entity": "grignard-reagent",
        "project": "IY035-SSL-model",
        "name": f"simCLR_experimental_cross_view_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_trj{num_traj}_len{sample_len}_{normalisation}_{timestamp}",
        "dataset": str(FULL_DATA_DIR),
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