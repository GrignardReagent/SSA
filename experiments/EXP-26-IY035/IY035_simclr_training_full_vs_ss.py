"""
SimCLR pretraining on EXPERIMENTAL time-lapse data with a CROSS-SOURCE
positive-pair definition: full trajectory vs steady-state trajectory of the
SAME cell.

Motivation: the same-file positive scheme used by every other IY035 script
(two cells from one file) plateaus at ~raw-KNN downstream accuracy (0.72-0.77,
below the best synthetic checkpoint's 0.809) -- with only 41 training files it
degenerates into memorising per-file batch effects. This script instead pairs
each cell's FULL trace (view x1, whole t=0-18 h recording) with the SAME cell's
STEADY-STATE trace (view x2, the shorter t~5-10 h sub-window), so the learning
signal becomes "the full-window and steady-state-window views of one cell are
the same instance" -- observation-window invariance per cell, rather than file
identity.

Data facts (EXP-25-IY008):
- 4_transformed_exp_time_series (SS) and 5_FULL_transformed_exp_time_series
  (FULL) hold the SAME cells: identical `id` in identical row order for all 66
  matched file stems, so cell i's FULL trace and cell i's SS trace pair by row
  index -- no id-join needed (asserted below as a guard).
- FULL length = 494/540 tp (starts t=0); SS length = 24-165 tp (varies per file,
  a strict sub-window of FULL starting ~t=4.97 h). Exp 18446 exists only in SS
  and is excluded anyway.

Length handling: raw lengths can't be kept as-is (FULL has two lengths, SS has
many), so each view is resampled to a fixed PER-VIEW target -- FULL ->
SAMPLE_LEN_FULL, SS -> SAMPLE_LEN_SS -- preserving the long-vs-short contrast.
This is fine for training: model(x1, x2) encodes the two views through separate
backbone.encode() calls (each mean-pooled over time), so x1 and x2 may differ in
length; only WITHIN a batch, within a view, must lengths be uniform.

The whole paired dataset (~66 files, ~12k cells, ~30 MB) is held in memory --
no CSV->npz conversion or scratch tempfiles. All pairing logic is kept local to
this script (custom FullVsSSDataset); the shared dataloaders.simclr is left
untouched. Model, optimizer, scheduler, loss, wandb and checkpointing are copied
from IY035_simclr_training_cross_view.py.
"""

import re
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from dataloaders.simclr import _norm_instance
from models.ssl_transformer import SSL_Transformer
from training.train import train_ssl_model, cross_view_info_nce
from utils.processing.imputation import fill_nans

IY035_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY035")
IY008_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY008")
FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series"
SS_DATA_DIR = IY008_DIR / "4_transformed_exp_time_series"

META_COLS = ["id", "group", "experiment"]
# Same convention as IY031/IY034: filename -> (exp_id, group_id, channel), and
# exp 18446 excluded project-wide (not properly recorded).
FILE_RE = re.compile(r"^(\d+)_.*_group_(.+?)_(GFP|mCherry)_time_series$")
EXCLUDED_EXPS = {"18446"}


def _resample_view(trace: np.ndarray, target: int, training: bool) -> torch.Tensor:
    """Instance-norm one (T, 1) trace then resample it to `target` timepoints.

    Mirrors create_view's D-block in dataloaders.simclr: random-phase stride
    crop when the trace is long enough and we're training (augmentation),
    else deterministic uniform linspace (also up-samples short traces, e.g.
    steady-state, without padding).
    """
    if trace.ndim == 1:
        trace = trace.reshape(-1, 1)
    t = _norm_instance(torch.tensor(trace, dtype=torch.float32))

    curr = t.shape[0]
    stride = curr / target
    if curr >= target and training:
        phase = np.random.uniform(0, stride)
        idx = np.clip(np.round(phase + np.arange(target) * stride).astype(int), 0, curr - 1)
    else:
        idx = np.round(np.linspace(0, curr - 1, target)).astype(int)
    return t[idx]


class FullVsSSDataset(Dataset):
    """Cross-source SimCLR dataset: positive pair = (cell i's FULL trace,
    cell i's STEADY-STATE trace), row-aligned within one file stem.

    Lazy like SimCLR_ExpDataset: __len__ is a stated pair count (n_pairs) and
    __getitem__ IGNORES idx -- every call draws a random stem, a random cell
    row i in that stem, and returns (full[i] -> sample_len_full, ss[i] ->
    sample_len_ss). Arrays are held in memory (full_arrays[k], ss_arrays[k] are
    row-aligned by cell for stem k).
    """

    def __init__(self, full_arrays, ss_arrays, n_pairs, sample_len_full, sample_len_ss, training=True):
        self.full_arrays = full_arrays
        self.ss_arrays = ss_arrays
        self.n_pairs = n_pairs
        self.sample_len_full = sample_len_full
        self.sample_len_ss = sample_len_ss
        self.training = training

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # idx ignored: independent random (stem, cell) draw per call.
        k = np.random.randint(0, len(self.full_arrays))
        full, ss = self.full_arrays[k], self.ss_arrays[k]
        i = np.random.randint(0, full.shape[0])  # same cell row for both views
        x1 = _resample_view(full[i], self.sample_len_full, self.training)
        x2 = _resample_view(ss[i], self.sample_len_ss, self.training)
        y = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)  # dummy; loss ignores it
        return x1, x2, y


# === Build paired (full, ss) arrays in memory ===
# For each FULL stem also present in SS (and not excluded): drop metadata
# columns, impute NaNs, and keep both matrices row-aligned. ~30 MB total, so no
# CSV->npz conversion or scratch tempfiles are needed.
paired = []  # list of (stem, full_arr, ss_arr)
for csv_path in sorted(FULL_DATA_DIR.glob("*.csv")):
    match = FILE_RE.match(csv_path.stem)
    if match is None or match.group(1) in EXCLUDED_EXPS:
        continue
    ss_path = SS_DATA_DIR / csv_path.name
    if not ss_path.exists():
        continue

    df_full = pd.read_csv(csv_path)
    df_ss = pd.read_csv(ss_path)
    full_cols = [c for c in df_full.columns if c not in META_COLS]
    ss_cols = [c for c in df_ss.columns if c not in META_COLS]
    full_arr = fill_nans(df_full[full_cols].to_numpy(dtype=np.float64)).astype(np.float32)
    ss_arr = fill_nans(df_ss[ss_cols].to_numpy(dtype=np.float64)).astype(np.float32)

    # FULL and SS are the same cells in identical row order -- guard it.
    assert full_arr.shape[0] == ss_arr.shape[0], (
        f"row mismatch for {csv_path.stem}: full={full_arr.shape[0]} ss={ss_arr.shape[0]}")

    paired.append((csv_path.stem, full_arr, ss_arr))
print(f"Built {len(paired)} paired (full, ss) contexts in memory")
# === Build paired arrays ===

# === Dataloader hyperparams & data prep ===
batch_size = 1024
sample_len_full = 480  # FULL view target length (~raw 494/540, lightly cropped)
sample_len_ss = 120    # steady-state view target length (raw 24-165 tp)
seed = 42

# num_groups_{train,val,test}: pairs/epoch, stated directly. Lazily drawn --
# no upfront generation pass, fresh (stem, cell) draws every epoch.
num_groups_train = 20000
num_groups_val = 5000
num_groups_test = 5000

# 80/20/20 split over stems (held-out contexts for val/test), same scheme/seed
# as dataloaders.simclr.ssl_exp_data_prep.
train_paired, test_paired = train_test_split(paired, test_size=0.2, random_state=seed)
train_paired, val_paired = train_test_split(train_paired, test_size=0.2, random_state=seed)
print(f"Stems split: {len(train_paired)} Train, {len(val_paired)} Val, {len(test_paired)} Test")


def make_loader(paired_split, n_pairs, shuffle, drop_last):
    ds = FullVsSSDataset(
        [p[1] for p in paired_split], [p[2] for p in paired_split],
        n_pairs, sample_len_full, sample_len_ss, training=True,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)


train_loader = make_loader(train_paired, num_groups_train, shuffle=True, drop_last=True)
val_loader = make_loader(val_paired, num_groups_val, shuffle=False, drop_last=False)
test_loader = make_loader(test_paired, num_groups_test, shuffle=False, drop_last=False)
# === Dataloader hyperparams & data prep ===

# === Model hyperparams ===
# Same architecture as the synthetic-data checkpoints (IY017/022/023/024) for
# direct embedding comparability downstream.
X1_b, X2_b, y_b = next(iter(train_loader))
print(f"Batch shapes -- x1 (FULL): {tuple(X1_b.shape)}, x2 (SS): {tuple(X2_b.shape)}")
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
save_path = str(IY035_DIR / f'IY035_simCLR_full_vs_ss_b{batch_size}_lr{lr}_L{num_layers}_H{nhead}_D{d_model}_instance_full{sample_len_full}_ss{sample_len_ss}_{timestamp}_model.pth')

# === wandb config ===
wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY035-SSL-model",
    "name": f"simCLR_full_vs_ss_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_full{sample_len_full}_ss{sample_len_ss}_instance_{timestamp}",
    "dataset": f"FULL={FULL_DATA_DIR}; SS={SS_DATA_DIR}",
    "pairing": "full_vs_ss_same_cell",
    "sample_len_full": sample_len_full,
    "sample_len_ss": sample_len_ss,
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
    "normalisation": "instance",
    "save_path": save_path,
    "nce_temp": nce_temp,
    "grad_clip": grad_clip,
    "total_paired_stems": len(paired),
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