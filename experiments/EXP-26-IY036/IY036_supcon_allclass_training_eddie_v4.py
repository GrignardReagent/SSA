"""
IY036 v4, Eddie HPC submission variant -- the same experiment as
IY036_supcon_allclass_training_v4.py (same data prep, augmentation, checkpoint-
selection metric, outputs), differing only in launch route and in the
optimizer/schedule that the launch route makes affordable: this one is
submitted via `qsub IY036_supcon_allclass_training_eddie_v4.sh` on the
University of Edinburgh Eddie cluster, at a large batch with LARS and a much
longer epoch budget, where the local script uses AdamW at batch 512. Kept as a
separate frozen file (rather than pointing the job script at the local .py) so
this run's exact submitted code stays a stable, standalone record, per this
repo's convention of one .py per named experiment run.

Supervised-Contrastive (SupCon) pretraining on ALL experimental
TF@condition classes, evaluated with the IY031 SVM readout.

WHAT CHANGED IN v4 (and why)
----------------------------
v4 is v1's script with ONE addition (4, below); the three other bullets are
deliberate ROLLBACKS of v2/v3, each undone for a stated reason. v1, v2 and v3
are left untouched as the record of the runs already reported.

1. DATA PIPELINE ROLLED BACK TO v1 (`prepare_dataset`, balanced to the
   least-populated class -> 78/class, 468 cells, 299 fit / 75 val / 94 test).
   v2/v3 kept all 1,099 benchmark cells on a 60/20/20 split and reported
   balanced accuracy. That is a better use of the data in principle, but it
   moved the split, so IY031/IY032's numbers stopped being row-comparable and
   every v2/v3 result had to carry its own re-run baselines. v4 goes back to
   the split those reference numbers were measured on.

2. KNN ROLLED BACK TO A PLAIN `KNeighborsClassifier` (no prior correction),
   and the SVMs back to no `class_weight="balanced"`. Both of those existed in
   v2/v3 only to compensate for the unbalanced split of (1). With the classes
   balanced by subsampling, the priors are already uniform and re-weighting
   would correct a bias that is no longer there. Accuracy (not balanced
   accuracy) is likewise the right metric again on a balanced set.

3. AUGMENTATION IS WINDOW SLICING ALONE (`WINSLICE_RATIO`), not v2's lone
   jitter and not v3's four-op chain. v3's A/B found the composed recipe no
   better than jitter-only, so v4 tests the single time-axis operation on its
   own: crop 70% of the trace and stretch it back, i.e. the two views differ by
   a timing/phase offset and nothing else.

4. SMOOTHED CHECKPOINT SELECTION (kept from v3, the one thing v4 does NOT roll
   back). Selection runs on `knn_val_acc_smooth`, the trailing mean of the last
   SMOOTH_WINDOW raw evals. Selecting on the raw value, as v1/v2 did, means
   taking the maximum of hundreds of noisy estimates: v2's run peaked at
   knn_val_acc 0.8464 at epoch 250 against a plateau of 0.7969 +/- 0.0188
   (2.6 sd above it), and because nothing ever beat that spike, early stopping
   fired at epoch 1370 of 3360. Smoothing means a lucky spike can neither win
   the checkpoint nor trigger the stop. That matters far more here than in the
   local script, since this variant's budget is 3360 epochs, not 200.

WHY THIS EXISTS
---------------
Every IY035 run sampled *pairs* (2 cells from one file) as the unit of data.
That has three problems: (i) the number of positive pairs per epoch is capped by
the file count, which caps batch size; (ii) the pair space is combinatorially
vast (635k train pairs), so capping the number sampled makes each run a lottery
draw -- runs are not reproducible; (iii) it wastes ~99% of cells per epoch.

The fix implemented here is to stop enumerating pairs entirely: the dataset
emits individual CELLS with a label, and `SupConLoss` (pytorch-metric-learning)
forms every same-label positive and every different-label negative *within
the batch*. Consequences:
  * batch size is decoupled from the file count,
  * an epoch is a deterministic seeded permutation over all cells (every cell
    seen exactly once) -- so runs reproduce exactly given a seed,
  * B forward passes yield O(B^2) pair relations instead of B pairs from 2B
    passes.

IMPORTANT CAVEAT (measured, not assumed)
----------------------------------------
For the 6 downstream classes, TF@condition is perfectly aliased with file:
each of the 6 classes maps to exactly ONE file/experiment. So SupCon keyed on
TF@condition over only those 6 would give positives *identical* to the
same-file scheme -- zero cross-file pressure. To get any genuine cross-file
signal we therefore pretrain over ALL 32 TF@condition classes, 5 of which do
span more than one experiment (the Msn2 classes). Even so, condition and
imaging session remain largely confounded in this dataset, so this script is
best read as a *supervised representation baseline*, NOT as a fix for batch
effects. The companion script (IY036_msn2_cross_session.py) tests
batch-invariance properly via a leave-session-out split.

LEAKAGE CONTROL
---------------
The downstream 6-class test cells are excluded from SupCon pretraining by
exact trace matching (asserted: exactly 94 cells removed). The per-timepoint
StandardScaler is fit on the downstream TRAIN split only and reused for both
pretraining inputs and evaluation, so the encoder sees one consistent input
distribution and no test statistics leak.

DOWNSTREAM READOUT (IY031-comparable)
-------------------------------------
Encode -> StandardScale embeddings -> RBF-SVM (C=1.0, gamma='scale', seed 42),
i.e. exactly IY031's `run_simclr_svm`. Note this keeps the embedding
StandardScaler, unlike the KNN readouts (IY032/IY035) where scaling was
dropped -- for an RBF-SVM feature scaling is genuinely required, and matching
IY031 is the point of this script.
"""

import re
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from pytorch_metric_learning.losses import SupConLoss

from models.ssl_transformer import SSL_Transformer
from training.train import train_supcon_model
from training.wandb_utils import wandb_log
from open_lars import LARS
from utils.embeddings import encode_channel, knn_downstream_accuracy
from utils.experiment_tracking import run_timestamp
from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.processing.imputation import fill_nans
from utils.processing.pipeline import prepare_dataset
from utils.augmentation import window_slice_torch

# ── Config ────────────────────────────────────────────────────────────────────
# Relative to this script's own location, not hardcoded to one machine's home
# directory -- Eddie's filesystem layout under the repo root may not match local.
IY036_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = IY036_DIR.parent
IY031_DIR = EXPERIMENTS_DIR / "EXP-26-IY031"
IY008_DIR = EXPERIMENTS_DIR / "EXP-25-IY008"
FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series"
META_PATH = IY008_DIR / "old_data_metadata.csv"
META_COLS = ["id", "group", "experiment"]

FILE_RE = re.compile(r"^(\d+)_.*_group_(.+?)_(GFP|mCherry)_time_series$")
EXCLUDED_EXPS = {"18446"}
FIXED_CLASSES = [
    "Nrg1 @ 0.01% glucose", "Nrg1 @ 0.1% glucose", "Nrg1 @ 2% glucose (mock/steady)",
    "Rtg1 @ 0.01% glucose", "Rtg1 @ 0.1% glucose", "Rtg1 @ 2% glucose (mock/steady)",
]
RANDOM_STATE = 42
SEQ_LEN = 540          # downstream traces are 540 tp; shorter files are resampled up
WINSLICE_RATIO = 0.70  # window-slice augmentation for the two SupCon views

# training
batch_size = 4096   # Khosla et al. (SupCon) used 6144 on resnet-50, 4096 on resnet-200
epochs = 3360
# SimCLR/SupCon-lineage linear-scaling rule for LARS (Chen et al. 2020, "A Simple
# Framework for Contrastive Learning..."): lr = 0.3 * batch_size / 256. This is a
# very different scale to the 1e-3 tuned for AdamW in the local (non-Eddie) script
# -- LARS's own per-layer trust ratio already normalises step size per layer, so
# the *global* lr it expects is 1-2 orders of magnitude larger than a typical Adam lr.
lr = 0.3 * batch_size / 256
lars_momentum = 0.9              # standard SGD-style momentum used inside LARS (Khosla et al.)
lars_trust_coefficient = 0.001   # "eta" in the LARS paper (You, Gitman & Ginsburg 2017); standard default
weight_decay = 1e-4
temperature = 0.07     # well-established alternative to the SupCon default of 0.1 (Khosla et al.)
eval_every = 10        # epochs between checkpoint-selection evaluations
patience = epochs // (eval_every * 3)  # early stopping patience in units of eval_every
# KNN drives checkpoint selection -- it has no kernel to compensate for messy embedding geometry, so it's a more direct probe of whether SupCon is actually clustering same-label cells together than SVM is; SVM (matching IY031) is reported, not selected on. Selecting on a genuine held-out VAL split (not train-cv, not train accuracy) matters because training accuracy can't detect the downstream classifier overfitting to the training pool.
# v4: the SMOOTHED variant is what selects -- trailing mean of the last SMOOTH_WINDOW evals, so a
# single lucky spike can neither win the checkpoint nor trip early stopping (see docstring, item 4).
SMOOTH_WINDOW = 3
eval_metric_key = "knn_val_acc_smooth"
VAL_FRACTION = 0.20    # reused for both the pretraining-pool SupCon val-loss carve AND the
# downstream 6-class train/val carve below
K_NEIGHBORS = 10       # KNN downstream readout (matches IY032/IY035's grid-search k)

# model (identical architecture to the IY031/IY032 checkpoints for comparability)
d_model, nhead, num_layers, dropout = 16, 4, 2, 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Full determinism: seeded epochs are the whole point of the cell-sampling reframe.
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)


# ── Data ──────────────────────────────────────────────────────────────────────
metadata = pd.read_csv(META_PATH)
metadata["exp_id"] = metadata["exp_id"].astype(str)
metadata["group_id"] = metadata["group_id"].astype(str)
LABEL_LOOKUP = {(r.exp_id, r.group_id, r.channel): (r.tf, r.condition)
                for _, r in metadata.iterrows()}

full_ts_raw, full_label_strs = load_labelled_time_series_csvs(
    data_dir=FULL_DATA_DIR, file_re=FILE_RE, label_lookup=LABEL_LOOKUP,
    meta_cols=META_COLS, excluded_exps=EXCLUDED_EXPS, verbose=False,
)

# Downstream 6-class evaluation set -- identical call to IY031/IY032/IY034.
d = prepare_dataset(full_ts_raw, full_label_strs, FIXED_CLASSES, "Full", RANDOM_STATE)
n_cls_eval = len(d["class_names"])
chance = 1.0 / n_cls_eval

# Reproduce prepare_dataset's scaler (fit on the TRAIN split only -> no leakage)
scaler_in = StandardScaler().fit(d["X_train_raw"])
assert np.allclose(scaler_in.transform(d["X_test_raw"]), d["X_test"], atol=1e-6), \
    "input scaler does not reproduce prepare_dataset's normalisation"

# Held out from the downstream 6-class TRAIN split ONLY -- d["X_test"] stays untouched until the one-shot final evaluation. Used by eval_fn below for genuine validation accuracy during SupCon training (replaces the previous CV-on-train approach)
X_down_tr, X_down_val, y_down_tr, y_down_val = train_test_split(
    d["X_train"], d["y_train"], test_size=VAL_FRACTION,
    random_state=RANDOM_STATE, stratify=d["y_train"])
print(f"Downstream train/val split: {len(y_down_tr)} train, {len(y_down_val)} val "
      f"(val_fraction={VAL_FRACTION})")

# ── Build the SupCon pretraining pool: all 32 classes, minus downstream test cells ──
test_hashes = {row[:SEQ_LEN].tobytes() for row in d["X_test_raw"].astype(np.float64)}

def _to_seq_len(row: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    """Truncate (or uniformly linspace-upsample) one trace to `target` points."""
    if len(row) == target:
        return row
    if len(row) > target:
        return row[:target]
    idx = np.round(np.linspace(0, len(row) - 1, target)).astype(int)
    return row[idx]

X_pre, y_pre_str, n_excluded = [], [], 0
lbl_iter = iter(full_label_strs)
for ts in full_ts_raw:
    ts_imp = fill_nans(np.asarray(ts, dtype=float))   # same path as prepare_dataset
    for row in ts_imp:
        lbl = next(lbl_iter)
        # drop the 94 downstream TEST cells (exact match on the raw truncated trace)
        if len(row) >= SEQ_LEN and row[:SEQ_LEN].tobytes() in test_hashes:
            n_excluded += 1
            continue
        X_pre.append(_to_seq_len(row))
        y_pre_str.append(lbl)

assert n_excluded == len(d["y_test"]), \
    f"expected to exclude {len(d['y_test'])} downstream test cells, excluded {n_excluded}"

X_pre = np.vstack(X_pre)
pre_class_names = sorted(set(y_pre_str))
# Map string labels to integer indices for SupCon pretraining.
pre_l2i = {c: i for i, c in enumerate(pre_class_names)}
y_pre = np.array([pre_l2i[l] for l in y_pre_str])
X_pre = scaler_in.transform(X_pre).astype(np.float32)   # same normalisation as eval

print(f"SupCon pretraining pool : {X_pre.shape[0]} cells x {X_pre.shape[1]} tp, "
      f"{len(pre_class_names)} TF@condition classes "
      f"({n_excluded} downstream test cells excluded)")
print(f"Downstream eval (6-class): train {len(d['y_train'])}, test {len(d['y_test'])}, "
      f"chance {chance:.4f}")

class_counts = pd.Series(y_pre).value_counts()
print(f"Pretraining class sizes: min={class_counts.min()}, max={class_counts.max()}, "
      f"n_classes={len(class_counts)}")

# Stratified val carve, held out purely to monitor SupCon val loss (not used for
# downstream eval or checkpoint selection -- eval_fn's SVM/KNN readout drives that).
try:
    X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(
        X_pre, y_pre, test_size=VAL_FRACTION, random_state=RANDOM_STATE, stratify=y_pre)
except ValueError:
    # some class too small to stratify at this fraction -- keep those classes entirely
    # in train and only carve the val split from classes with enough members.
    min_needed = max(2, int(np.ceil(1 / VAL_FRACTION)))
    splittable = class_counts[class_counts >= min_needed].index
    split_mask = np.isin(y_pre, splittable)
    X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(
        X_pre[split_mask], y_pre[split_mask], test_size=VAL_FRACTION,
        random_state=RANDOM_STATE, stratify=y_pre[split_mask])
    X_pre_tr = np.vstack([X_pre_tr, X_pre[~split_mask]])
    y_pre_tr = np.concatenate([y_pre_tr, y_pre[~split_mask]])
    print(f"  ({(~split_mask).sum()} cells from small classes kept entirely in train)")

print(f"SupCon train/val split: {len(y_pre_tr)} train, {len(y_pre_val)} val "
      f"(val_fraction={VAL_FRACTION})")


class CellDataset(Dataset):
    """Emits ONE cell per item -- (trace, label). The multi-positive SupCon loss
    forms all same-label pairs inside the batch, so no pair enumeration is
    needed and __len__ is simply the number of cells (deterministic epochs)."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # take one row, convert from numpy to torch, ensure float, insert new trailing dimension (540, ) -> (540, 1)
        return torch.from_numpy(self.X[i]).float().unsqueeze(-1), int(self.y[i])

# deterministic shuffling by using a seeded generator (torch's DataLoader RNG is per-worker, so can't be seeded directly)
loader_gen = torch.Generator().manual_seed(RANDOM_STATE)
train_loader = DataLoader(
    CellDataset(X_pre_tr, y_pre_tr), batch_size=batch_size, shuffle=True,
    num_workers=4, drop_last=True, generator=loader_gen,
)
# NOT batch_size: an eval-mode (model.eval() + torch.no_grad()) forward pass
# measured ~2.3-2.5x MORE peak memory per sample than a full train step
# (forward+backward+optimizer.step()) at the same batch size on this
# architecture -- profiled empirically, likely an SDPA backend difference
# between grad/no-grad mode. Reusing the (large) training batch_size here
# would make the val loop the actual OOM bottleneck, not the training step.
VAL_BATCH_SIZE = 256
val_loader = DataLoader(
    CellDataset(X_pre_val, y_pre_val), batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4,
)
print(f"Batches/epoch: {len(train_loader)}  (batch_size={batch_size})")


# ── Model ─────────────────────────────────────────────────────────────────────
model = SSL_Transformer(input_size=1, d_model=d_model, nhead=nhead,
                        num_layers=num_layers, dropout=dropout,
                        use_conv1d=False).to(DEVICE)
# LARS: unlike the local script's small batch_size (where AdamW is the pragmatic
# choice), this Eddie run uses a genuinely large batch (see `batch_size`), which is exactly
# the regime LARS exists for. Uses the open_lars package (pip) -- verified
# bit-for-bit identical to this repo's prior hand-rolled implementation before
# swapping it in; see requirements.yml.
optimizer = LARS(model.parameters(), lr=lr, momentum=lars_momentum,
                 weight_decay=weight_decay, trust_coefficient=lars_trust_coefficient)
supcon_criterion = SupConLoss(temperature=temperature)  # pytorch-metric-learning

# Linear-warmup + cosine-decay is the standard schedule paired with LARS in the
# SimCLR/SupCon lineage (Chen et al. 2020 use exactly this: 10 epochs of linear
# warmup, then cosine decay without restarts) -- LARS is known to be unstable
# very early in training at large batch/lr, and warmup is the standard fix
# (Goyal et al. 2017). Kept as-is here; only the optimizer/lr changed above.
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * epochs), num_training_steps=epochs)

def svm_eval(model):
    """IY031's readout: encode -> StandardScale embeddings -> RBF-SVM.

    Fit on the FULL downstream train split (d["X_train"]), evaluated on the
    held-out test split (d["X_test"]) -- computed once, at the end (see 'Final
    evaluation' below), never tracked per-epoch or used for checkpoint selection.
    """
    model.eval()
    Z_tr = encode_channel(model, d["X_train"], DEVICE)
    Z_te = encode_channel(model, d["X_test"], DEVICE)
    sc = StandardScaler()
    Z_tr_sc, Z_te_sc = sc.fit_transform(Z_tr), sc.transform(Z_te)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(Z_tr_sc, d["y_train"])
    y_pred = clf.predict(Z_te_sc)
    model.train()
    return accuracy_score(d["y_test"], y_pred), y_pred


# Trailing window of raw knn_val_acc values, feeding the smoothed selection
# metric. STATEFUL, but this script trains once, so it is never reset.
_knn_val_window: deque = deque(maxlen=SMOOTH_WINDOW)


def eval_fn(model):
    """
    Periodic downstream train/val readout on a genuine held-out split of the
    6-class benchmark's TRAIN portion (X_down_tr/X_down_val) -- d["X_test"] is
    never touched here, and is reserved for the one-shot final evaluation below.

    knn_val_acc drives checkpoint selection: unlike SVM's RBF kernel, KNN has no
    way to compensate for messy embedding geometry, so it's a more direct probe
    of whether SupCon is actually clustering same-label cells together.

    v4 selects on `knn_val_acc_smooth` -- the trailing mean of the last
    SMOOTH_WINDOW raw values -- rather than on the raw value, since the raw
    maximum over the ~336 evals in this budget reliably picks a spike (see
    docstring, item 4). The raw value is still returned so the spike/plateau
    behaviour stays visible in the history CSV and in wandb. Known caveat: a
    checkpoint saved at a smoothed peak holds the weights from the END of the
    window, not its centre.

    knn_train_acc / svm_train_acc are tracked alongside PURELY to visualise the
    train/val gap (a widening gap = overfitting) -- training accuracy is never
    used to select a checkpoint or to stop, since it can't detect overfitting
    by construction (it's evaluated on the same data the classifier was fit on).
    """
    model.eval()
    Z_tr = encode_channel(model, X_down_tr, DEVICE)
    Z_val = encode_channel(model, X_down_val, DEVICE)
    model.train()

    # KNN: raw embeddings, no StandardScaler (established convention -- an
    # ablation over IY035 checkpoints found embedding scaling a wash that
    # homogenises checkpoints, suppressing the best ones).
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric="euclidean", n_jobs=-1)
    knn.fit(Z_tr, y_down_tr)
    knn_train_acc = accuracy_score(y_down_tr, knn.predict(Z_tr))
    knn_val_acc = accuracy_score(y_down_val, knn.predict(Z_val))

    # SVM: StandardScaled embeddings (matches IY031's methodology)
    sc = StandardScaler().fit(Z_tr)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(sc.transform(Z_tr), y_down_tr)
    svm_train_acc = accuracy_score(y_down_tr, clf.predict(sc.transform(Z_tr)))
    svm_val_acc = accuracy_score(y_down_val, clf.predict(sc.transform(Z_val)))

    _knn_val_window.append(knn_val_acc)
    return {"knn_val_acc_smooth": float(np.mean(_knn_val_window)),
            "knn_val_acc": knn_val_acc, "svm_val_acc": svm_val_acc,
            "knn_train_acc": knn_train_acc, "svm_train_acc": svm_train_acc}


# ── Train ─────────────────────────────────────────────────────────────────────
# Reuse the job script's RUN_TIMESTAMP so the .out log and these artifacts match
timestamp = run_timestamp()
save_path = IY036_DIR / (f"IY036_v4_winslice_b{batch_size}_lr{lr}_"
                         f"L{num_layers}_H{nhead}_D{d_model}_{timestamp}_model.pth")

wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY036-SupCon-model",
    "name": f"supcon_allclass_eddie_v4_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_temp{temperature}_{timestamp}",
    "dataset": str(FULL_DATA_DIR),
    "augmentation": "window_slicing", "window_reduce_ratio": WINSLICE_RATIO,
    "split_scheme": "v1/v4: balanced to the least-populated class, stratified 80/20 train/test, 20% of train carved for val",
    "downstream_metric": "accuracy",
    "num_cells_pretrain_train": len(y_pre_tr), "num_cells_pretrain_val": len(y_pre_val),
    "num_cells_downstream_train_fit": len(y_down_tr), "num_cells_downstream_val": len(y_down_val),
    "val_fraction": VAL_FRACTION, "num_classes_pretrain": len(pre_class_names),
    "n_excluded_downstream_test_cells": n_excluded,
    "batch_size": batch_size, "input_size": 1, "d_model": d_model, "nhead": nhead,
    "num_layers": num_layers, "dropout": dropout, "use_conv1d": False,
    "epochs": epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay,
    "optimizer": type(optimizer).__name__, "lars_momentum": lars_momentum,
    "lars_trust_coefficient": lars_trust_coefficient, "scheduler": type(scheduler).__name__,
    "loss_fn": "SupConLoss", "temperature": temperature,
    "eval_every": eval_every, "eval_metric_key": eval_metric_key,
    "smooth_window": SMOOTH_WINDOW, "knn_neighbors": K_NEIGHBORS,
    "seq_len": SEQ_LEN, "save_path": str(save_path), "grad_clip": None,
}

print("\nStarting SupCon training...")
# return_run=True keeps the wandb run OPEN so the one-shot final test metrics
# below -- which can only be computed after training, on the reloaded best
# checkpoint -- can still be attached to it. A closed run cannot accept them.
history, wandb_run = train_supcon_model(
    model, train_loader, val_loader=val_loader,
    epochs=epochs, patience=patience, lr=lr, optimizer=optimizer, scheduler=scheduler,
    loss_fn=supcon_criterion,
    # augment each view by cropping WINSLICE_RATIO of the trace and stretching it
    # back to full length -- the two views then differ by a timing/phase offset
    augment_fn=lambda x: window_slice_torch(x, reduce_ratio=WINSLICE_RATIO),
    device=DEVICE, grad_clip=None, save_path=str(save_path),
    eval_fn=eval_fn, eval_every=eval_every, eval_metric_key=eval_metric_key,
    wandb_logging=True, wandb_config=wandb_config, verbose=True,
    return_run=True,
)
print(f"Saved: {save_path}")

# ── Final evaluation with the selected checkpoint ─────────────────────────────
model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
final_svm_test, y_pred = svm_eval(model)
print(f"\n=== IY036 SupCon + SVM (Full, 6-class) ===")
print(f"Test accuracy: {final_svm_test:.4f}  (chance {chance:.4f}, {final_svm_test - chance:+.4f})")
print(classification_report(d["y_test"], y_pred, target_names=d["class_names"]))

model.eval()
final_knn, y_pred_knn = knn_downstream_accuracy(
    model, d["X_train"], d["X_test"], d["y_train"], d["y_test"], DEVICE, n_neighbors=K_NEIGHBORS)
model.train()
print(f"\n=== IY036 SupCon + KNN (k={K_NEIGHBORS}, Full, 6-class) ===")
print(f"Test accuracy: {final_knn:.4f}  (chance {chance:.4f}, {final_knn - chance:+.4f})")
print(classification_report(d["y_test"], y_pred_knn, target_names=d["class_names"]))

# ── One-shot final metrics -> wandb ───────────────────────────────────────────
# Single values per run, so they belong in run.summary rather than run.log:
# summary values are what the wandb runs TABLE sorts and filters on, which is how
# runs get compared against each other. Also logged as a final step so they
# appear in this run's own charts alongside the per-epoch curves.
if wandb_run is not None:
    final_metrics = {
        # headline test numbers, on the reloaded best checkpoint
        "final/svm_acc": float(final_svm_test),
        "final/knn_acc": float(final_knn),
        "final/svm_acc_over_chance": float(final_svm_test - chance),
        "final/knn_acc_over_chance": float(final_knn - chance),
        "final/chance": float(chance),
        "final/n_test": int(len(d["y_test"])),
        # training diagnostics -- the smooth/raw gap is the spike that raw-value
        # selection would have chased, and epochs_run vs epochs_budget shows
        # whether early stopping fired before the budget was spent
        "final/best_knn_val_smooth": float(max(history["eval/knn_val_acc_smooth"])),
        "final/best_knn_val_raw": float(max(history["eval/knn_val_acc"])),
        "final/epochs_run": int(max(history["epoch"])),
        "final/epochs_budget": int(epochs),
    }
    wandb_run.summary.update(final_metrics)
    wandb_log(wandb_run, final_metrics)
    wandb_run.finish()       # train_supcon_model deliberately left it open
    print("Logged final test metrics to wandb summary.")

# Log and write the final test results along with the hyperparameters
final_results = [{
    "method": "IY036 SupCon, label-supervised",
    # run identity -- matches this run's .out log, checkpoint and history CSV
    "timestamp": timestamp,
    # test results
    "svm_accuracy": float(final_svm_test),
    "knn_accuracy": float(final_knn),
    "chance": float(chance),          # 1/n_classes, so accuracies stay interpretable
    # knn
    "k_neighbors": int(K_NEIGHBORS),
    # fixed classes (used for training)
    "fixed_classes": str(FIXED_CLASSES),
    # training hyperparameters
    "batch_size": int(batch_size),
    "epochs": int(epochs),
    "lr": float(lr),
    "val_fraction": float(VAL_FRACTION),
    "window_reduce_ratio": float(WINSLICE_RATIO),   # augmentation strength for the two SupCon views
    # optimizer -- recorded because the local (AdamW) and Eddie (LARS) variants
    # differ here, and lr is only comparable within one optimizer
    "optimizer": type(optimizer).__name__,
    # LARS hyperparameters
    "lars_momentum": float(lars_momentum),
    "lars_trust_coefficient": float(lars_trust_coefficient),
    "weight_decay": float(weight_decay),
    # SupCon hyperparameters
    "temperature": float(temperature),
    "patience": int(patience),
    "eval_metric_key": str(eval_metric_key),
    "smooth_window": int(SMOOTH_WINDOW),
    # training diagnostics -- the gap between these two is exactly the spike that
    # raw-value selection would have chased, and epochs_run vs epochs_budget shows
    # whether early stopping still fired prematurely (see docstring, item 4)
    "epochs_run": int(max(history["epoch"])),
    "epochs_budget": int(epochs),
    "best_knn_val_smooth": float(max(history["eval/knn_val_acc_smooth"])),
    "best_knn_val_raw": float(max(history["eval/knn_val_acc"])),
    # model architecture
    "d_model": int(d_model),
    "nhead": int(nhead),
    "num_layers": int(num_layers),
    "dropout": float(dropout),
    "seq_len": int(SEQ_LEN),
}]

# write final results + hyperparameters to CSV
final_df = pd.DataFrame(final_results)
out_path = IY036_DIR / f"IY036_v4_final_results_eddie_{timestamp}.csv"
final_df.to_csv(out_path, index=False)
print(f"Wrote final test results & hyperparameters to: {out_path}")

# write training history to csv
pd.DataFrame({k: pd.Series(v) for k, v in history.items()}).to_csv(
    IY036_DIR / f"IY036_v4_history_eddie_{timestamp}.csv", index=False)
