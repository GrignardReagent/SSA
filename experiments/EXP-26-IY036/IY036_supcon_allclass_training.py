"""
IY036 (1/2): Supervised-Contrastive (SupCon) pretraining on ALL experimental
TF@condition classes, evaluated with the IY031 SVM readout.

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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from pytorch_metric_learning.losses import SupConLoss

from models.ssl_transformer import SSL_Transformer
from training.train import train_supcon_model
from utils.embeddings import encode_channel, knn_downstream_accuracy
from utils.experiment_tracking import run_timestamp
from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.processing.imputation import fill_nans
from utils.processing.pipeline import prepare_dataset
from utils.augmentation import jitter_torch

# ── Config ────────────────────────────────────────────────────────────────────
IY036_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY036")
IY031_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY031")
IY008_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY008")
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
NOISE_STD = 0.05       # Gaussian-noise augmentation for the two SupCon views

# training
batch_size = 512   # Khosla et al. (SupCon) used 6144 on resnet-50, 4096 on resnet-200
epochs = 200
lr = 1e-3
weight_decay = 1e-4
temperature = 0.1     # SupCon default (Khosla et al.), though 0.07 is a well-established alternative default
eval_every = 10        # epochs between checkpoint-selection evaluations
patience = epochs // (eval_every * 3)  # early stopping patience in units of eval_every
eval_metric_key = "knn_val_acc" # the metric used to select the best checkpoint -- KNN has no kernel to
# compensate for messy embedding geometry, so it's a more direct probe of whether SupCon is actually
# clustering same-label cells together than SVM is; SVM (matching IY031) is reported, not selected on.
# Selecting on a genuine held-out VAL split (not train-cv, not train accuracy) matters because
# training accuracy can't detect the downstream classifier overfitting to the training pool.
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

# Held out from the downstream 6-class TRAIN split ONLY -- d["X_test"] stays untouched
# until the one-shot final evaluation. Used by eval_fn below for genuine validation
# accuracy during SupCon training (replaces the previous CV-on-train approach: CV
# never leaves the training pool, so it can't detect the SVM/KNN classifier
# overfitting to it -- a true held-out split can).
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
val_loader = DataLoader(
    CellDataset(X_pre_val, y_pre_val), batch_size=batch_size, shuffle=False, num_workers=4,
)
print(f"Batches/epoch: {len(train_loader)}  (batch_size={batch_size})")


# ── Model ─────────────────────────────────────────────────────────────────────
model = SSL_Transformer(input_size=1, d_model=d_model, nhead=nhead,
                        num_layers=num_layers, dropout=dropout,
                        use_conv1d=False).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # ! Khosla used LARS ! But this is for large batch sizes only, for CNNs, not necessary for us. 
supcon_criterion = SupConLoss(temperature=temperature)  # pytorch-metric-learning

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


def eval_fn(model):
    """
    Periodic downstream train/val readout on a genuine held-out split of the
    6-class benchmark's TRAIN portion (X_down_tr/X_down_val) -- d["X_test"] is
    never touched here, and is reserved for the one-shot final evaluation below.

    knn_val_acc drives checkpoint selection: unlike SVM's RBF kernel, KNN has no
    way to compensate for messy embedding geometry, so it's a more direct probe
    of whether SupCon is actually clustering same-label cells together.

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

    return {"knn_val_acc": knn_val_acc, "svm_val_acc": svm_val_acc,
            "knn_train_acc": knn_train_acc, "svm_train_acc": svm_train_acc}


# ── Train ─────────────────────────────────────────────────────────────────────
# Run interactively (no job script), so this falls back to the current time --
# same helper as the Eddie variant, which instead inherits $RUN_TIMESTAMP
timestamp = run_timestamp()
save_path = IY036_DIR / (f"IY036_supcon_allclass_b{batch_size}_lr{lr}_"
                         f"L{num_layers}_H{nhead}_D{d_model}_{timestamp}_model.pth")

wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY036-SupCon-model",
    "name": f"supcon_allclass_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_temp{temperature}_{timestamp}",
    "dataset": str(FULL_DATA_DIR),
    "augmentation": "noise", "noise_std": NOISE_STD,
    "num_cells_pretrain_train": len(y_pre_tr), "num_cells_pretrain_val": len(y_pre_val),
    "num_cells_downstream_train_fit": len(y_down_tr), "num_cells_downstream_val": len(y_down_val),
    "val_fraction": VAL_FRACTION, "num_classes_pretrain": len(pre_class_names),
    "n_excluded_downstream_test_cells": n_excluded,
    "batch_size": batch_size, "input_size": 1, "d_model": d_model, "nhead": nhead,
    "num_layers": num_layers, "dropout": dropout, "use_conv1d": False,
    "epochs": epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay,
    "optimizer": type(optimizer).__name__, "scheduler": type(scheduler).__name__,
    "loss_fn": "SupConLoss", "temperature": temperature,
    "eval_every": eval_every, "eval_metric_key": eval_metric_key, "knn_neighbors": K_NEIGHBORS,
    "seq_len": SEQ_LEN, "save_path": str(save_path), "grad_clip": None,
}

print("\nStarting SupCon training...")
history = train_supcon_model(
    model, train_loader, val_loader=val_loader,
    epochs=epochs, patience=patience, lr=lr, optimizer=optimizer, scheduler=scheduler,
    loss_fn=supcon_criterion, augment_fn=lambda x: jitter_torch(x, sigma=NOISE_STD),  # augment each view with independent Gaussian noise
    device=DEVICE, grad_clip=None, save_path=str(save_path),
    eval_fn=eval_fn, eval_every=eval_every, eval_metric_key=eval_metric_key,
    wandb_logging=True, wandb_config=wandb_config, verbose=True,
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

# Log and write the final test results along with the hyperparameters
final_results = [{
    "method": "IY036 SupCon, label-supervised",
    # run identity -- matches this run's checkpoint and history CSV
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
    "noise_std": float(NOISE_STD),    # augmentation strength for the two SupCon views
    # optimizer -- recorded because the local (AdamW) and Eddie (LARS) variants
    # differ here, and lr is only comparable within one optimizer
    "optimizer": type(optimizer).__name__,
    "weight_decay": float(weight_decay),
    # SupCon hyperparameters
    "temperature": float(temperature),
    "patience": int(patience),
    "eval_metric_key": str(eval_metric_key),
    # model architecture
    "d_model": int(d_model),
    "nhead": int(nhead),
    "num_layers": int(num_layers),
    "dropout": float(dropout),
    "seq_len": int(SEQ_LEN),
}]

# write final results + hyperparameters to CSV
final_df = pd.DataFrame(final_results)
out_path = IY036_DIR / f"IY036_supcon_allclass_final_results_{timestamp}.csv"
final_df.to_csv(out_path, index=False)
print(f"Wrote final test results & hyperparameters to: {out_path}")

# # ── Comparison vs IY031 (identical SVM readout, identical split) ──────────────
# iy031 = pd.read_csv(IY031_DIR / "IY031_tf_condition_full_simclr_results.csv")
# iy031_best = iy031[iy031.status == "ok"].accuracy.max()
# comparison = pd.DataFrame([
#     {"method": "Chance", "accuracy": chance},
#     {"method": "Catch22 + SVM (IY031)", "accuracy": 0.5426},
#     {"method": "Raw SVM (IY031)", "accuracy": 0.7553},
#     {"method": "Best SimCLR + SVM, self-supervised (IY031)", "accuracy": iy031_best},
#     {"method": "IY036 SupCon + SVM, label-supervised", "accuracy": final_test},
# ]).sort_values("accuracy", ascending=False).reset_index(drop=True)
# print("\n=== Comparison vs IY031 (Full, 6-class, same SVM readout) ===")
# print(comparison.to_string(index=False))
# comparison.to_csv(IY036_DIR / "IY036_supcon_allclass_vs_iy031.csv", index=False)

# # KNN comparison kept in a separate CSV (SVM/KNN aren't apples-to-apples in one table)
# knn_comparison = pd.DataFrame([
#     {"method": "Chance", "accuracy": chance},
#     {"method": "Raw KNN (IY032)", "accuracy": 0.7234},
#     {"method": "IY036 SupCon + KNN, label-supervised", "accuracy": final_knn},
# ]).sort_values("accuracy", ascending=False).reset_index(drop=True)
# print("\n=== KNN comparison vs IY032 (Full, 6-class) ===")
# print(knn_comparison.to_string(index=False))
# knn_comparison.to_csv(IY036_DIR / "IY036_supcon_allclass_knn_vs_iy032.csv", index=False)

pd.DataFrame({k: pd.Series(v) for k, v in history.items()}).to_csv(
    IY036_DIR / f"IY036_supcon_allclass_history_{timestamp}.csv", index=False)