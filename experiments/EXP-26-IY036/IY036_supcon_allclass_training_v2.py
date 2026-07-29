"""
IY036 (1/2) v2, local interactive variant -- run directly on a local GPU, with
no job script. Identical data preparation, splits, metrics and outputs to
IY036_supcon_allclass_training_eddie_v2.py; the differences are the small-batch
regime it was tuned for (batch 512, AdamW at lr 1e-3, 200 epochs) versus the
Eddie variant's large-batch LARS setup. The v1 scripts are left untouched as
the record of the runs already reported.

Supervised-Contrastive (SupCon) pretraining on ALL experimental
TF@condition classes, evaluated with the IY031 SVM readout.

WHAT CHANGED IN v2 (and why)
----------------------------
v1 balanced the 6 benchmark classes to the minority count (78/class), cutting
1,099 cells to 468 and leaving 299 fit / 75 val / 94 test. Two consequences,
both measured on run 20260728_142748:

  * Checkpoint selection was dominated by winner's curse. `knn_val_acc` was
    scored 331 times on 75 cells; the selected maximum (0.933) sat 2.4 sd above
    the late-training mean (0.874, sd 0.024) and tested at 0.766. The metric
    selected on fell 17 pp, while the passenger SVM metric fell only 7 pp --
    the signature of selection noise rather than a val/test distribution shift.
  * A 94-cell test set carries a binomial SE of 4.1 pp, so the headline number
    could not resolve the ~2 pp differences the comparison table was reporting.

v2 therefore:
  1. Keeps all 1,099 benchmark cells and reports BALANCED accuracy (equal to
     accuracy on a balanced set in expectation, so the metric stays comparable
     to IY031) with `class_weight="balanced"` on the SVMs.
  2. Splits 60/20/20, giving ~659 fit / ~220 val / ~220 test. Val selection
     noise drops ~30%; test SE falls from 4.1 pp to ~2.7 pp.
  3. Derives the pretraining pool FROM the train split rather than subtracting
     test cells afterwards (see LEAKAGE CONTROL below).

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
Structural in v2, rather than post hoc. `prepare_split_dataset` performs ONE
pass over the data and returns the benchmark 3-way split plus every non-benchmark
cell; the pretraining pool is then built from the benchmark TRAIN split and the
non-benchmark cells. A val or test cell therefore cannot reach pretraining --
it is in a different index set. v1 instead built the pool from everything and
subtracted the test cells afterwards by hashing raw traces (`tobytes()`), which
worked but silently depended on float-exact trace equality.

The per-timepoint StandardScaler is still fit on the benchmark TRAIN split only
and is reused (via the returned `scaler`) for the pretraining inputs, so the
encoder sees one consistent input distribution and no val/test statistics leak.

DOWNSTREAM READOUT (IY031-comparable)
-------------------------------------
Encode -> StandardScale embeddings -> RBF-SVM (C=1.0, gamma='scale', seed 42),
i.e. exactly IY031's `run_simclr_svm`, plus `class_weight="balanced"` since the
splits are no longer balanced by subsampling. Note this keeps the embedding
StandardScaler, unlike the KNN readouts (IY032/IY035) where scaling was
dropped -- for an RBF-SVM feature scaling is genuinely required, and matching
IY031 is the point of this script.

Because the split differs from v1, the IY031/IY032 reference numbers are no
longer directly comparable, so the classical baselines (raw SVM, raw KNN,
catch22 + SVM) are RE-RUN here on the identical v2 split. The legacy 468-cell
figures are kept in the comparison table as clearly-labelled secondary rows.
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
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pytorch_metric_learning.losses import SupConLoss

from models.ssl_transformer import SSL_Transformer
from training.train import train_supcon_model

from features.catch22 import run_catch22_series_svm
from utils.embeddings import encode_channel, fit_predict_knn, knn_downstream_accuracy
from utils.experiment_tracking import run_timestamp
from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.processing.pipeline import prepare_split_dataset
from utils.augmentation import jitter_torch

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
NOISE_STD = 0.05       # Gaussian-noise augmentation for the two SupCon views

# training (small-batch local regime -- the Eddie v2 variant uses 4096 + LARS)
batch_size = 512   # Khosla et al. (SupCon) used 6144 on resnet-50, 4096 on resnet-200
epochs = 200
lr = 1e-3
weight_decay = 1e-4
temperature = 0.1     # SupCon default (Khosla et al.), though 0.07 is a well-established alternative
eval_every = 10        # epochs between checkpoint-selection evaluations
patience = epochs // (eval_every * 3)  # early stopping patience in units of eval_every
eval_metric_key = "knn_val_acc" # the metric used to select the best checkpoint -- KNN has no kernel to
# compensate for messy embedding geometry, so it's a more direct probe of whether SupCon is actually
# clustering same-label cells together than SVM is; SVM (matching IY031) is reported, not selected on.
# Selecting on a genuine held-out VAL split (not train-cv, not train accuracy) matters because
# training accuracy can't detect the downstream classifier overfitting to the training pool.
# Split fractions for the 6-class benchmark: 60 / 20 / 20 of all 1,099 cells.
# v1 balanced to 78/class first, so the same nominal fractions yielded only
# 299 / 75 / 94 cells; without balancing these give ~659 / ~220 / ~220.
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
# The 26 non-benchmark classes are never evaluated, so they need no test split:
# they are split train/val only, purely to monitor SupCon val loss.
PRETRAIN_VAL_FRACTION = 0.20
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

# ── ONE split, two pools ──────────────────────────────────────────────────────
# `d` holds the 6-class benchmark split 60/20/20 (unbalanced -- see docstring),
# and `X_rest_raw`/`y_rest_str` hold every cell from the other 26 classes at the
# same trace length. Both come from a single imputation pass, so the two pools
# are preprocessed identically AND are disjoint by construction: nothing in
# d["X_val"]/d["X_test"] can appear in the pretraining pool.
d = prepare_split_dataset(
    full_ts_raw, full_label_strs, FIXED_CLASSES, "Full",
    random_state=RANDOM_STATE, val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION, balance=False, seq_len=SEQ_LEN, return_rest=True,
)
n_cls_eval = len(d["class_names"])
chance = 1.0 / n_cls_eval

# Fit on the benchmark TRAIN split only, then reused for every other array --
# returned by prepare_split_dataset, so it cannot drift from the one that
# produced d["X_train"]/d["X_val"]/d["X_test"].
scaler_in = d["scaler"]

# The downstream classifier fits on the benchmark train split and is scored on
# the benchmark val split; d["X_test"] stays untouched until the one-shot final
# evaluation. In v1 this val split was carved by hand from the train split here.
X_down_tr, y_down_tr = d["X_train"], d["y_train"]
X_down_val, y_down_val = d["X_val"], d["y_val"]
print(f"Downstream 6-class: fit {len(y_down_tr)}, val {len(y_down_val)}, "
      f"test {len(d['y_test'])}, chance {chance:.4f}")

# ── SupCon pretraining pool: benchmark TRAIN cells + all non-benchmark cells ──
# No exclusion step is needed: val/test cells simply were never added.
X_pre_raw = np.vstack([d["X_train_raw"], d["X_rest_raw"]])
y_pre_str = np.concatenate([
    np.array(d["class_names"], dtype=object)[d["y_train"]],   # ints -> class names
    d["y_rest_str"],
])
pre_class_names = sorted(set(y_pre_str))
# Map string labels to integer indices for SupCon pretraining.
pre_l2i = {c: i for i, c in enumerate(pre_class_names)}
y_pre = np.array([pre_l2i[l] for l in y_pre_str])
X_pre = scaler_in.transform(X_pre_raw).astype(np.float32)   # same normalisation as eval

print(f"SupCon pretraining pool : {X_pre.shape[0]} cells x {X_pre.shape[1]} tp, "
      f"{len(pre_class_names)} TF@condition classes "
      f"({len(d['y_val']) + len(d['y_test'])} benchmark val/test cells never added)")

class_counts = pd.Series(y_pre).value_counts()
print(f"Pretraining class sizes: min={class_counts.min()}, max={class_counts.max()}, "
      f"n_classes={len(class_counts)}")

# Stratified val carve, held out purely to monitor SupCon val loss (not used for
# downstream eval or checkpoint selection -- eval_fn's SVM/KNN readout drives that).
try:
    X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(
        X_pre, y_pre, test_size=PRETRAIN_VAL_FRACTION,
        random_state=RANDOM_STATE, stratify=y_pre)
except ValueError:
    # some class too small to stratify at this fraction -- keep those classes entirely
    # in train and only carve the val split from classes with enough members.
    min_needed = max(2, int(np.ceil(1 / PRETRAIN_VAL_FRACTION)))
    splittable = class_counts[class_counts >= min_needed].index
    split_mask = np.isin(y_pre, splittable)
    X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(
        X_pre[split_mask], y_pre[split_mask], test_size=PRETRAIN_VAL_FRACTION,
        random_state=RANDOM_STATE, stratify=y_pre[split_mask])
    X_pre_tr = np.vstack([X_pre_tr, X_pre[~split_mask]])
    y_pre_tr = np.concatenate([y_pre_tr, y_pre[~split_mask]])
    print(f"  ({(~split_mask).sum()} cells from small classes kept entirely in train)")

print(f"SupCon train/val split: {len(y_pre_tr)} train, {len(y_pre_val)} val "
      f"(val_fraction={PRETRAIN_VAL_FRACTION})")

# Belt-and-braces check of the structural guarantee above: a raw benchmark
# val/test trace must not be present anywhere in the pretraining pool.
_pre_hashes = {row.tobytes() for row in X_pre_raw}
for _name in ("val", "test"):
    _overlap = sum(row.tobytes() in _pre_hashes for row in d[f"X_{_name}_raw"])
    assert _overlap == 0, f"{_overlap} benchmark {_name} cells leaked into pretraining"


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
# AdamW: at this batch size LARS buys nothing (it exists for the large-batch
# regime the Eddie variant runs in), so the pragmatic small-batch choice is kept.
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    Scored with BALANCED accuracy: v2 no longer subsamples the classes to the
    minority count, so plain accuracy would over-weight the larger classes
    (Rtg1 @ 0.01% has 275 cells against Rtg1 @ 2%'s 78). Balanced accuracy on
    this unbalanced split equals plain accuracy on a balanced one in
    expectation, which is what keeps the number comparable to IY031.
    """
    model.eval()
    Z_tr = encode_channel(model, d["X_train"], DEVICE)
    Z_te = encode_channel(model, d["X_test"], DEVICE)
    sc = StandardScaler()
    Z_tr_sc, Z_te_sc = sc.fit_transform(Z_tr), sc.transform(Z_te)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE,
              class_weight="balanced")
    clf.fit(Z_tr_sc, d["y_train"])
    y_pred = clf.predict(Z_te_sc)
    model.train()
    return balanced_accuracy_score(d["y_test"], y_pred), y_pred


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

    All four are BALANCED accuracies, for the reason given in `svm_eval`.
    """
    model.eval()
    Z_tr = encode_channel(model, X_down_tr, DEVICE)
    Z_val = encode_channel(model, X_down_val, DEVICE)
    model.train()

    # KNN: raw embeddings, no StandardScaler (established convention -- an
    # ablation over IY035 checkpoints found embedding scaling a wash that
    # homogenises checkpoints, suppressing the best ones).
    # PRIOR-CORRECTED: KNN has no class_weight, so on the unbalanced v2 fit set
    # majority classes would win neighbourhoods by sheer numbers. Measured on
    # raw traces this costs 5.2 pp of balanced accuracy if left uncorrected --
    # and since knn_val_acc selects the checkpoint, that bias would steer
    # training, not just the reported number.
    knn_kw = dict(n_neighbors=K_NEIGHBORS, metric="euclidean", prior_correction=True)
    knn_train_acc = balanced_accuracy_score(
        y_down_tr, fit_predict_knn(Z_tr, y_down_tr, Z_tr, **knn_kw))
    knn_val_acc = balanced_accuracy_score(
        y_down_val, fit_predict_knn(Z_tr, y_down_tr, Z_val, **knn_kw))

    # SVM: StandardScaled embeddings (matches IY031's methodology)
    sc = StandardScaler().fit(Z_tr)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE,
              class_weight="balanced")
    clf.fit(sc.transform(Z_tr), y_down_tr)
    svm_train_acc = balanced_accuracy_score(y_down_tr, clf.predict(sc.transform(Z_tr)))
    svm_val_acc = balanced_accuracy_score(y_down_val, clf.predict(sc.transform(Z_val)))

    return {"knn_val_acc": knn_val_acc, "svm_val_acc": svm_val_acc,
            "knn_train_acc": knn_train_acc, "svm_train_acc": svm_train_acc}


# ── Train ─────────────────────────────────────────────────────────────────────
# Reuse the job script's RUN_TIMESTAMP so the .out log and these artifacts match
timestamp = run_timestamp()
save_path = IY036_DIR / (f"IY036_v2_supcon_allclass_b{batch_size}_lr{lr}_"
                         f"L{num_layers}_H{nhead}_D{d_model}_{timestamp}_model.pth")

wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY036-SupCon-model",
    "name": f"supcon_allclass_v2_b{batch_size}_lr{lr}_d{dropout}_L{num_layers}_H{nhead}_D{d_model}_temp{temperature}_{timestamp}",
    "dataset": str(FULL_DATA_DIR),
    "augmentation": "noise", "noise_std": NOISE_STD,
    "num_cells_pretrain_train": len(y_pre_tr), "num_cells_pretrain_val": len(y_pre_val),
    "num_cells_downstream_train_fit": len(y_down_tr), "num_cells_downstream_val": len(y_down_val),
    "num_cells_downstream_test": len(d["y_test"]),
    "val_fraction": VAL_FRACTION, "test_fraction": TEST_FRACTION,
    "pretrain_val_fraction": PRETRAIN_VAL_FRACTION,
    "num_classes_pretrain": len(pre_class_names),
    "split_scheme": "v2: unbalanced 60/20/20, pools derived from the split",
    "downstream_metric": "balanced_accuracy",
    "batch_size": batch_size, "input_size": 1, "d_model": d_model, "nhead": nhead,
    "num_layers": num_layers, "dropout": dropout, "use_conv1d": False,
    "epochs": epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay,
    "optimizer": type(optimizer).__name__,
    "scheduler": type(scheduler).__name__,
    "loss_fn": "SupConLoss", "temperature": temperature,
    "eval_every": eval_every, "eval_metric_key": eval_metric_key, "knn_neighbors": K_NEIGHBORS,
    "seq_len": SEQ_LEN, "save_path": str(save_path), "grad_clip": None,
}

print("\nStarting SupCon training...")
history = train_supcon_model(
    model, train_loader, val_loader=val_loader,
    epochs=epochs, patience=patience, lr=lr, optimizer=optimizer, scheduler=scheduler,
    loss_fn=supcon_criterion, augment_fn=lambda x: jitter_torch(x, sigma=NOISE_STD),
    device=DEVICE, grad_clip=None, save_path=str(save_path),
    eval_fn=eval_fn, eval_every=eval_every, eval_metric_key=eval_metric_key,
    wandb_logging=True, wandb_config=wandb_config, verbose=True,
)
print(f"Saved: {save_path}")

# ── Final evaluation with the selected checkpoint ─────────────────────────────
model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))

# Binomial SE of a test-set proportion, so every headline number below is
# reported with the resolution it actually has. At n=220 this is ~2.7 pp, i.e.
# differences smaller than ~5 pp between any two rows are not resolvable.
n_test = len(d["y_test"])
def _se(acc):
    return float(np.sqrt(max(acc * (1 - acc), 0.0) / n_test))

final_svm_test, y_pred = svm_eval(model)
print(f"\n=== IY036 v2 SupCon + SVM (Full, 6-class) ===")
print(f"Balanced test accuracy: {final_svm_test:.4f} ± {_se(final_svm_test):.4f} (SE, n={n_test})"
      f"  (chance {chance:.4f}, {final_svm_test - chance:+.4f})")
print(classification_report(d["y_test"], y_pred, target_names=d["class_names"]))

model.eval()
# The helper returns plain accuracy; recompute from its predictions so KNN is
# scored on the same balanced metric as the SVM readout.
_, y_pred_knn = knn_downstream_accuracy(
    model, d["X_train"], d["X_test"], d["y_train"], d["y_test"], DEVICE,
    n_neighbors=K_NEIGHBORS, prior_correction=True)
model.train()
final_knn = balanced_accuracy_score(d["y_test"], y_pred_knn)
print(f"\n=== IY036 v2 SupCon + KNN (k={K_NEIGHBORS}, Full, 6-class) ===")
print(f"Balanced test accuracy: {final_knn:.4f} ± {_se(final_knn):.4f} (SE, n={n_test})"
      f"  (chance {chance:.4f}, {final_knn - chance:+.4f})")
print(classification_report(d["y_test"], y_pred_knn, target_names=d["class_names"]))

# ── Classical baselines, re-run on the IDENTICAL v2 split ─────────────────────
# The IY031/IY032 reference numbers were measured on the old balanced 468-cell
# split, so they cannot be compared row-by-row with anything above. These three
# re-runs use exactly the arrays this script trained and tested on.
print("\n=== Re-running classical baselines on the v2 split ===")

_raw_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE,
               class_weight="balanced").fit(d["X_train"], d["y_train"])
raw_svm_acc = balanced_accuracy_score(d["y_test"], _raw_svm.predict(d["X_test"]))
print(f"Raw SVM      : {raw_svm_acc:.4f} ± {_se(raw_svm_acc):.4f}")

raw_knn_acc = balanced_accuracy_score(d["y_test"], fit_predict_knn(
    d["X_train"], d["y_train"], d["X_test"], n_neighbors=K_NEIGHBORS,
    prior_correction=True))
print(f"Raw KNN (k={K_NEIGHBORS}): {raw_knn_acc:.4f} ± {_se(raw_knn_acc):.4f}")

# run_catch22_series_svm reads X_train_raw / X_test_raw / y_* / class_names,
# all of which prepare_split_dataset returns under the same names.
_, y_pred_c22 = run_catch22_series_svm(d, "IY036 v2 split", random_state=RANDOM_STATE,
                                       report=False)
catch22_acc = balanced_accuracy_score(d["y_test"], y_pred_c22)
print(f"Catch22 + SVM: {catch22_acc:.4f} ± {_se(catch22_acc):.4f}")

# Log and write the final test results along with the hyperparameters
final_results = [{
    "method": "IY036 SupCon, label-supervised",
    # run identity -- matches this run's .out log, checkpoint and history CSV
    "timestamp": timestamp,
    # test results -- BALANCED accuracy on the unbalanced v2 test split
    "metric": "balanced_accuracy",
    "svm_accuracy": float(final_svm_test),
    "knn_accuracy": float(final_knn),
    "chance": float(chance),          # 1/n_classes, so accuracies stay interpretable
    # binomial SE on the test split, so a reader can see what is resolvable
    "n_test": int(n_test),
    "svm_accuracy_se": _se(final_svm_test),
    "knn_accuracy_se": _se(final_knn),
    # classical baselines re-run on this exact split (not the IY031/IY032 values)
    "raw_svm_accuracy": float(raw_svm_acc),
    "raw_knn_accuracy": float(raw_knn_acc),
    "catch22_svm_accuracy": float(catch22_acc),
    # knn
    "k_neighbors": int(K_NEIGHBORS),
    # fixed classes (used for training)
    "fixed_classes": str(FIXED_CLASSES),
    # split scheme
    "split_scheme": "v2_unbalanced_60_20_20",
    "n_downstream_fit": int(len(y_down_tr)),
    "n_downstream_val": int(len(y_down_val)),
    # training hyperparameters
    "batch_size": int(batch_size),
    "epochs": int(epochs),
    "lr": float(lr),
    "val_fraction": float(VAL_FRACTION),
    "test_fraction": float(TEST_FRACTION),
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
out_path = IY036_DIR / f"IY036_v2_supcon_allclass_final_results_{timestamp}.csv"
final_df.to_csv(out_path, index=False)
print(f"Wrote final test results & hyperparameters to: {out_path}")

# ── Comparison table ──────────────────────────────────────────────────────────
# Rows are split into two blocks. The v2 block is measured on this run's exact
# test split and IS internally comparable. The legacy block is carried over from
# the old balanced 468-cell split (n=94) purely for continuity with the existing
# IY031/IY032 ELN entries -- it must NOT be read as a like-for-like ranking
# against the v2 rows, which is why `split` and `n_test` are columns here.
LEGACY_N_TEST = 94
comparison = pd.DataFrame([
    {"method": "Chance", "split": "-", "n_test": None,
     "accuracy": chance, "accuracy_se": None},
    {"method": "Catch22 + SVM", "split": "v2", "n_test": n_test,
     "accuracy": catch22_acc, "accuracy_se": _se(catch22_acc)},
    {"method": "Raw SVM", "split": "v2", "n_test": n_test,
     "accuracy": raw_svm_acc, "accuracy_se": _se(raw_svm_acc)},
    {"method": f"Raw KNN (k={K_NEIGHBORS})", "split": "v2", "n_test": n_test,
     "accuracy": raw_knn_acc, "accuracy_se": _se(raw_knn_acc)},
    {"method": "IY036 v2 SupCon + SVM, label-supervised", "split": "v2", "n_test": n_test,
     "accuracy": final_svm_test, "accuracy_se": _se(final_svm_test)},
    {"method": f"IY036 v2 SupCon + KNN (k={K_NEIGHBORS}), label-supervised",
     "split": "v2", "n_test": n_test,
     "accuracy": final_knn, "accuracy_se": _se(final_knn)},
    # legacy rows -- different test set, NOT comparable to the block above
    {"method": "Catch22 + SVM (IY031, legacy split)", "split": "legacy",
     "n_test": LEGACY_N_TEST, "accuracy": 0.5426, "accuracy_se": None},
    {"method": "Raw SVM (IY031, legacy split)", "split": "legacy",
     "n_test": LEGACY_N_TEST, "accuracy": 0.7553, "accuracy_se": None},
    {"method": "Raw KNN (IY032, legacy split)", "split": "legacy",
     "n_test": LEGACY_N_TEST, "accuracy": 0.7234, "accuracy_se": None},
    {"method": "Best SimCLR + SVM, self-supervised (IY031, legacy split)",
     "split": "legacy", "n_test": LEGACY_N_TEST, "accuracy": 0.7979, "accuracy_se": None},
]).sort_values(["split", "accuracy"], ascending=[True, False]).reset_index(drop=True)

print("\n=== IY036 v2 comparison (Full, 6-class, balanced accuracy) ===")
print(comparison.to_string(index=False))
print(f"\nNOTE: 'legacy' rows come from the old balanced 468-cell split (n=94) and are\n"
      f"NOT comparable to the v2 rows. Within the v2 block, the binomial SE is\n"
      f"~{_se(final_svm_test):.3f}, so differences below ~{1.96 * np.sqrt(2) * _se(final_svm_test):.2f} (1.96 x the SE of a difference) are not\nresolvable by a single run.")
comparison.to_csv(
    IY036_DIR / f"IY036_v2_supcon_allclass_comparison_{timestamp}.csv", index=False)

# write training history to csv
pd.DataFrame({k: pd.Series(v) for k, v in history.items()}).to_csv(
    IY036_DIR / f"IY036_v2_supcon_allclass_history_{timestamp}.csv", index=False)
