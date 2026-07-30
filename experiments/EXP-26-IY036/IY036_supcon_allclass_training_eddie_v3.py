"""
IY036 (1/2) v3, Eddie HPC submission variant. ONE script, TWO arms -- launched
via `qsub IY036_supcon_allclass_training_eddie_v3_composed.sh` and
`..._v3_jitter.sh`, which differ only in the AUG_RECIPE they export. The v1 and
v2 scripts are left untouched as the record of the runs already reported.

Supervised-Contrastive (SupCon) pretraining on ALL experimental
TF@condition classes, evaluated with the IY031 SVM readout.

WHAT CHANGED IN v3 (and why)
----------------------------
The v2 Eddie run (20260729_191805) was honest but null: SupCon + SVM scored
0.7655 +/- 0.0286 balanced accuracy against 0.7467 +/- 0.0293 for a plain RBF-SVM
on raw standardised traces, over the same 220 test cells. Tested properly:
McNemar on overall accuracy p = 1.0 (15 cells SupCon-only-correct vs 14
raw-only-correct), paired bootstrap on the balanced-accuracy difference
[-6.2, +9.8] pp. So 3360 epochs of pretraining was indistinguishable from doing
nothing at all. Two causes are addressed here.

1. THE EXPERIMENT COULD NOT DETECT AN IMPROVEMENT.
   One seed; ~8 pp resolution; and the checkpoint was picked by a noise spike --
   knn_val_acc peaked at 0.8464 at epoch 250 against a flat plateau of
   0.7969 +/- 0.0188 (2.6 sd above it), and because nothing ever beat that spike,
   early stopping fired at epoch 1370 of 3360. Model selection was partly RNG and
   59% of the compute budget was discarded for a statistical artifact. v3 runs
   SEEDS (mean +/- sd), selects on a trailing mean (SMOOTH_WINDOW), and reports a
   PAIRED bootstrap + McNemar against the raw-feature baseline every seed.

2. THE AUGMENTATION SPECIFIED ALMOST NO INVARIANCE.
   v2's augment_fn was a single jitter_torch(sigma=0.05) -- 5% of a standard
   deviation on standardised data, so the two contrastive "views" were near
   identical copies and the encoder was under no pressure to discard amplitude,
   drift or timing variation. v3 composes four operations chosen to mimic
   imaging-session artifacts (see `make_augment_fn`).

Because AUG_RECIPE="jitter_only" reproduces v2's augmentation exactly while
everything else stays fixed, the two arms form a controlled A/B in which
augmentation is the only difference.

WHAT v2 ALREADY FIXED (carried over unchanged)
----------------------------------------------
v1 balanced the 6 benchmark classes to the minority count (78/class), cutting
1,099 cells to 468 and leaving 299 fit / 75 val / 94 test. v2 therefore:
  1. Keeps all 1,099 benchmark cells and reports BALANCED accuracy (equal to
     accuracy on a balanced set in expectation, so the metric stays comparable
     to IY031) with `class_weight="balanced"` on the SVMs, and a prior-corrected
     KNN (which has no class_weight of its own).
  2. Splits 60/20/20, giving 659 fit / 220 val / 220 test.
  3. Derives the pretraining pool FROM the train split rather than subtracting
     test cells afterwards (see LEAKAGE CONTROL below).

SCOPE NOTE
----------
v3 keeps the random 60/20/20 split. Because condition is 1:1 with imaging session
in this benchmark, a random split structurally REWARDS encoding session identity,
so an invariance gain from the new augmentation may not show up here even if it
is real. A leave-session-out design is the setting where invariance pays, and is
deliberately out of scope for v3.

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
effects. A leave-session-out design would test batch-invariance properly; see
the SCOPE NOTE above.

LEAKAGE CONTROL
---------------
Structural from v2 onwards, rather than post hoc. `prepare_split_dataset` performs ONE
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
catch22 + SVM) are RE-RUN here on the identical v3 split, once, since they
involve no training randomness and the split never moves.
"""

import os
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
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# Exact paired test of OVERALL accuracy between two classifiers on the same
# cells. Reported alongside the bootstrap, which covers balanced accuracy.
from statsmodels.stats.contingency_tables import mcnemar

from pytorch_metric_learning.losses import SupConLoss
# Linear-warmup + cosine-decay, the schedule paired with LARS in the SimCLR/
# SupCon lineage. Imported at the top in v3 (v2 imported it mid-file) because
# run_seed() builds a fresh scheduler per seed.
from transformers import get_cosine_schedule_with_warmup

from models.ssl_transformer import SSL_Transformer
from training.train import train_supcon_model
from training.wandb_utils import wandb_log
from open_lars import LARS
from features.catch22 import run_catch22_series_svm
from utils.embeddings import encode_channel, fit_predict_knn, knn_downstream_accuracy
from utils.experiment_tracking import run_timestamp
from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.metrics import paired_bootstrap_diff
from utils.processing.pipeline import prepare_split_dataset
from utils.augmentation import (jitter_torch, scaling_torch,
                                magnitude_warp_torch, window_slice_torch)

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

# ── v3: repeated seeds ────────────────────────────────────────────────────────
# TRAINING seeds only. RANDOM_STATE above stays 42 for every SPLIT and every
# classifier, so all seeds and both arms are scored on the IDENTICAL 220-cell
# test set -- an arm difference therefore cannot be split variance. What varies
# per seed is model init, batch order, and augmentation randomness.
SEEDS = [42, 43, 44]

# ── v3: smoothed checkpoint selection ─────────────────────────────────────────
# v2 selected on a single noisy eval: run 20260729_191805 peaked at knn_val_acc
# 0.8464 at epoch 250 against a flat plateau of 0.7969 +/- 0.0188 (2.6 sd above),
# and since nothing ever beat that spike, early stopping fired at epoch 1370 of
# 3360. Selecting on a trailing mean makes a lucky spike unable to win outright,
# which fixes both the checkpoint choice and the premature stop.
SMOOTH_WINDOW = 3
eval_metric_key = "knn_val_acc_smooth"   # overrides the v2 value set above

# ── v3: augmentation recipe (the A/B) ─────────────────────────────────────────
# Set by the job script, so both arms run from ONE .py with nothing else changed.
#   "jitter_only" -- byte-identical to v2; the control arm.
#   "composed"    -- the four-op pipeline below.
AUG_RECIPE = os.environ.get("AUG_RECIPE", "composed")
if AUG_RECIPE not in {"jitter_only", "composed"}:
    raise ValueError(f"AUG_RECIPE must be 'jitter_only' or 'composed', got {AUG_RECIPE!r}")

JITTER_SIGMA_V2 = 0.05      # v2's value, used by the jitter_only control arm
JITTER_SIGMA = 0.15         # raised for the composed arm
SCALING_SIGMA = 0.10        # per-trace multiplicative gain
MAGWARP_SIGMA, MAGWARP_KNOT = 0.20, 4   # smooth multiplicative drift / bleaching
WINSLICE_RATIO = 0.90       # crop 90% and stretch back -> timing / phase offset

# model (identical architecture to the IY031/IY032 checkpoints for comparability)
d_model, nhead, num_layers, dropout = 16, 4, 2, 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds for the SPLITS and data prep below. The per-seed training reseed happens
# inside run_seed(); this one must stay fixed so every arm sees the same splits.
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

# ── ONE split, two pools (identical for every seed and both arms) ──────────────────────────────────────────────────────
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


# NOT batch_size: an eval-mode (model.eval() + torch.no_grad()) forward pass
# measured ~2.3-2.5x MORE peak memory per sample than a full train step
# (forward+backward+optimizer.step()) at the same batch size on this
# architecture -- profiled empirically, likely an SDPA backend difference
# between grad/no-grad mode. Reusing the (large) training batch_size here
# would make the val loop the actual OOM bottleneck, not the training step.
VAL_BATCH_SIZE = 256


# ── Augmentation recipes (the v3 A/B) ─────────────────────────────────────────
def make_augment_fn(recipe: str):
    """Return the stochastic view-generator `T` for this arm.

    ONE call returns ONE view. `train_supcon_model` keeps n_views=2 and calls
    this twice per batch, so each batch yields two views, each having passed
    through the whole chain under its own independent random draws -- the
    SimCLR/SupCon convention of sampling twice from a pipeline. This teaches
    invariance to the operations in COMBINATION, and leaves n_views, batch size
    and memory footprint identical to v2 so the A/B stays clean.

    Ordering: the amplitude operations run before the time-axis one, so the crop
    resamples an already-perturbed trace (one interpolation, not two).

    CAVEAT, stated plainly: inputs are per-timepoint standardised before they get
    here, so `scaling`/`magnitude_warp` scale deviations from each timepoint's
    mean rather than literal illumination gain, and `window_slice` shifts traces
    relative to the per-timepoint scaler statistics. These are proxies for
    session artifacts, not exact models of them.
    """
    if recipe == "jitter_only":
        # v2's augmentation exactly -- the control arm
        return lambda x: jitter_torch(x, sigma=JITTER_SIGMA_V2)

    def _composed(x):
        x = jitter_torch(x, sigma=JITTER_SIGMA)              # sensor / shot noise
        x = scaling_torch(x, sigma=SCALING_SIGMA)            # per-trace gain
        x = magnitude_warp_torch(x, sigma=MAGWARP_SIGMA,     # drift / bleaching
                                 knot=MAGWARP_KNOT)
        x = window_slice_torch(x, reduce_ratio=WINSLICE_RATIO)  # timing / phase
        return x

    return _composed


augment_fn = make_augment_fn(AUG_RECIPE)
print(f"\nAugmentation recipe: {AUG_RECIPE}")
if AUG_RECIPE == "composed":
    print(f"  jitter σ={JITTER_SIGMA} -> scaling σ={SCALING_SIGMA} -> "
          f"magnitude_warp σ={MAGWARP_SIGMA} (knot {MAGWARP_KNOT}) -> "
          f"window_slice ratio={WINSLICE_RATIO}")
else:
    print(f"  jitter σ={JITTER_SIGMA_V2} (v2 control arm)")


# ── Evaluation ────────────────────────────────────────────────────────────────
def svm_eval(model):
    """IY031's readout: encode -> StandardScale embeddings -> RBF-SVM.

    Fit on the FULL downstream train split (d["X_train"]), evaluated on the
    held-out test split (d["X_test"]) -- computed once per seed, after training,
    never tracked per-epoch or used for checkpoint selection.

    Scored with BALANCED accuracy: v2 onwards no longer subsamples the classes to
    the minority count, so plain accuracy would over-weight the larger classes
    (Rtg1 @ 0.01% has 275 cells against Rtg1 @ 2%'s 78). Balanced accuracy on
    this unbalanced split equals plain accuracy on a balanced one in expectation,
    which is what keeps the number comparable to IY031.
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


# Trailing window of raw knn_val_acc values, feeding the smoothed selection
# metric. STATEFUL -- run_seed() clears it, or values would leak between seeds.
_knn_val_window: deque = deque(maxlen=SMOOTH_WINDOW)


def eval_fn(model):
    """
    Periodic downstream train/val readout on the benchmark's val split (220
    cells) -- d["X_test"] is never touched here, and is reserved for the one-shot
    final evaluation per seed.

    `knn_val_acc_smooth` (the trailing mean of the last SMOOTH_WINDOW raw values)
    drives checkpoint selection. Selecting on the raw value, as v2 did, means
    selecting the maximum of ~140-330 noisy estimates, which reliably picks a
    lucky spike rather than a genuinely better encoder -- and, because that spike
    is then unbeatable, also trips early stopping long before the epoch budget is
    spent. The raw value is still returned so the spike/plateau behaviour stays
    visible in the history CSV and in wandb.

    Known caveat: the checkpoint saved at a smoothed peak holds the weights from
    the END of the window, not its centre.

    knn_train_acc / svm_train_acc are tracked PURELY to visualise the train/val
    gap (a widening gap = overfitting); training accuracy can never select or
    stop, since it is evaluated on the data the classifier was fit on.

    All accuracies are BALANCED, for the reason given in `svm_eval`.
    """
    model.eval()
    Z_tr = encode_channel(model, X_down_tr, DEVICE)
    Z_val = encode_channel(model, X_down_val, DEVICE)
    model.train()

    # KNN: raw embeddings, no StandardScaler (established convention -- an
    # ablation over IY035 checkpoints found embedding scaling a wash that
    # homogenises checkpoints, suppressing the best ones).
    # PRIOR-CORRECTED: KNN has no class_weight, so on the unbalanced fit set
    # majority classes would win neighbourhoods by sheer numbers. Measured on raw
    # traces this costs 5.2 pp of balanced accuracy if left uncorrected -- and
    # since this metric selects the checkpoint, that bias would steer training,
    # not merely the reported number.
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

    _knn_val_window.append(knn_val_acc)
    return {"knn_val_acc_smooth": float(np.mean(_knn_val_window)),
            "knn_val_acc": knn_val_acc, "svm_val_acc": svm_val_acc,
            "knn_train_acc": knn_train_acc, "svm_train_acc": svm_train_acc}


# Binomial SE of a test-set proportion, so every headline number is reported with
# the resolution it actually has. At n=220 this is ~2.9 pp.
n_test = len(d["y_test"])


def _se(acc):
    return float(np.sqrt(max(acc * (1 - acc), 0.0) / n_test))


# ── Classical baselines: fixed across seeds, so computed ONCE ─────────────────
# They involve no training randomness, and the split never moves, so re-running
# them per seed would produce identical numbers. The IY031/IY032 reference values
# were measured on the old balanced 468-cell split and are NOT row-comparable to
# these, which is why they are recomputed here on the exact v3 arrays.
print("\n=== Classical baselines on the v3 split (identical for every seed) ===")
_raw_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE,
               class_weight="balanced").fit(d["X_train"], d["y_train"])
y_pred_raw_svm = _raw_svm.predict(d["X_test"])
raw_svm_acc = balanced_accuracy_score(d["y_test"], y_pred_raw_svm)
print(f"Raw SVM      : {raw_svm_acc:.4f} ± {_se(raw_svm_acc):.4f}")

y_pred_raw_knn = fit_predict_knn(d["X_train"], d["y_train"], d["X_test"],
                                 n_neighbors=K_NEIGHBORS, prior_correction=True)
raw_knn_acc = balanced_accuracy_score(d["y_test"], y_pred_raw_knn)
print(f"Raw KNN (k={K_NEIGHBORS}): {raw_knn_acc:.4f} ± {_se(raw_knn_acc):.4f}")

_, y_pred_c22 = run_catch22_series_svm(d, "IY036 v3 split", random_state=RANDOM_STATE,
                                       report=False)
catch22_acc = balanced_accuracy_score(d["y_test"], y_pred_c22)
print(f"Catch22 + SVM: {catch22_acc:.4f} ± {_se(catch22_acc):.4f}")


# ── One training run ──────────────────────────────────────────────────────────
timestamp = run_timestamp()


def run_seed(seed: int) -> dict:
    """Train once under `seed` and return this run's row of results.

    Only model init, batch order and augmentation draws depend on `seed`; every
    split and every classifier stays pinned to RANDOM_STATE, so seed-to-seed
    spread measures training variance alone.
    """
    print(f"\n{'=' * 78}\n=== seed {seed}  |  recipe {AUG_RECIPE} \n{'=' * 78}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _knn_val_window.clear()      # or the previous seed's tail leaks into selection

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CellDataset(X_pre_tr, y_pre_tr), batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, generator=loader_gen,
    )
    val_loader = DataLoader(
        CellDataset(X_pre_val, y_pre_val), batch_size=VAL_BATCH_SIZE,
        shuffle=False, num_workers=4,
    )
    print(f"Batches/epoch: {len(train_loader)}  (batch_size={batch_size})")

    model = SSL_Transformer(input_size=1, d_model=d_model, nhead=nhead,
                            num_layers=num_layers, dropout=dropout,
                            use_conv1d=False).to(DEVICE)
    # LARS: this Eddie run uses a genuinely large batch (see `batch_size`), which
    # is exactly the regime LARS exists for. Uses the open_lars package (pip) --
    # verified bit-for-bit identical to this repo's prior hand-rolled
    # implementation before swapping it in; see requirements.yml.
    optimizer = LARS(model.parameters(), lr=lr, momentum=lars_momentum,
                     weight_decay=weight_decay, trust_coefficient=lars_trust_coefficient)
    supcon_criterion = SupConLoss(temperature=temperature)
    # Linear-warmup + cosine-decay is the standard schedule paired with LARS in
    # the SimCLR/SupCon lineage (Chen et al. 2020: 10 epochs linear warmup, then
    # cosine decay without restarts) -- LARS is unstable early at large batch/lr,
    # and warmup is the standard fix (Goyal et al. 2017).
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * epochs), num_training_steps=epochs)

    save_path = IY036_DIR / (f"IY036_v3_{AUG_RECIPE}_b{batch_size}_lr{lr}_"
                             f"L{num_layers}_H{nhead}_D{d_model}_"
                             f"{timestamp}_seed{seed}_model.pth")

    wandb_config = {
        "entity": "grignard-reagent",
        "project": "IY036-SupCon-model",
        "name": (f"supcon_allclass_eddie_v3_{AUG_RECIPE}_b{batch_size}_lr{lr}_"
                 f"L{num_layers}_H{nhead}_D{d_model}_temp{temperature}_"
                 f"{timestamp}_seed{seed}"),
        "dataset": str(FULL_DATA_DIR),
        # the A/B variable, plus every strength that defines it
        "aug_recipe": AUG_RECIPE, "seed": seed,
        "jitter_sigma": JITTER_SIGMA if AUG_RECIPE == "composed" else JITTER_SIGMA_V2,
        "scaling_sigma": SCALING_SIGMA if AUG_RECIPE == "composed" else None,
        "magwarp_sigma": MAGWARP_SIGMA if AUG_RECIPE == "composed" else None,
        "magwarp_knot": MAGWARP_KNOT if AUG_RECIPE == "composed" else None,
        "winslice_ratio": WINSLICE_RATIO if AUG_RECIPE == "composed" else None,
        "n_views": 2, "view_scheme": "chained pipeline, sampled twice",
        "num_cells_pretrain_train": len(y_pre_tr), "num_cells_pretrain_val": len(y_pre_val),
        "num_cells_downstream_train_fit": len(y_down_tr),
        "num_cells_downstream_val": len(y_down_val),
        "num_cells_downstream_test": len(d["y_test"]),
        "val_fraction": VAL_FRACTION, "test_fraction": TEST_FRACTION,
        "pretrain_val_fraction": PRETRAIN_VAL_FRACTION,
        "num_classes_pretrain": len(pre_class_names),
        "split_scheme": "v2/v3: unbalanced 60/20/20, pools derived from the split",
        "downstream_metric": "balanced_accuracy",
        "batch_size": batch_size, "input_size": 1, "d_model": d_model, "nhead": nhead,
        "num_layers": num_layers, "dropout": dropout, "use_conv1d": False,
        "epochs": epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay,
        "optimizer": type(optimizer).__name__, "lars_momentum": lars_momentum,
        "lars_trust_coefficient": lars_trust_coefficient,
        "scheduler": type(scheduler).__name__,
        "loss_fn": "SupConLoss", "temperature": temperature,
        "eval_every": eval_every, "eval_metric_key": eval_metric_key,
        "smooth_window": SMOOTH_WINDOW, "knn_neighbors": K_NEIGHBORS,
        "seq_len": SEQ_LEN, "save_path": str(save_path), "grad_clip": None,
    }

    # return_run=True keeps the wandb run OPEN so the one-shot final metrics
    # below -- computed after training, on the reloaded best checkpoint -- can
    # still be attached to it. `wandb_run` is None when wandb_logging is off.
    history, wandb_run = train_supcon_model(
        model, train_loader, val_loader=val_loader,
        epochs=epochs, patience=patience, lr=lr, optimizer=optimizer,
        scheduler=scheduler, loss_fn=supcon_criterion, augment_fn=augment_fn,
        device=DEVICE, grad_clip=None, save_path=str(save_path),
        eval_fn=eval_fn, eval_every=eval_every, eval_metric_key=eval_metric_key,
        wandb_logging=True, wandb_config=wandb_config, verbose=True,
        return_run=True,
    )
    print(f"Saved: {save_path}")

    # ── Final evaluation with the selected checkpoint ─────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    final_svm_test, y_pred_svm = svm_eval(model)
    print(f"\n=== seed {seed}: SupCon + SVM (Full, 6-class) ===")
    print(f"Balanced test accuracy: {final_svm_test:.4f} ± {_se(final_svm_test):.4f} "
          f"(SE, n={n_test})  (chance {chance:.4f}, {final_svm_test - chance:+.4f})")
    print(classification_report(d["y_test"], y_pred_svm, target_names=d["class_names"]))

    model.eval()
    # The helper returns plain accuracy; recompute from its predictions so KNN is
    # scored on the same balanced metric as the SVM readout.
    _, y_pred_knn = knn_downstream_accuracy(
        model, d["X_train"], d["X_test"], d["y_train"], d["y_test"], DEVICE,
        n_neighbors=K_NEIGHBORS, prior_correction=True)
    model.train()
    final_knn = balanced_accuracy_score(d["y_test"], y_pred_knn)
    print(f"\n=== seed {seed}: SupCon + KNN (k={K_NEIGHBORS}, Full, 6-class) ===")
    print(f"Balanced test accuracy: {final_knn:.4f} ± {_se(final_knn):.4f} "
          f"(SE, n={n_test})  (chance {chance:.4f}, {final_knn - chance:+.4f})")

    # ── Did pretraining actually beat raw features? ───────────────────────────
    # Both readouts score the SAME 220 cells, so this is a PAIRED comparison and
    # the shared test-set noise cancels -- far tighter than eyeballing two
    # independent error bars. McNemar is reported alongside because it is exact,
    # but it tests OVERALL accuracy, not the balanced accuracy we report.
    boot = paired_bootstrap_diff(d["y_test"], y_pred_svm, y_pred_raw_svm,
                                 random_state=RANDOM_STATE)
    a_ok, b_ok = (y_pred_svm == d["y_test"]), (y_pred_raw_svm == d["y_test"])
    n10, n01 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    mcn = mcnemar([[int((a_ok & b_ok).sum()), n10],
                   [n01, int((~a_ok & ~b_ok).sum())]], exact=True)
    print(f"\n--- seed {seed}: SupCon+SVM vs Raw SVM, paired on the same {n_test} cells ---")
    print(f"balanced-accuracy difference {boot['diff']:+.4f} "
          f"95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}], P(SupCon better)={boot['p_gt_0']:.2f}")
    print(f"McNemar (overall accuracy): SupCon-only-correct={n10}, "
          f"Raw-only-correct={n01}, p={mcn.pvalue:.4f}")

    # ── One-shot final metrics -> wandb ───────────────────────────────────────
    # Single values per run, so they belong in run.summary rather than run.log:
    # summary values are what the wandb runs TABLE sorts and filters on, which is
    # how the two arms get compared. Also logged as a final step so they appear
    # in the run's own charts.
    if wandb_run is not None:
        final_metrics = {
            "final/svm_balanced_acc": float(final_svm_test),
            "final/knn_balanced_acc": float(final_knn),
            "final/svm_balanced_acc_se": _se(final_svm_test),
            "final/knn_balanced_acc_se": _se(final_knn),
            "final/vs_raw_svm_diff": boot["diff"],
            "final/vs_raw_svm_ci_lo": boot["lo"],
            "final/vs_raw_svm_ci_hi": boot["hi"],
            "final/vs_raw_svm_mcnemar_p": float(mcn.pvalue),
            "final/baseline_raw_svm": float(raw_svm_acc),
            "final/baseline_raw_knn": float(raw_knn_acc),
            "final/baseline_catch22_svm": float(catch22_acc),
            "final/chance": float(chance),
            "final/n_test": int(n_test),
            "final/selected_epoch_of": int(max(history["epoch"])),
        }
        wandb_run.summary.update(final_metrics)
        wandb_log(wandb_run, final_metrics)
        wandb_run.finish()       # train_supcon_model deliberately left it open
        print("Logged final test metrics to wandb summary.")

    pd.DataFrame({k: pd.Series(v) for k, v in history.items()}).to_csv(
        IY036_DIR / f"IY036_v3_{AUG_RECIPE}_history_eddie_{timestamp}_seed{seed}.csv",
        index=False)

    return {
        "method": "IY036 v3 SupCon, label-supervised",
        "aug_recipe": AUG_RECIPE,
        "seed": int(seed),
        "timestamp": timestamp,
        "metric": "balanced_accuracy",
        # test results
        "svm_accuracy": float(final_svm_test),
        "knn_accuracy": float(final_knn),
        "svm_accuracy_se": _se(final_svm_test),
        "knn_accuracy_se": _se(final_knn),
        "chance": float(chance),
        "n_test": int(n_test),
        # did it beat raw features on the same cells?
        "vs_raw_svm_diff": boot["diff"],
        "vs_raw_svm_ci_lo": boot["lo"],
        "vs_raw_svm_ci_hi": boot["hi"],
        "vs_raw_svm_p_gt_0": boot["p_gt_0"],
        "vs_raw_svm_mcnemar_p": float(mcn.pvalue),
        # baselines on this exact split (constant across seeds)
        "raw_svm_accuracy": float(raw_svm_acc),
        "raw_knn_accuracy": float(raw_knn_acc),
        "catch22_svm_accuracy": float(catch22_acc),
        # training diagnostics
        "epochs_run": int(max(history["epoch"])),
        "epochs_budget": int(epochs),
        "best_knn_val_smooth": float(max(history["eval/knn_val_acc_smooth"])),
        "best_knn_val_raw": float(max(history["eval/knn_val_acc"])),
        # augmentation (the A/B variable)
        "jitter_sigma": float(JITTER_SIGMA if AUG_RECIPE == "composed" else JITTER_SIGMA_V2),
        "scaling_sigma": float(SCALING_SIGMA) if AUG_RECIPE == "composed" else None,
        "magwarp_sigma": float(MAGWARP_SIGMA) if AUG_RECIPE == "composed" else None,
        "winslice_ratio": float(WINSLICE_RATIO) if AUG_RECIPE == "composed" else None,
        # config
        "smooth_window": int(SMOOTH_WINDOW),
        "eval_metric_key": str(eval_metric_key),
        "k_neighbors": int(K_NEIGHBORS),
        "fixed_classes": str(FIXED_CLASSES),
        "split_scheme": "v3_unbalanced_60_20_20",
        "n_downstream_fit": int(len(y_down_tr)),
        "n_downstream_val": int(len(y_down_val)),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "noise_std": float(NOISE_STD),
        "optimizer": type(optimizer).__name__,
        "lars_momentum": float(lars_momentum),
        "lars_trust_coefficient": float(lars_trust_coefficient),
        "weight_decay": float(weight_decay),
        "temperature": float(temperature),
        "patience": int(patience),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "seq_len": int(SEQ_LEN),
    }


# ── Run every seed, then aggregate ────────────────────────────────────────────
rows = [run_seed(seed) for seed in SEEDS]

results = pd.DataFrame(rows)
out_path = IY036_DIR / f"IY036_v3_{AUG_RECIPE}_final_results_eddie_{timestamp}.csv"
results.to_csv(out_path, index=False)
print(f"\nWrote {len(rows)} per-seed rows to: {out_path}")

print(f"\n{'=' * 78}")
print(f"=== IY036 v3 [{AUG_RECIPE}] across {len(SEEDS)} seeds: mean ± sd ===")
print(f"{'=' * 78}")
for col, label in [("svm_accuracy", "SupCon + SVM"),
                   ("knn_accuracy", f"SupCon + KNN (k={K_NEIGHBORS})"),
                   ("vs_raw_svm_diff", "  vs Raw SVM (paired diff)"),
                   ("epochs_run", "epochs actually run")]:
    v = results[col]
    print(f"{label:32s} {v.mean():8.4f} ± {v.std():.4f}   (seeds: "
          f"{', '.join(f'{x:.4f}' for x in v)})")
print(f"\n{'Raw SVM (constant)':32s} {raw_svm_acc:8.4f}")
print(f"{'Raw KNN (constant)':32s} {raw_knn_acc:8.4f}")
print(f"{'Catch22 + SVM (constant)':32s} {catch22_acc:8.4f}")
print(f"{'Chance':32s} {chance:8.4f}")
print(f"\nSeed-to-seed sd is TRAINING variance only -- the split and every "
      f"classifier are pinned to RANDOM_STATE={RANDOM_STATE}, so all seeds and "
      f"both arms score the identical {n_test}-cell test set.")
print(f"Run IY036_v3_compare_arms.py once both arms are done for the paired "
      f"composed-vs-jitter_only comparison.")
