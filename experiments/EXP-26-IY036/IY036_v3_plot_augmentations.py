"""Visualise the v3 A/B augmentations: traces before vs after.

Produces two figures for `IY036_supcon_allclass_training_eddie_v3.py`'s two arms:

  1. IY036_augmentation_before_after.png
     Original trace overlaid with the two SupCon views it generates, one row per
     recipe ("jitter_only" control vs "composed"), one column per example cell.
     Two views are drawn because `train_supcon_model` calls the augment_fn twice
     per batch -- the pair the contrastive loss actually has to pull together.

  2. IY036_composed_augmentation_steps.png
     The composed recipe unrolled: the same trace after each of its four ops in
     turn (jitter -> scaling -> magnitude_warp -> window_slice), so the marginal
     contribution of each operation is visible rather than only their sum.

Data are prepared through the SAME path as the training script (identical
loader, split, seed and per-timepoint scaler), so what is plotted is exactly
what the model sees -- augmentation acts on standardised traces, not raw AU.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.processing.pipeline import prepare_split_dataset
from utils.augmentation import (jitter_torch, scaling_torch,
                                magnitude_warp_torch, window_slice_torch)

# ── Config: mirrored verbatim from the v3 training script ─────────────────────
IY036_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = IY036_DIR.parent
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
SEQ_LEN = 540
VAL_FRACTION, TEST_FRACTION = 0.20, 0.20

# Augmentation hyperparameters -- must stay in step with the training script
JITTER_SIGMA_V2 = 0.05      # jitter_only (control) arm
JITTER_SIGMA = 0.15         # composed arm
SCALING_SIGMA = 0.10
MAGWARP_SIGMA, MAGWARP_KNOT = 0.20, 4
WINSLICE_RATIO = 0.90

# Plotting
MINUTES_PER_TP = 2.0        # imaging interval (traces resampled to SEQ_LEN tp)
N_EXAMPLES = 3              # cells shown, one per column
PLOT_SEED = 0               # fixed so the figure is reproducible

sns.set_theme(style="ticks", font="sans-serif")
PALETTE = sns.color_palette("colorblind")
COL_ORIG, COL_V1, COL_V2 = "black", PALETTE[0], PALETTE[1]
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


# ── Data: identical preparation to the training script ────────────────────────
metadata = pd.read_csv(META_PATH)
metadata["exp_id"] = metadata["exp_id"].astype(str)
metadata["group_id"] = metadata["group_id"].astype(str)
LABEL_LOOKUP = {(r.exp_id, r.group_id, r.channel): (r.tf, r.condition)
                for _, r in metadata.iterrows()}

full_ts_raw, full_label_strs = load_labelled_time_series_csvs(
    data_dir=FULL_DATA_DIR, file_re=FILE_RE, label_lookup=LABEL_LOOKUP,
    meta_cols=META_COLS, excluded_exps=EXCLUDED_EXPS, verbose=False,
)

d = prepare_split_dataset(
    full_ts_raw, full_label_strs, FIXED_CLASSES, "Full",
    random_state=RANDOM_STATE, val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION, balance=False, seq_len=SEQ_LEN, return_rest=True,
)

# The training pool the SupCon views are drawn from, already standardised by the
# scaler fitted on the benchmark train split (exactly as in the training script).
X_train = d["X_train"].astype(np.float32)
class_names = np.array(d["class_names"], dtype=object)
y_train = d["y_train"]

# Pick one example cell per class, from classes spread evenly across the label
# set (so both TFs and both glucose extremes appear) rather than N near-
# duplicates from one condition.
rng = np.random.default_rng(PLOT_SEED)
chosen_classes = np.linspace(0, len(class_names) - 1, N_EXAMPLES).round().astype(int)
example_idx = [int(rng.choice(np.flatnonzero(y_train == c))) for c in chosen_classes]
examples = X_train[example_idx]                       # (N, SEQ_LEN)
example_labels = class_names[y_train[example_idx]]

# Shape the batch the way the model receives it: (n_cells, seq_len, 1)
x = torch.from_numpy(examples).float().unsqueeze(-1)
time_min = np.arange(SEQ_LEN) * MINUTES_PER_TP


# ── The two recipes, defined exactly as in make_augment_fn() ──────────────────
def jitter_only(t):
    return jitter_torch(t, sigma=JITTER_SIGMA_V2)


def composed(t):
    t = jitter_torch(t, sigma=JITTER_SIGMA)                                   # sensor / shot noise
    t = scaling_torch(t, sigma=SCALING_SIGMA)                                 # per-trace gain
    t = magnitude_warp_torch(t, sigma=MAGWARP_SIGMA, knot=MAGWARP_KNOT)       # drift / bleaching
    t = window_slice_torch(t, reduce_ratio=WINSLICE_RATIO)                    # timing / phase
    return t


RECIPES = [("jitter_only (control)", jitter_only), ("composed", composed)]


# ── Figure 1: original vs the two SupCon views, per recipe ────────────────────
torch.manual_seed(PLOT_SEED)
np.random.seed(PLOT_SEED)   # window_slice / magnitude_warp draw from numpy's RNG

fig, axes = plt.subplots(len(RECIPES), N_EXAMPLES,
                         figsize=(5 * N_EXAMPLES, 4 * len(RECIPES)),
                         sharex=True, sharey=True, constrained_layout=True)

for row, (recipe_name, fn) in enumerate(RECIPES):
    # Two independent draws == the positive pair the contrastive loss sees
    view1 = fn(x).squeeze(-1).numpy()
    view2 = fn(x).squeeze(-1).numpy()
    for col in range(N_EXAMPLES):
        ax = axes[row, col]
        ax.plot(time_min, examples[col], color=COL_ORIG, lw=1.6,
                label="original", zorder=3)
        ax.plot(time_min, view1[col], color=COL_V1, lw=1.0, alpha=0.85, label="view 1")
        ax.plot(time_min, view2[col], color=COL_V2, lw=1.0, alpha=0.85, label="view 2")
        ax.set_title(f"{recipe_name}\n{example_labels[col]}")
        if row == len(RECIPES) - 1:
            ax.set_xlabel("Time / min")
        if col == 0:
            ax.set_ylabel("Standardised fluorescence / a.u.")

# One legend for the whole figure, outside the axes so it cannot cover data
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper right", ncol=3, frameon=False)
fig.suptitle("SupCon views before vs after augmentation (IY036 v3 A/B)", fontsize=14)
out1 = IY036_DIR / "IY036_augmentation_before_after.png"
fig.savefig(out1, dpi=300)
plt.close(fig)
print(f"Saved {out1}")


# ── Figure 2: the composed recipe, one operation at a time ────────────────────
torch.manual_seed(PLOT_SEED)
np.random.seed(PLOT_SEED)

# Cumulative stages: each entry is the tensor after one more op has been applied
stages = [("original", x)]
t = jitter_torch(x, sigma=JITTER_SIGMA)
stages.append((f"+ jitter (σ={JITTER_SIGMA})", t))
t = scaling_torch(t, sigma=SCALING_SIGMA)
stages.append((f"+ scaling (σ={SCALING_SIGMA})", t))
t = magnitude_warp_torch(t, sigma=MAGWARP_SIGMA, knot=MAGWARP_KNOT)
stages.append((f"+ magnitude warp (σ={MAGWARP_SIGMA}, knot={MAGWARP_KNOT})", t))
t = window_slice_torch(t, reduce_ratio=WINSLICE_RATIO)
stages.append((f"+ window slice (ratio={WINSLICE_RATIO})", t))

fig, axes = plt.subplots(N_EXAMPLES, len(stages),
                         figsize=(5 * len(stages), 4 * N_EXAMPLES),
                         sharex=True, sharey=True, constrained_layout=True)

for row in range(N_EXAMPLES):
    for col, (stage_name, stage_tensor) in enumerate(stages):
        ax = axes[row, col]
        # Faint original in every panel as the common reference
        ax.plot(time_min, examples[row], color=COL_ORIG, lw=1.0, alpha=0.35,
                label="original")
        if col > 0:
            ax.plot(time_min, stage_tensor[row, :, 0].numpy(), color=COL_V1,
                    lw=1.2, label="augmented")
        if row == 0:
            ax.set_title(stage_name)
        if row == N_EXAMPLES - 1:
            ax.set_xlabel("Time / min")
        if col == 0:
            ax.set_ylabel(f"{example_labels[row]}\nStandardised fluorescence / a.u.")

handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper right", ncol=2, frameon=False)
fig.suptitle("Composed augmentation, cumulative effect of each operation", fontsize=14)
out2 = IY036_DIR / "IY036_composed_augmentation_steps.png"
fig.savefig(out2, dpi=300)
plt.close(fig)
print(f"Saved {out2}")


# ── Quantify how far each recipe moves a trace (sanity check on the A/B) ──────
torch.manual_seed(PLOT_SEED)
np.random.seed(PLOT_SEED)
X_all = torch.from_numpy(X_train).float().unsqueeze(-1)
for recipe_name, fn in RECIPES:
    aug = fn(X_all).squeeze(-1).numpy()
    # RMS deviation per trace, relative to the trace's own RMS amplitude
    rms_dev = np.sqrt(np.mean((aug - X_train) ** 2, axis=1))
    rms_sig = np.sqrt(np.mean(X_train ** 2, axis=1))
    ratio = rms_dev / rms_sig
    print(f"{recipe_name:>22}: RMS(aug - orig) / RMS(orig) = "
          f"{ratio.mean():.2f} ± {ratio.std():.2f} (mean ± sd over {len(ratio)} cells)")
