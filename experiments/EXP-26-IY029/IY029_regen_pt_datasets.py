#!/usr/bin/env python3
"""
IY029 — Regenerate static pairwise `.pt` datasets at a larger dataset size.

Each of the 8 conditions (IY011×4 + IY014×4) currently has 3 000/600/600
train/val/test samples.  This script regenerates them at:

    num_groups_train = 10 000
    num_groups_val   =  2 000  (20 % of train)
    num_groups_test  =  2 000  (20 % of train)
    total            = 14 000

Generation pipeline (unchanged from original training scripts):
  1. Glob all .npz files in the source directory (one file per parameter set,
     each containing 1 000 simulated trajectories).
  2. Split the files 64/16/20 at the FILE level → zero trajectory leakage
     across splits.
  3. BaselineDataset generates pairwise samples on-the-fly:
       positive (label=1): 2 trajectories from the SAME parameter set
       negative (label=0): 1 trajectory each from 2 DIFFERENT parameter sets
                           with log-Euclidean distance > param_dist_threshold
     Trajectories are concatenated in time with a separator token (-100) and
     instance-normalised per sample.
  4. save_loader_to_disk() materialises the loader into a static .pt file
     containing dict{'X': (N, T, 1), 'y': (N, 1)}.

New files are written to EXP-26-IY029/data/{2_fold,10_fold}/{condition}/
(e.g. `EXP-26-IY029/data/2_fold/baseline/IY029_static_train.pt`).
Source data directories are never modified.
Run with:
    micromamba run -n stochastic_sim python IY029_regen_pt_datasets.py

To submit to Eddie HPC, use the companion shell script:
    qsub IY029_regen_pt_datasets.sh
"""

import sys
from pathlib import Path

# Allow importing src/ modules regardless of working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataloaders import baseline_data_prep, save_loader_to_disk

# ── Dataset size ──────────────────────────────────────────────────────────────
NUM_GROUPS_TRAIN = 10_000
NUM_GROUPS_VAL   = int(NUM_GROUPS_TRAIN * 0.2)   # 2 000
NUM_GROUPS_TEST  = int(NUM_GROUPS_TRAIN * 0.2)   # 2 000
BATCH_SIZE       = 64
NUM_TRAJ         = 2   # trajectories per pairwise sample

# ── Experiment roots ──────────────────────────────────────────────────────────
IY011_ROOT = REPO_ROOT / "experiments" / "EXP-25-IY011"
IY014_ROOT = REPO_ROOT / "experiments" / "EXP-26-IY014"
IY029_ROOT = REPO_ROOT / "experiments" / "EXP-26-IY029"
IY029_DATA = IY029_ROOT / "data"   # all .pt files land here

# ── Job definitions ───────────────────────────────────────────────────────────
# Each entry: npz_dir (source NPZ files), save_dir (output subfolder),
#             param_dist_threshold (log-distance threshold for negative pairs).
#
# Outputs are always named  IY029_static_{train,val,test}.pt
# and written to a  2_fold/  or  10_fold/  subfolder so originals are untouched.
#
# param_dist_threshold:
#   2-fold  (IY011) → 0.7  (pairs must differ by ≥ 2-fold in log-parameter space)
#   10-fold (IY014) → 2.3  (pairs must differ by ≥ 10-fold)
JOBS = [
    # ── 2-fold conditions (IY011 NPZ pools) ──────────────────────────────────
    dict(
        desc="2-fold baseline (all-varying)",
        npz_dir=IY011_ROOT / "data",
        save_dir=IY029_DATA / "2_fold" / "baseline",
        param_dist_threshold=0.7,
    ),
    dict(
        desc="2-fold mu variation",
        npz_dir=IY011_ROOT / "data_mu_variation",
        save_dir=IY029_DATA / "2_fold" / "mu",
        param_dist_threshold=0.7,
    ),
    dict(
        desc="2-fold CV variation",
        npz_dir=IY011_ROOT / "data_cv_variation",
        save_dir=IY029_DATA / "2_fold" / "cv",
        param_dist_threshold=0.7,
    ),
    dict(
        desc="2-fold t_ac variation",
        npz_dir=IY011_ROOT / "data_t_ac_variation",
        save_dir=IY029_DATA / "2_fold" / "t_ac",
        param_dist_threshold=0.7,
    ),
    # ── 10-fold conditions (IY014 NPZ pools; mu reuses IY011's pool) ─────────
    dict(
        desc="10-fold baseline (all-varying)",
        npz_dir=IY014_ROOT / "data",
        save_dir=IY029_DATA / "10_fold" / "baseline",
        param_dist_threshold=2.3,
    ),
    dict(
        # mu variation: IY014 uses the same NPZ pool as IY011 (same simulation,
        # just a stricter pair-distance threshold for the 10-fold fold label)
        desc="10-fold mu variation",
        npz_dir=IY011_ROOT / "data_mu_variation",
        save_dir=IY029_DATA / "10_fold" / "mu",
        param_dist_threshold=2.3,
    ),
    dict(
        desc="10-fold CV variation",
        npz_dir=IY014_ROOT / "data_cv_variation",
        save_dir=IY029_DATA / "10_fold" / "cv",
        param_dist_threshold=2.3,
    ),
    dict(
        desc="10-fold t_ac variation",
        npz_dir=IY014_ROOT / "data_t_ac_variation",
        save_dir=IY029_DATA / "10_fold" / "t_ac",
        param_dist_threshold=2.3,
    ),
]


PREFIX = "IY029"   # all output files share this prefix


def regen_condition(job: dict) -> None:
    """Regenerate IY029_static_{train,val,test}.pt for one condition."""
    desc     = job["desc"]
    npz_dir  = Path(job["npz_dir"])
    save_dir = Path(job["save_dir"])
    thresh   = job["param_dist_threshold"]

    print(f"\n{'='*70}")
    print(f"Condition : {desc}")
    print(f"NPZ dir   : {npz_dir}")
    print(f"Save dir  : {save_dir}")
    print(f"Prefix    : {PREFIX}_static_{{train,val,test}}.pt")
    print(f"Threshold : {thresh}")
    print(f"Sizes     : train={NUM_GROUPS_TRAIN}, val={NUM_GROUPS_VAL}, "
          f"test={NUM_GROUPS_TEST}")
    print(f"{'='*70}")

    # 1. Create output subfolder (2_fold/ or 10_fold/) if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Discover all NPZ files (one per simulated parameter set)
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")
    print(f"Found {len(npz_files)} NPZ files.")

    # 3. Build dynamic DataLoaders (file-level split, no data leakage)
    train_loader, val_loader, test_loader = baseline_data_prep(
        all_file_paths=npz_files,
        batch_size=BATCH_SIZE,
        num_groups_train=NUM_GROUPS_TRAIN,
        num_groups_val=NUM_GROUPS_VAL,
        num_groups_test=NUM_GROUPS_TEST,
        num_traj=NUM_TRAJ,
        param_dist_threshold=thresh,
        verbose=True,
    )

    # 4. Materialise each split to disk
    for split, loader in [("train", train_loader),
                           ("val",   val_loader),
                           ("test",  test_loader)]:
        out_path = save_dir / f"{PREFIX}_static_{split}.pt"
        save_loader_to_disk(loader, out_path)
        print(f"  ✅ {out_path.relative_to(IY029_ROOT)} saved.")


def main() -> None:
    print(f"Regenerating {len(JOBS)} conditions — "
          f"n_train={NUM_GROUPS_TRAIN}, n_val={NUM_GROUPS_VAL}, "
          f"n_test={NUM_GROUPS_TEST} per condition.")

    for i, job in enumerate(JOBS, 1):
        print(f"\n[{i}/{len(JOBS)}]", end="")
        regen_condition(job)

    print("\n\nAll conditions regenerated successfully.")
    print(f"Total samples per condition: "
          f"{NUM_GROUPS_TRAIN + NUM_GROUPS_VAL + NUM_GROUPS_TEST:,}")


if __name__ == "__main__":
    main()
