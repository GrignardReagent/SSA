"""
IY039: Optuna hyperparameter/architecture sweep for the SimCLR transformer
encoder (cross-view InfoNCE), pretrained on the 4 IY020 synthetic
telegraph-model datasets with instance normalisation -- same pretraining
setup as IY024, but every architecture/training knob is now swept instead of
hand-picked.

Checkpoint selection + early stopping are gated on a downstream KNN readout
on the real 6-class experimental benchmark (Nrg1/Rtg1 x 3 glucose conditions),
the same downstream task tracked in IY036, via the new eval_fn/eval_metric_key
support added to `train_ssl_model` (see src/training/train.py). Unlike IY036,
no pretraining-pool exclusion is needed: the pretraining data here (synthetic
IY020 trajectories) is categorically disjoint from this real experimental
benchmark, so there's no leakage path.

Runs as one of several parallel Optuna workers sharing a single
JournalStorage-backed study -- see IY039_submit_sweep.sh for how the 4 Eddie
jobs are launched against the same storage/study_name so their trial numbers
(and therefore checkpoint/run names) are globally unique.
"""

import argparse
import gc
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
import wandb
import yaml
from open_lars import LARS
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup

from dataloaders.simclr import ssl_data_prep
from models.ssl_transformer import SSL_Transformer
from training.train import cross_view_info_nce, train_ssl_model
from utils.embeddings import make_knn_eval_fn
from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.processing.pipeline import prepare_dataset

# ── Paths ───────────────────────────────────────────────────────────────────
IY039_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = IY039_DIR.parent
IY020_DIR = EXPERIMENTS_DIR / "EXP-26-IY020"
IY008_DIR = EXPERIMENTS_DIR / "EXP-25-IY008"
FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series"
META_PATH = IY008_DIR / "old_data_metadata.csv"
META_COLS = ["id", "group", "experiment"]
YAML_CONFIG_PATH = IY039_DIR / "IY039_sweep_config.yaml"

# ── Downstream benchmark config (identical to IY036) ───────────────────────
FILE_RE = re.compile(r"^(\d+)_.*_group_(.+?)_(GFP|mCherry)_time_series$")
EXCLUDED_EXPS = {"18446"}  # not properly recorded, excluded from all analyses
FIXED_CLASSES = [
    "Nrg1 @ 0.01% glucose", "Nrg1 @ 0.1% glucose", "Nrg1 @ 2% glucose (mock/steady)",
    "Rtg1 @ 0.01% glucose", "Rtg1 @ 0.1% glucose", "Rtg1 @ 2% glucose (mock/steady)",
]
RANDOM_STATE = 42
VAL_FRACTION = 0.20
K_NEIGHBORS = 10  # matches IY032/IY035/IY036's grid-search k

# ── Fixed (not swept) pretraining config ────────────────────────────────────
NORMALISATION = "instance"
SAMPLE_LEN = 500
NUM_TRAJ = 1
LOG_SCALE = False
GRAD_CLIP = None          # matches IY024's canonical script
EVAL_EVERY = 10
SMOOTH_WINDOW = 3
EVAL_METRIC_KEY = "knn_val_acc_smooth"

# ── Optimizer-branch config ─────────────────────────────────────────────────
LARGE_BATCH_THRESHOLD = 1000  # batch_size > this -> LARS with linear-scaling-rule lr
LARS_MOMENTUM = 0.9              # IY036 Eddie's validated default, not swept
LARS_TRUST_COEFFICIENT = 0.001   # "eta" in the LARS paper, IY036 Eddie's default

WANDB_ENTITY = "grignard-reagent"
WANDB_PROJECT = "IY039-SSL-sweep"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Pretraining data: 4 IY020 sources combined, same filter/paths as IY024 ──
DATA_SOURCES = [
    (IY020_DIR / "data", "IY020_simulation_parameters_sobol.csv"),
    (IY020_DIR / "data_mu_variation", "IY020_simulation_mu_parameters_sobol.csv"),
    (IY020_DIR / "data_cv_variation", "IY020_simulation_cv_parameters_sobol.csv"),
    (IY020_DIR / "data_t_ac_variation", "IY020_simulation_t_ac_parameters_sobol.csv"),
]


def _collect_trajectory_paths() -> list[Path]:
    paths = []
    for data_root, results_csv in DATA_SOURCES:
        df = pd.read_csv(data_root / results_csv)
        df = df[(df["success"] == True) &
                (df["error_message"].isna()) &
                (df["mean_rel_error_pct"] < 10) &
                (df["cv_rel_error_pct"] < 10) &
                (df["t_ac_rel_error_pct"] < 10)]
        paths.extend(data_root / f for f in df["trajectory_filename"].values)
        print(f"  {data_root.name}: {len(df)} trajectories")
    return paths


TRAJ_PATHS = _collect_trajectory_paths()
print(f"Total pretraining trajectories: {len(TRAJ_PATHS)}")

# Loaders only vary with batch_size (sample_len/num_traj/normalisation are fixed),
# so cache by batch_size to avoid re-splitting/re-scanning files every trial that
# repeats a categorical batch_size value.
_LOADER_CACHE: dict[int, tuple] = {}


def get_loaders(batch_size: int):
    if batch_size not in _LOADER_CACHE:
        _LOADER_CACHE[batch_size] = ssl_data_prep(
            TRAJ_PATHS, batch_size=batch_size, sample_len=SAMPLE_LEN,
            log_scale=LOG_SCALE, normalisation=NORMALISATION, num_traj=NUM_TRAJ,
        )
    return _LOADER_CACHE[batch_size]


# ── Downstream eval data: same 6-class experimental benchmark as IY036 ─────
metadata = pd.read_csv(META_PATH)
metadata["exp_id"] = metadata["exp_id"].astype(str)
metadata["group_id"] = metadata["group_id"].astype(str)
LABEL_LOOKUP = {(r.exp_id, r.group_id, r.channel): (r.tf, r.condition)
                for _, r in metadata.iterrows()}

full_ts_raw, full_label_strs = load_labelled_time_series_csvs(
    data_dir=FULL_DATA_DIR, file_re=FILE_RE, label_lookup=LABEL_LOOKUP,
    meta_cols=META_COLS, excluded_exps=EXCLUDED_EXPS, verbose=False,
)
d = prepare_dataset(full_ts_raw, full_label_strs, FIXED_CLASSES, "Full", RANDOM_STATE)

# Held out from the downstream 6-class TRAIN split only -- d["X_test"] stays
# untouched (no downstream test-set metric is computed in this sweep script).
X_down_tr, X_down_val, y_down_tr, y_down_val = train_test_split(
    d["X_train"], d["y_train"], test_size=VAL_FRACTION,
    random_state=RANDOM_STATE, stratify=d["y_train"])
print(f"Downstream train/val split: {len(y_down_tr)} train, {len(y_down_val)} val "
      f"(val_fraction={VAL_FRACTION})")


def _suggest(trial: optuna.Trial, key: str, spec: dict):
    if "values" in spec:
        return trial.suggest_categorical(key, spec["values"])
    if "value" in spec:
        return spec["value"]
    return trial.suggest_float(key, spec["min"], spec["max"], log=spec.get("log", False))


def build_optimizer(model, batch_size: int, lr: float, lars_base_lr: float, weight_decay: float):
    """AdamW for ordinary batch sizes; LARS + linear-scaling-rule lr for large ones
    (mirrors IY036's local-vs-Eddie split, as one batch-size-gated branch)."""
    if batch_size > LARGE_BATCH_THRESHOLD:
        actual_lr = lars_base_lr * batch_size / 256
        optimizer = LARS(model.parameters(), lr=actual_lr, momentum=LARS_MOMENTUM,
                          weight_decay=weight_decay, trust_coefficient=LARS_TRUST_COEFFICIENT)
        return optimizer, actual_lr, "LARS"
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, lr, "AdamW"


def objective(trial: optuna.Trial) -> float:
    with open(YAML_CONFIG_PATH) as f:
        params_def = yaml.safe_load(f)["parameters"]

    nhead = _suggest(trial, "nhead", params_def["nhead"])
    num_layers = _suggest(trial, "num_layers", params_def["num_layers"])
    d_model = _suggest(trial, "d_model", params_def["d_model"])
    dropout = _suggest(trial, "dropout", params_def["dropout"])
    use_conv1d = _suggest(trial, "use_conv1d", params_def["use_conv1d"])
    batch_size = _suggest(trial, "batch_size", params_def["batch_size"])
    lr = _suggest(trial, "lr", params_def["lr"])
    lars_base_lr = _suggest(trial, "lars_base_lr", params_def["lars_base_lr"])
    weight_decay = _suggest(trial, "weight_decay", params_def["weight_decay"])
    nce_temp = _suggest(trial, "nce_temp", params_def["nce_temp"])
    epochs = _suggest(trial, "epochs", params_def["epochs"])
    patience = epochs // (EVAL_EVERY * 3)  # matches IY036's convention

    model, optimizer, run = None, None, None
    try:
        train_loader, val_loader, _ = get_loaders(batch_size)
        X1_b, _, _ = next(iter(train_loader))
        input_size = X1_b.shape[2]

        model = SSL_Transformer(
            input_size=input_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, use_conv1d=use_conv1d,
        ).to(DEVICE)

        optimizer, actual_lr, opt_name = build_optimizer(model, batch_size, lr, lars_base_lr, weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * epochs), num_training_steps=epochs)
        loss_fn = lambda q, k: cross_view_info_nce(q, k, temperature=nce_temp)
        eval_fn = make_knn_eval_fn(
            X_down_tr, y_down_tr, X_down_val, y_down_val, DEVICE,
            n_neighbors=K_NEIGHBORS, smooth_window=SMOOTH_WINDOW,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = IY039_DIR / (
            f"IY039_sweep_trial{trial.number}_b{batch_size}_lr{actual_lr:.2e}_"
            f"L{num_layers}_H{nhead}_D{d_model}_temp{nce_temp:.2f}_{timestamp}_model.pth"
        )
        run_name = (
            f"trial{trial.number}_b{batch_size}_lr{actual_lr:.2e}_d{dropout:.2f}_"
            f"L{num_layers}_H{nhead}_D{d_model}_temp{nce_temp:.2f}_instance"
        )

        wandb_config = {
            "entity": WANDB_ENTITY, "project": WANDB_PROJECT, "name": run_name,
            "trial_number": trial.number,
            "dataset": [str(p) for p, _ in DATA_SOURCES], "normalisation": NORMALISATION,
            "batch_size": batch_size, "input_size": input_size,
            "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
            "dropout": dropout, "use_conv1d": use_conv1d,
            "epochs": epochs, "patience": patience, "eval_every": EVAL_EVERY,
            "eval_metric_key": EVAL_METRIC_KEY, "smooth_window": SMOOTH_WINDOW,
            "knn_neighbors": K_NEIGHBORS,
            "lr": lr, "lars_base_lr": lars_base_lr, "actual_lr": actual_lr,
            "weight_decay": weight_decay, "optimizer": opt_name,
            "lars_momentum": LARS_MOMENTUM if opt_name == "LARS" else None,
            "lars_trust_coefficient": LARS_TRUST_COEFFICIENT if opt_name == "LARS" else None,
            "scheduler": type(scheduler).__name__, "loss_fn": "cross_view_info_nce",
            "nce_temp": nce_temp, "sample_len": SAMPLE_LEN, "num_traj_per_view": NUM_TRAJ,
            "log_scale": LOG_SCALE, "grad_clip": GRAD_CLIP, "save_path": str(save_path),
            "total_trajectories": len(TRAJ_PATHS),
            "num_cells_downstream_train_fit": len(y_down_tr),
            "num_cells_downstream_val": len(y_down_val),
        }

        history, run = train_ssl_model(
            model, train_loader, val_loader, epochs=epochs, patience=patience,
            optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=DEVICE,
            grad_clip=GRAD_CLIP, save_path=str(save_path),
            eval_fn=eval_fn, eval_every=EVAL_EVERY, eval_metric_key=EVAL_METRIC_KEY,
            wandb_logging=True, wandb_config=wandb_config, verbose=False, return_run=True,
        )

        eval_history = history.get(f"eval/{EVAL_METRIC_KEY}", [])
        best_value = max(eval_history) if eval_history else 0.0
        if run is not None:
            run.summary["best_" + EVAL_METRIC_KEY] = best_value
            run.finish()
        print(f"Trial {trial.number} finished. Best {EVAL_METRIC_KEY}: {best_value:.4f}")
        return float(best_value)

    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {type(e).__name__}: {e}")
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        return 0.0  # below chance (~0.167 for 6 classes) -- steers the sampler away

    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", required=True,
                         help="Path to the shared Optuna JournalStorage file "
                              "(identical across all parallel workers).")
    parser.add_argument("--study-name", required=True,
                         help="Optuna study name (identical across all parallel workers).")
    parser.add_argument("--n-trials", type=int, default=25,
                         help="Trials this worker runs (total across all workers = n_trials * n_workers).")
    args = parser.parse_args()

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.storage_path)
    )
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize", load_if_exists=True,
    )
    # catch=(Exception,): a trial that raises OUTSIDE objective's own try/except
    # (e.g. during YAML parsing) still doesn't kill this worker's remaining trials.
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    print(f"Worker finished {args.n_trials} trials. Study now has {len(study.trials)} total trials.")
