#!/usr/bin/env python3
"""Train and fine-tune the TFTransformer on synthetic then experimental
data.

This mirrors the intended workflow for TF trajectory classification:

1. **Pre‑train** on a large synthetic dataset produced by the telegraph model
   (see ``IY010_simulation.py``) to learn generic representations.  The simulation
   script produces many CSV files, each containing a batch of trajectories with a
   ``label`` column.
2. **Fine‑tune** on a smaller experimental dataset by freezing the encoder and
   re‑initialising the classifier head.  The experimental measurements are
   stored in a TSV file with columns ``id``, ``group`` (class label), ``time``
   and ``CV_mCherry`` which are reshaped into trajectories.

Trailing zeros in the final wide-form tables denote padding and are ignored via
an attention mask.  The file locations are hard‑coded below to match the output
of ``IY010_simulation.py`` and the provided experimental recording.  Adjust as
necessary for your environment.  If a dataset is absent the corresponding phase
is skipped and a warning is issued.  Models are saved alongside this script as
``IY010_pretrained.pt`` and ``IY010_finetuned.pt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import re
from collections import defaultdict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.TF_transformer import TFTransformer, ModelCfg
from utils.standardise_time_series import standardise_time_series
from utils.data_processing import add_binary_labels


# ---------------------------------------------------------------------------
# File locations and basic training hyper-parameters.  The synthetic dataset
# path mirrors the output location of ``IY010_simulation.py`` whereas the
# experimental dataset is a placeholder for unseen trajectories.  Adjust as
# necessary for your environment.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Directory containing many CSV files produced by ``IY010_simulation_6.py``
SYNTHETIC_DIR = BASE_DIR / "data_6"

# Experimental measurements: use pre-transformed wide time-series CSVs
EXPERIMENTAL_DIR = BASE_DIR / "transformed_exp_time_series_data"
OUT_DIR = BASE_DIR
# Training schedules
EPOCHS_PRETRAIN = 50
LR_PRETRAIN = 1e-3

# Requested: full unfreeze, smaller LR, longer schedule
EPOCHS_FINETUNE = 50
LR_FINETUNE = 1e-4

BATCH_SIZE = 32
PARAMS_CSV = BASE_DIR / "IY010_simulation_parameters_6.csv"

# Reproducibility
SEED = 42

def _set_seed(seed: int = SEED) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# currently this is sorting the labels by cv: lower cv --> label 0, higher cv --> label 1
def _add_pair_labels(df: pd.DataFrame) -> None:
    """Append a ``label`` column to trajectories lacking one.

    Synthetic CSV files produced by ``IY010_simulation.py`` are named
    ``mRNA_trajectories_<mu>_<cv>_<t_ac>.csv``.  For each ``mu``/``t_ac``
    combination there are many ``cv`` values.  The loader records the filename
    stem in a temporary ``source`` column.  Files are paired by increasing ``cv``
    and labelled ``0``/``1`` respectively.
    """

    # Extract parameters from filenames like "mRNA_trajectories_1.0_0.5_2.0.csv"
    # where the numbers are mu, cv, and t_ac values
    pattern = re.compile(r"mRNA_trajectories_([0-9.]+)_([0-9.]+)_([0-9.]+)")
    
    # Group files by their mu and t_ac values (ignoring cv for now)
    # This creates groups of files that have the same mu and t_ac but different cv values
    groups: dict[tuple[float, float], list[tuple[float, str]]] = defaultdict(list) # The defaultdict(list) means when you access a key that doesn't exist, it automatically creates an empty list.
    
    for src in df["source"].unique():
        # ``source`` stores the filename stem of the synthetic CSV each row
        # originated from.  Grouping by it lets us pair trajectories from the
        # same parameter sweep.
        match = pattern.match(src)
        if not match:
            raise ValueError(f"unrecognised filename pattern: {src}")
        mu, cv, t_ac = map(float, match.groups())
        # Group files by (mu, t_ac) and store their cv values with filenames
        groups[(mu, t_ac)].append((cv, src))

    # Now assign labels by pairing files within each group
    label_map: dict[str, int] = {}
    for key, items in groups.items():
        # Sort files by cv value (lowest to highest)
        items.sort(key=lambda x: x[0])
        
        # Pair files: take every two consecutive files and label them 0 and 1
        # The file with lower cv gets label 0, the file with higher cv gets label 1
        for i in range(0, len(items), 2):  # Step by 2 to process pairs
            if i + 1 >= len(items):  # Handle odd number of files - label the last file as 0
                _, src_last = items[i]
                label_map[src_last] = 0
                continue
            # Get the two files to pair (lower cv and higher cv)
            _, src0 = items[i]      # File with lower cv value
            _, src1 = items[i + 1]  # File with higher cv value
            # Assign labels: lower cv = 0, higher cv = 1
            label_map[src0] = 0
            label_map[src1] = 1

    # Check that we successfully labeled all files
    if len(label_map) != df["source"].nunique():
        raise ValueError("failed to assign labels to all synthetic CSV files")

    # Apply the labels to the dataframe
    df["label"] = df["source"].map(label_map)


def _prepare_dataset(df: pd.DataFrame) -> TensorDataset:
    """Convert a DataFrame into a :class:`TensorDataset`.

    If the ``label`` column is missing, labels are assigned by pairing
    trajectories based on the statistics encoded in their source file names.
    Trailing zeros are interpreted as padding and ignored via a key-padding
    mask.
    """

    if "label" not in df.columns:
        # add label to synthetic data
        _add_pair_labels(df)
    if "source" in df.columns:
        # ``source`` is the file stem for each trajectory CSV file and is only
        # needed for automatic labelling, so drop it before converting to
        # tensors.
        df.drop(columns=["source"], inplace=True)

    labels = torch.tensor(df["label"].values, dtype=torch.long)
    series = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)

    # Infer lengths from non-zero padding (legacy path)
    lengths = (series != 0).sum(dim=1)
    max_len = series.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    mean = (series * mask).sum(dim=1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    var = ((series - mean).pow(2) * mask).sum(dim=1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    std = var.sqrt()
    series = (series - mean) / (std + 1e-8)
    series[~mask] = 0.0
    series = series.unsqueeze(-1)
    return TensorDataset(series, lengths, labels)


def _prepare_equal_length_dataset(df: pd.DataFrame) -> TensorDataset:
    """Convert a wide equal-length DataFrame into a TensorDataset.

    Expects a ``label`` column followed by time-step columns of equal length
    across all rows. Unlike :func:`_prepare_dataset`, this does not try to
    infer padding from zeros and instead sets all sequence lengths to the
    common width. Per-series normalisation (zero mean, unit std) is applied.
    """

    if "label" not in df.columns:
        raise ValueError("expected a 'label' column for equal-length dataset")

    # Replace any missing values introduced during preprocessing
    df = df.copy().fillna(0)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    series = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)

    # Per-series normalisation (no masking needed as lengths are equal)
    mean = series.mean(dim=1, keepdim=True)
    std = series.std(dim=1, keepdim=True)
    series = (series - mean) / (std + 1e-8)

    # Build lengths vector: all rows have the same sequence length
    seq_len = series.size(1)
    lengths = torch.full((series.size(0),), fill_value=seq_len, dtype=torch.long)

    series = series.unsqueeze(-1)
    return TensorDataset(series, lengths, labels)


def _load_synthetic_dataset(path: Path) -> Tuple[TensorDataset, int]:
    """Load and concatenate all synthetic CSV files in ``path``.

    If the files lack a ``label`` column they are automatically paired and
    labelled using :func:`_add_pair_labels`.

    """

    csv_files = sorted(p for p in path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no synthetic CSV files found in {path}")

    # Load per-file DataFrames and standardise lengths across all synthetic CSVs
    raw_frames: List[pd.DataFrame] = []
    sources: List[str] = []
    for p in csv_files:
        df = pd.read_csv(p)
        raw_frames.append(df)
        sources.append(p.stem)

    # Use the utility to enforce equal temporal length across files
    std_df = standardise_time_series(raw_frames, prefix="t_")

    # Reconstruct per-row source labels (in the same order as concatenation)
    counts = [len(df) for df in raw_frames]
    src_col: list[str] = []
    for src, n in zip(sources, counts):
        src_col.extend([src] * n)
    std_df.insert(0, "source", src_col)

    # Load parameter mapping and attach target statistics for labelling
    if not PARAMS_CSV.exists():
        raise FileNotFoundError(f"missing parameters CSV: {PARAMS_CSV}")
    params = pd.read_csv(PARAMS_CSV, usecols=["trajectory_filename", "mu_target", "cv_target", "t_ac_target"])  # noqa: E501

    std_df["trajectory_filename"] = std_df["source"] + ".csv"
    merged = std_df.merge(params, on="trajectory_filename", how="left")
    if merged[["mu_target", "cv_target", "t_ac_target"]].isna().any().any():
        missing = merged[merged["mu_target"].isna()]["source"].unique()[:5]
        raise ValueError(
            f"parameter rows missing for some synthetic files (e.g. {list(missing)})."
        )

    # Use utility to create binary labels based on CV (upper half = 1)
    labelled = add_binary_labels(merged, column="cv_target")
    labelled.insert(0, "label", labelled.pop("label"))
    labelled = labelled.drop(columns=["source", "trajectory_filename", "mu_target", "cv_target", "t_ac_target"])  # noqa: E501

    dataset = _prepare_equal_length_dataset(labelled)
    return dataset, labelled["label"].nunique()


def _load_experimental_from_dir(dir_path: Path, test_files: List[Path] | None = None) -> Tuple[TensorDataset, TensorDataset, int, List[Path]]:
    """Load experimental wide-form CSVs and create train/test datasets.

    Parameters
    ----------
    dir_path:
        Directory containing wide-form CSVs with columns ``id``, ``group``,
        ``experiment`` and then time-point columns.
    test_files:
        Optional explicit list of files to use as the test set. If ``None``,
        the first two files in lexicographic order are used.

    Returns
    -------
    (train_ds, test_ds, n_classes, used_test_files)
        Tensor datasets for training and testing, the number of classes (unique
        groups) and the list of paths used for testing.
    """
    files = sorted(p for p in dir_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no experimental CSV files found in {dir_path}")

    if test_files is None:
        # Prefer holding out files from groups that appear in multiple files,
        # so the model still sees those classes during training.
        group_to_files: dict[str, list[Path]] = {}
        for p in files:
            try:
                g = pd.read_csv(p, usecols=["group"])  # small IO compared to full load
                groups = list(map(str, g["group"].unique()))
                # Each file should contain a single group; use the first
                grp = groups[0] if groups else None
                if grp is None:
                    continue
                group_to_files.setdefault(grp, []).append(p)
            except Exception:
                continue
        candidate_files: list[Path] = []
        for grp, fs in group_to_files.items():
            if len(fs) >= 2:
                # hold out the first file for this group
                candidate_files.append(sorted(fs)[0])
        # Take up to two such files; fallback to first two overall if insufficient
        test_files = candidate_files[:2] if len(candidate_files) >= 2 else files[:2]
    test_set = set(test_files)
    train_files = [p for p in files if p not in test_set]

    # Load raw DataFrames and keep their labels
    def read_exp(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p)
        # Ensure expected columns exist
        if not {"id", "group"}.issubset(df.columns):
            raise ValueError(f"experimental CSV missing required columns: {p}")
        return df

    train_raw = [read_exp(p) for p in train_files]
    test_raw = [read_exp(p) for p in test_files]

    # Build a global label map over all available groups (train+test)
    all_groups = pd.concat([df[["group"]] for df in train_raw + test_raw], ignore_index=True)[
        "group"
    ].astype(str).unique()
    label_map = {g: i for i, g in enumerate(sorted(all_groups))}

    # Extract only time columns for standardisation
    def time_only(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in ["id", "group", "experiment"] if c in df.columns])

    train_time = [time_only(df) for df in train_raw]
    test_time = [time_only(df) for df in test_raw]

    # Standardise to a single common length across both train and test
    all_std = standardise_time_series(train_time + test_time, prefix="t_")
    # Split back into train/test portions
    n_train_rows = [len(df) for df in train_raw]
    n_test_rows = [len(df) for df in test_raw]
    train_std_parts = []
    test_std_parts = []
    start = 0
    for n in n_train_rows:
        train_std_parts.append(all_std.iloc[start : start + n].copy())
        start += n
    for n in n_test_rows:
        test_std_parts.append(all_std.iloc[start : start + n].copy())
        start += n
    train_std = pd.concat(train_std_parts, ignore_index=True) if train_std_parts else pd.DataFrame()
    test_std = pd.concat(test_std_parts, ignore_index=True) if test_std_parts else pd.DataFrame()

    # Attach labels back (group per row from source file)
    def attach_labels(std_df: pd.DataFrame, raws: List[pd.DataFrame]) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        start = 0
        for raw in raws:
            n = len(raw)
            part = std_df.iloc[start : start + n].copy()
            part.insert(0, "label", [label_map[str(g)] for g in raw["group"].values])
            parts.append(part)
            start += n
        return pd.concat(parts, ignore_index=True)

    train_df = attach_labels(train_std, train_raw)
    test_df = attach_labels(test_std, test_raw)

    return _prepare_equal_length_dataset(train_df), _prepare_equal_length_dataset(test_df), len(label_map), test_files


def _run_epoch(
    model: TFTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """Run a single training or evaluation epoch."""

    loss_sum, correct = 0.0, 0
    for x, lengths, y in loader:
        # Move data to model's device
        x = x.to(model.device)
        lengths = lengths.to(model.device)
        y = y.to(model.device)
        
        logits = model(x, lengths)
        loss = criterion(logits, y)
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, correct / n


def _train(
    model: TFTransformer,
    data: TensorDataset,
    epochs: int,
    lr: float,
    batch_size: int,
    class_weights: torch.Tensor | None = None,
    sampler: torch.utils.data.Sampler | None = None,
    val_data: TensorDataset | None = None,
    patience: int = 10,
) -> None:
    """Simple training loop used for both phases."""

    if sampler is not None:
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler)
    else:
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    if class_weights is not None:
        class_weights = class_weights.to(model.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        loss, acc = _run_epoch(model, loader, criterion, optimiser)
        msg = f"epoch {epoch:02d} loss={loss:.4f} acc={acc:.3f}"
        if val_data is not None:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            v_loss, v_acc = _run_epoch(model, val_loader, criterion, optimiser=None)
            msg += f" | val_loss={v_loss:.4f} val_acc={v_acc:.3f}"
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(msg)
                    print(f"Early stopping: no val improvement for {patience} epochs")
                    break
        print(msg)


def _unfreeze_last_encoder_layers(model: TFTransformer, n_layers: int = 2) -> None:
    """Unfreeze the last ``n_layers`` of the Transformer encoder for fine-tuning."""
    enc = model.encoder
    if isinstance(enc, nn.DataParallel):
        enc = enc.module
    if hasattr(enc, "layers") and len(enc.layers) > 0:
        k = min(n_layers, len(enc.layers))
        for layer in enc.layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        # Fallback: unfreeze the entire encoder if structure is unknown
        for p in model.encoder.parameters():
            p.requires_grad = True


def _transfer_encoder_weights(src: TFTransformer, dst: TFTransformer) -> None:
    """Copy projection, CLS token and encoder weights from ``src`` to ``dst``."""
    # Handle potential DataParallel wrappers
    def _unwrap(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    src_proj = _unwrap(src.proj)
    dst_proj = _unwrap(dst.proj)
    dst_proj.load_state_dict(src_proj.state_dict())

    # CLS token
    _unwrap(dst).cls_token.data.copy_(_unwrap(src).cls_token.data)

    src_enc = _unwrap(src.encoder)
    dst_enc = _unwrap(dst.encoder)
    dst_enc.load_state_dict(src_enc.state_dict())


def main() -> None:
    syn_data: TensorDataset | None = None
    syn_classes = 0
    try:
        syn_data, syn_classes = _load_synthetic_dataset(SYNTHETIC_DIR)
    except FileNotFoundError:
        print("[WARN] synthetic CSV not found; skipping pre-training")
    except ValueError as e:
        print(f"[WARN] {e}; skipping pre-training")

    exp_train: TensorDataset | None = None
    exp_test: TensorDataset | None = None
    exp_classes = 0
    used_test_files: List[Path] = []
    if EXPERIMENTAL_DIR.exists():
        try:
            exp_train, exp_test, exp_classes, used_test_files = _load_experimental_from_dir(EXPERIMENTAL_DIR)
        except Exception as e:
            print(f"[WARN] failed to load experimental data: {e}; skipping fine-tuning")
    else:
        print("[WARN] experimental directory not found; skipping fine-tuning")

    # Phase 1: Pre-train with binary head on synthetic data
    if syn_data is not None:
        print("=== Pre-training on synthetic data (binary head) ===")
        pre_cfg = ModelCfg(n_classes=2, d_model=64, verbose=True)
        pre_model = TFTransformer(pre_cfg)
        pre_model.train()
        _train(pre_model, syn_data, EPOCHS_PRETRAIN, LR_PRETRAIN, BATCH_SIZE)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pre_model.save_model(str(OUT_DIR / "IY010_pretrained.pt"))
    else:
        pre_model = None

    # Phase 2: Fine-tune with 8-class head on experimental data
    if exp_train is not None and exp_test is not None:
        print("=== Fine-tuning on experimental data (full unfreeze) ===")
        n_classes = max(exp_classes, 1)
        fin_cfg = ModelCfg(n_classes=n_classes, d_model=64, verbose=True)
        model = TFTransformer(fin_cfg)
        if pre_model is not None:
            _transfer_encoder_weights(pre_model, model)
        # Ensure full unfreeze
        model.freeze_encoder(False)
        model.reset_classifier()
        # Compute class weights and stratified sampling
        labels_np = exp_train.tensors[-1].cpu().numpy()
        import numpy as _np
        counts = _np.bincount(labels_np, minlength=n_classes)
        weights = _np.where(counts > 0, counts.max() / counts, 0.0)
        class_w = torch.tensor(weights, dtype=torch.float32)
        # WeightedRandomSampler for stratified sampling
        sample_weights = weights[labels_np]
        from torch.utils.data import WeightedRandomSampler
        # Build a small validation split from the training data (stratified per class)
        import math, numpy as np
        rng = np.random.default_rng(SEED)
        idx_by_class: dict[int, list[int]] = {c: [] for c in range(n_classes)}
        for i, c in enumerate(labels_np):
            idx_by_class[int(c)].append(i)
        train_idx: list[int] = []
        val_idx: list[int] = []
        for c, idxs in idx_by_class.items():
            if not idxs:
                continue
            rng.shuffle(idxs)
            n_val = max(1, math.floor(0.1 * len(idxs)))
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
        # Subset datasets
        from torch.utils.data import Subset
        exp_train_ds = Subset(exp_train, train_idx)
        exp_val_ds = Subset(exp_train, val_idx)
        # Sampler based on training subset only
        sub_weights = weights[labels_np[train_idx]]
        sampler = WeightedRandomSampler(sub_weights, num_samples=len(train_idx), replacement=True)
        _train(
            model,
            exp_train_ds,
            EPOCHS_FINETUNE,
            LR_FINETUNE,
            BATCH_SIZE,
            class_weights=class_w,
            sampler=sampler,
            val_data=exp_val_ds,
            patience=10,
        )
        model.save_model(str(OUT_DIR / "IY010_finetuned.pt"))

        # Evaluate on held-out experimental sets
        print("=== Evaluating on held-out experimental files ===")
        eval_loader = DataLoader(exp_test, batch_size=BATCH_SIZE, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = _run_epoch(model, eval_loader, criterion, optimiser=None)
        print("Test files used:")
        for p in used_test_files:
            print(f" - {p.name}")
        print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")


if __name__ == "__main__":
    _set_seed(SEED)
    main()
