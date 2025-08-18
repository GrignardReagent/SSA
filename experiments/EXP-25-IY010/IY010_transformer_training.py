#!/usr/bin/env python3
"""Train and fine-tune the TFTransformer on synthetic then experimental
data.

This mirrors the intended workflow for CVmCherry trajectory classification:

1. **Pre‑train** on a large synthetic dataset produced by the telegraph model
   (see ``IY010.py``) to learn generic representations.  The simulation script
   emits many CSV files, each containing a batch of trajectories with a
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
from typing import Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.TF_transformer import TFTransformer, ModelCfg


# ---------------------------------------------------------------------------
# File locations and basic training hyper-parameters.  The synthetic dataset
# path mirrors the output location of ``IY010_simulation.py`` whereas the
# experimental dataset is a placeholder for unseen trajectories.  Adjust as
# necessary for your environment.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
# Directory containing many CSV files produced by ``IY010_simulation.py``
SYNTHETIC_DIR = BASE_DIR / "data"
# Experimental measurements are stored in a TSV file
EXPERIMENTAL_TSV = (
    BASE_DIR
    / "data"
    / "19316_2020_10_26_steadystate_glucose_144m_2w2_00_post_media_switch.tsv"
)
OUT_DIR = BASE_DIR
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3


def _prepare_dataset(df: pd.DataFrame) -> TensorDataset:
    """Convert a DataFrame into a :class:`TensorDataset`.

    The DataFrame must contain a ``label`` column followed by time-series
    columns representing the trajectory.  Trailing zeros are interpreted as
    padding and ignored via a key-padding mask.
    """

    labels = torch.tensor(df["label"].values, dtype=torch.long)
    series = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)

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


def _load_synthetic_dataset(path: Path) -> Tuple[TensorDataset, int]:
    """Load and concatenate all synthetic CSV files in ``path``.

    Each CSV should contain a ``label`` column and time-series columns.  The
    returned tuple contains the prepared dataset and the number of distinct
    labels observed.
    """

    csv_files = sorted(p for p in path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no synthetic CSV files found in {path}")
    df = pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)
    if "label" not in df.columns:
        raise ValueError("synthetic CSVs must include a 'label' column")
    return _prepare_dataset(df), df["label"].nunique()


def _load_experimental_dataset(path: Path) -> Tuple[TensorDataset, int]:
    """Load experimental TSV data and convert to trajectories.

    The TSV file contains measurements in tidy format with columns ``id``,
    ``group``, ``time`` and ``CV_mCherry``.  Each unique ``id`` corresponds to a
    trajectory.  ``group`` is mapped to integer class labels.
    """

    df = pd.read_csv(path, sep="\t", usecols=["id", "group", "time", "CV_mCherry"])
    df = df.pivot_table(index=["group", "id"], columns="time", values="CV_mCherry")
    df = df.sort_index(axis=1).reset_index()
    labels = df["group"]
    label_map = {g: i for i, g in enumerate(sorted(labels.unique()))}
    df["label"] = labels.map(label_map)
    df = df.drop(columns=["group", "id"])
    df = df.fillna(0)
    df.insert(0, "label", df.pop("label"))
    return _prepare_dataset(df), len(label_map)


def _run_epoch(
    model: TFTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """Run a single training or evaluation epoch."""

    loss_sum, correct = 0.0, 0
    for x, lengths, y in loader:
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
) -> None:
    """Simple training loop used for both phases."""

    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        loss, acc = _run_epoch(model, loader, criterion, optimiser)
        print(f"epoch {epoch:02d} loss={loss:.4f} acc={acc:.3f}")


def main() -> None:
    syn_data: TensorDataset | None = None
    syn_classes = 0
    try:
        syn_data, syn_classes = _load_synthetic_dataset(SYNTHETIC_DIR)
    except FileNotFoundError:
        print("[WARN] synthetic CSV not found; skipping pre-training")
    except ValueError as e:
        print(f"[WARN] {e}; skipping pre-training")

    exp_data: TensorDataset | None = None
    exp_classes = 0
    if EXPERIMENTAL_TSV.exists():
        exp_data, exp_classes = _load_experimental_dataset(EXPERIMENTAL_TSV)
    else:
        print("[WARN] experimental TSV not found; skipping fine-tuning")

    n_classes = max(syn_classes, exp_classes, 1)
    cfg = ModelCfg(n_classes=n_classes)
    model = TFTransformer(cfg)
    model.train()

    if syn_data is not None:
        print("=== Pre-training on synthetic data ===")
        _train(model, syn_data, EPOCHS, LR, BATCH_SIZE)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), OUT_DIR / "IY010_pretrained.pt")

    if exp_data is not None:
        print("=== Fine-tuning on experimental data ===")
        model.freeze_encoder(True)
        model.reset_classifier()
        _train(model, exp_data, EPOCHS, LR, BATCH_SIZE)
        torch.save(model.state_dict(), OUT_DIR / "IY010_finetuned.pt")


if __name__ == "__main__":
    main()
