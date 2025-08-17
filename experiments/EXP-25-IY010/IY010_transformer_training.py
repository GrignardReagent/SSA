#!/usr/bin/env python3
"""Train and fine-tune the TFTransformer on synthetic then experimental
data.

This mirrors the intended workflow for CVmCherry trajectory classification:

1. **Pre‑train** on a large synthetic dataset produced by the telegraph model
   (see ``IY010.py``) to learn generic representations.
2. **Fine‑tune** on a smaller experimental dataset by freezing the encoder and
   re‑initialising the classifier head.

Both datasets are expected as CSV files with one trajectory per row, the first
column named ``label`` and the remaining columns containing the time series.
Trailing zeros are interpreted as padding and ignored via a key padding mask.

The file locations are hard‑coded below to match the output of
``IY010_simulation.py``.  Adjust the paths to point at the synthetic
telegraph‑model trajectories and the experimental recordings before running the
script.  If a dataset is absent the corresponding phase is skipped and a
warning is issued.  Models are saved alongside this script as
``IY010_pretrained.pt`` and ``IY010_finetuned.pt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.simple_transformer import TFTransformer, ModelCfg


# ---------------------------------------------------------------------------
# File locations and basic training hyper-parameters.  The synthetic dataset
# path mirrors the output location of ``IY010_simulation.py`` whereas the
# experimental dataset is a placeholder for unseen trajectories.  Adjust as
# necessary for your environment.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SYNTHETIC_CSV = BASE_DIR / "data" / "synthetic_trajectories.csv"
EXPERIMENTAL_CSV = BASE_DIR / "data" / "experimental_trajectories.csv"
OUT_DIR = BASE_DIR
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3


def _load_csv(path: Path) -> TensorDataset:
    """Return trajectories and labels stored in ``path``.

    The resulting dataset yields tuples ``(x, lengths, y)``.  ``x`` has shape
    ``[T, 1]`` and ``lengths`` counts the non-zero entries in each sequence,
    allowing the Transformer to mask padded positions.
    """

    df = pd.read_csv(path)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    series = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)

    # Trailing zeros denote padding; ``lengths`` counts the valid entries for
    # each trajectory.  EDA of the synthetic set revealed large differences in
    # mean expression between classes, so we normalise each sequence to zero
    # mean and unit variance before training.
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
    cfg = ModelCfg(n_classes=2)
    model = TFTransformer(cfg)
    model.train()

    if SYNTHETIC_CSV.exists():
        print("=== Pre-training on synthetic data ===")
        syn_data = _load_csv(SYNTHETIC_CSV)
        _train(model, syn_data, EPOCHS, LR, BATCH_SIZE)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), OUT_DIR / "IY010_pretrained.pt")
    else:
        print("[WARN] synthetic CSV not found; skipping pre-training")

    if EXPERIMENTAL_CSV.exists():
        print("=== Fine-tuning on experimental data ===")
        model.freeze_encoder(True)
        model.reset_classifier()
        exp_data = _load_csv(EXPERIMENTAL_CSV)
        _train(model, exp_data, EPOCHS, LR, BATCH_SIZE)
        torch.save(model.state_dict(), OUT_DIR / "IY010_finetuned.pt")
    else:
        print("[WARN] experimental CSV not found; skipping fine-tuning")


if __name__ == "__main__":
    main()
