#!/usr/bin/env python3
"""Train and fine-tune the TFTransformer on synthetic then experimental data.

This script demonstrates the intended workflow for TF trajectory classification:

1. **Pre‑train** on a large synthetic dataset produced by the telegraph model
   (see ``IY010_simulation.py``) to learn generic representations.
2. **Fine‑tune** on a smaller experimental dataset by freezing the encoder and
   re‑initialising the classifier head.

The new version leverages the built-in training infrastructure of TFTransformer
for cleaner, more robust code with automatic early stopping, validation, and
comprehensive logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import re
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.TF_transformer import TFTransformer, ModelCfg


# ---------------------------------------------------------------------------
# Configuration - Adjust these paths and hyperparameters as needed
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SYNTHETIC_DIR = BASE_DIR / "data"
EXPERIMENTAL_TSV = (
    BASE_DIR
    / "exp_data"
    / "19316_2020_10_26_steadystate_glucose_144m_2w2_00_post_media_switch.tsv"
)
OUT_DIR = BASE_DIR

# Training hyperparameters
PRETRAINING_EPOCHS = 50
FINETUNING_EPOCHS = 10
BATCH_SIZE = 64
PRETRAINING_LR = 0.01
FINETUNING_LR = 1e-4  # Lower LR for fine-tuning
PATIENCE = 5  # Early stopping patience
VAL_SPLIT = 0.2  # Fraction of data to use for validation
TRAINING_SEQ_SIZE = 50 # The size of dataset (number of time series to fed the transformer as training data, for each)

#TODO: Use nearest neighbour as the labelling metric? 
def _add_pair_labels(df: pd.DataFrame) -> None:
    """Append a ``label`` column to trajectories lacking one.
    
    Files are paired by CV values: lower CV → label 0, higher CV → label 1
    """
    pattern = re.compile(r"mRNA_trajectories_([0-9.]+)_([0-9.]+)_([0-9.]+)")
    groups: dict[tuple[float, float], list[tuple[float, str]]] = defaultdict(list)
    
    for src in df["source"].unique():
        match = pattern.match(src)
        if not match:
            raise ValueError(f"unrecognised filename pattern: {src}")
        mu, cv, t_ac = map(float, match.groups())
        groups[(mu, t_ac)].append((cv, src))

    label_map: dict[str, int] = {}
    for key, items in groups.items():
        items.sort(key=lambda x: x[0])  # Sort by CV
        
        for i in range(0, len(items), 2):
            if i + 1 >= len(items):
                _, src_last = items[i]
                label_map[src_last] = 0
                continue
            _, src0 = items[i]      # Lower CV
            _, src1 = items[i + 1]  # Higher CV
            label_map[src0] = 0
            label_map[src1] = 1

    if len(label_map) != df["source"].nunique():
        raise ValueError("failed to assign labels to all synthetic CSV files")

    df["label"] = df["source"].map(label_map)


def _prepare_dataset(df: pd.DataFrame) -> TensorDataset:
    """Convert DataFrame to TensorDataset with normalization and padding handling."""
    
    if "label" not in df.columns:
        if "source" not in df.columns:
            raise ValueError("cannot infer labels without 'source' column")
        _add_pair_labels(df)
    
    # drop source column once no longer needed
    if "source" in df.columns:
        df.drop(columns=["source"], inplace=True)

    labels = torch.tensor(df["label"].values, dtype=torch.long)
    series = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)

    # Handle padding and normalization
    lengths = (series != 0).sum(dim=1)
    max_len = series.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Normalize per sequence
    mean = (series * mask).sum(dim=1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    var = ((series - mean).pow(2) * mask).sum(dim=1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    std = var.sqrt()
    series = (series - mean) / (std + 1e-8)
    series[~mask] = 0.0
    
    # Add feature dimension for transformer input [B, T, 1]
    series = series.unsqueeze(-1)
    
    # Return only series and labels (TF_transformer expects 2 values)
    return TensorDataset(series, labels)


def _create_data_loaders(dataset: TensorDataset, batch_size: int, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Split dataset and create train/validation DataLoaders."""
    
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def _load_synthetic_dataset(path: Path) -> Tuple[TensorDataset, int]:
    """Load and concatenate all synthetic CSV files."""
    
    csv_files = sorted(p for p in path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no synthetic CSV files found in {path}")

    print(f"📊 Loading {len(csv_files)} synthetic CSV files...")
    frames = []
    for p in csv_files:
        df = pd.read_csv(p)
        df["source"] = p.stem
        frames.append(df)
    
    df = pd.concat(frames, ignore_index=True)
    print(f"📈 Loaded {len(df)} trajectories from synthetic data")
    
    dataset = _prepare_dataset(df)
    return dataset, df["label"].nunique()


def _load_experimental_dataset(path: Path) -> Tuple[TensorDataset, int]:
    """Load experimental TSV data and convert to trajectories."""
    
    print(f"📊 Loading experimental data from {path.name}...")
    df = pd.read_csv(path, sep="\t", usecols=["id", "group", "time", "CV_mCherry"])
    df = df.pivot_table(index=["group", "id"], columns="time", values="CV_mCherry")
    df = df.sort_index(axis=1).reset_index()
    
    labels = df["group"]
    label_map = {g: i for i, g in enumerate(sorted(labels.unique()))}
    df["label"] = labels.map(label_map)
    df = df.drop(columns=["group", "id"])
    df = df.fillna(0)
    df.insert(0, "label", df.pop("label"))
    
    print(f"📈 Loaded {len(df)} trajectories from experimental data")
    print(f"🏷️  Classes: {list(label_map.keys())} → {list(label_map.values())}")
    
    return _prepare_dataset(df), len(label_map)


def run_pretraining(model: TFTransformer, syn_data: TensorDataset) -> None:
    """Run pretraining phase using built-in training infrastructure."""
    
    print("\n🎭 === PRE-TRAINING ON SYNTHETIC DATA ===")
    
    # Create train/validation split
    train_loader, val_loader = _create_data_loaders(syn_data, BATCH_SIZE, VAL_SPLIT)
    
    print(f"📊 Training set: {len(train_loader.dataset)} samples")
    print(f"📊 Validation set: {len(val_loader.dataset)} samples")
    
    # Update model config for pretraining
    model.cfg.learning_rate = PRETRAINING_LR
    model._setup_training()  # Reinitialize optimizer with new LR
    
    # Train the model
    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=PRETRAINING_EPOCHS,
        patience=PATIENCE,
        save_path=str(OUT_DIR / "IY010_pretrained.pt")
    )
    
    print(f"✅ Pretraining complete! Best validation accuracy: {max(history['val_acc']):.4f}")
    return history


def run_finetuning(model: TFTransformer, exp_data: TensorDataset) -> None:
    """Run fine-tuning phase with frozen encoder."""
    
    print("\n🔧 === FINE-TUNING ON EXPERIMENTAL DATA ===")
    
    # Prepare for fine-tuning
    model.freeze_encoder(freeze=True)
    model.reset_classifier()
    
    # Create train/validation split for experimental data
    train_loader, val_loader = _create_data_loaders(exp_data, BATCH_SIZE, VAL_SPLIT)
    
    print(f"📊 Training set: {len(train_loader.dataset)} samples")
    print(f"📊 Validation set: {len(val_loader.dataset)} samples")
    
    # Update learning rate for fine-tuning (typically lower)
    model.cfg.learning_rate = FINETUNING_LR
    model._setup_training()  # Reinitialize optimizer with new LR
    
    # Fine-tune the model
    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=FINETUNING_EPOCHS,
        patience=PATIENCE,
        save_path=str(OUT_DIR / "IY010_finetuned.pt")
    )
    
    print(f"✅ Fine-tuning complete! Best validation accuracy: {max(history['val_acc']):.4f}")
    return history


def main() -> None:
    """Main training pipeline."""
    
    print("🚀 TF Transformer Training Pipeline")
    print("=" * 50)
    
    # Load datasets
    syn_data: TensorDataset | None = None
    syn_classes = 0
    try:
        syn_data, syn_classes = _load_synthetic_dataset(SYNTHETIC_DIR)
    except FileNotFoundError:
        print("⚠️  [WARN] Synthetic CSV not found; skipping pre-training")
    except ValueError as e:
        print(f"⚠️  [WARN] {e}; skipping pre-training")

    exp_data: TensorDataset | None = None
    exp_classes = 0
    if EXPERIMENTAL_TSV.exists():
        try:
            exp_data, exp_classes = _load_experimental_dataset(EXPERIMENTAL_TSV)
        except Exception as e:
            print(f"⚠️  [WARN] Failed to load experimental data: {e}")
    else:
        print("⚠️  [WARN] Experimental TSV not found; skipping fine-tuning")

    # Determine number of classes
    n_classes = max(syn_classes, exp_classes, 2)  # Default to binary classification
    
    # Create model with optimal configuration
    cfg = ModelCfg(
        n_classes=n_classes,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        max_len=2048,  # Increased to handle 2000-length sequences
        verbose=True,
        learning_rate=PRETRAINING_LR,
        optimizer='AdamW',
        gradient_clip=1.0,
        label_smoothing=0.0  # Disable label smoothing to debug NaN loss
    )
    
    model = TFTransformer(cfg)
    print(f"🏗️  Model created with {n_classes} classes")
    
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Pretraining (if synthetic data available)
    pretraining_history = None
    if syn_data is not None:
        pretraining_history = run_pretraining(model, syn_data)
    
    # Phase 2: Fine-tuning (if experimental data available)
    finetuning_history = None
    if exp_data is not None:
        finetuning_history = run_finetuning(model, exp_data)
    
    # Summary
    print("\n📋 === TRAINING SUMMARY ===")
    if pretraining_history:
        print(f"🎭 Pretraining: {len(pretraining_history['train_loss'])} epochs")
        print(f"   Best training accuracy: {max(pretraining_history['train_acc']):.4f}")
        print(f"   Best validation accuracy: {max(pretraining_history['val_acc']):.4f}")
    
    if finetuning_history:
        print(f"🔧 Fine-tuning: {len(finetuning_history['train_loss'])} epochs")
        print(f"   Best training accuracy: {max(finetuning_history['train_acc']):.4f}")
        print(f"   Best validation accuracy: {max(finetuning_history['val_acc']):.4f}")
    
    print(f"\n💾 Models saved in: {OUT_DIR}")
    print("🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
