#!/usr/bin/env python3
"""Quick test of the v2 training script with a small subset of data."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.TF_transformer import TFTransformer, ModelCfg

def test_small_dataset():
    """Test with just a few CSV files."""
    
    # Get first 5 CSV files for quick testing
    data_dir = Path(__file__).parent / "data"
    csv_files = sorted(list(data_dir.glob("*.csv")))[:5]
    
    print(f"Testing with {len(csv_files)} CSV files")
    
    # Load and combine data
    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source"] = csv_file.stem
        frames.append(df)
    
    combined_df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined_df)} trajectories")
    
    # Check sequence length
    seq_len = combined_df.shape[1] - 1  # Exclude source column
    print(f"Sequence length: {seq_len}")
    
    # Simple binary labeling for testing
    combined_df["label"] = 0  # Just assign all to class 0 for testing
    combined_df = combined_df.drop(columns=["source"])
    
    # Convert to tensors
    labels = torch.tensor(combined_df["label"].values, dtype=torch.long)
    series = torch.tensor(combined_df.drop(columns=["label"]).values, dtype=torch.float32)
    
    # Add feature dimension [B, T, 1]
    series = series.unsqueeze(-1)
    
    print(f"Tensor shapes: series={series.shape}, labels={labels.shape}")
    
    # Create dataset and loader
    dataset = TensorDataset(series, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Test model creation
    cfg = ModelCfg(
        n_classes=2,
        d_model=32,  # Smaller for testing
        n_heads=2,
        n_layers=1,
        max_len=seq_len + 100,  # A bit larger than actual sequence length
        verbose=True
    )
    
    model = TFTransformer(cfg)
    print(f"Model created successfully")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            print(f"Testing batch: {batch_x.shape} -> {batch_y.shape}")
            logits = model(batch_x)
            print(f"Forward pass successful: {logits.shape}")
            break  # Just test first batch
    
    print("✅ Basic functionality test passed!")

if __name__ == "__main__":
    test_small_dataset()
