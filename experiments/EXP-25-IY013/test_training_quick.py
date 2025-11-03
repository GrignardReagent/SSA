#!/usr/bin/env python3
"""Quick training test with a few epochs to validate the full pipeline."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.TF_transformer import TFTransformer, ModelCfg

def quick_training_test():
    """Test training for just a few batches."""
    
    # Get first 5 CSV files for quick testing
    data_dir = Path(__file__).parent / "data"
    csv_files = sorted(list(data_dir.glob("*.csv")))[:5]
    
    print(f"🧪 Quick training test with {len(csv_files)} CSV files")
    
    # Load and combine data
    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source"] = csv_file.stem
        frames.append(df)
    
    combined_df = pd.concat(frames, ignore_index=True)
    
    # Simple binary labeling based on filename pattern
    combined_df["label"] = combined_df["source"].str.contains("0.5").astype(int)
    combined_df = combined_df.drop(columns=["source"])
    
    # Convert to tensors
    labels = torch.tensor(combined_df["label"].values, dtype=torch.long)
    series = torch.tensor(combined_df.drop(columns=["label"]).values, dtype=torch.float32)
    
    # Normalize the data (this is important to prevent NaN losses!)
    # Calculate lengths and mask
    lengths = (series != 0).sum(dim=1)
    max_len = series.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Per-sequence normalization
    mean = (series * mask).sum(dim=1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    var = ((series - mean).pow(2) * mask).sum(dim=1, keepdim=True) / lengths.clamp(min=1).unsqueeze(1)
    std = var.sqrt()
    series = (series - mean) / (std + 1e-8)
    series[~mask] = 0.0
    
    series = series.unsqueeze(-1)
    
    print(f"📊 Data: {series.shape}, Classes: {labels.unique().tolist()}")
    
    # Create dataset and split
    dataset = TensorDataset(series, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    cfg = ModelCfg(
        n_classes=2,
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_len=2100,
        verbose=True,
        learning_rate=1e-3,
        optimizer='AdamW'
    )
    
    model = TFTransformer(cfg)
    
    # Quick training test (just 2 epochs)
    print("\n🏃 Running quick training test...")
    
    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        patience=10,  # No early stopping
        save_path=str(Path(__file__).parent / "test_quick_model.pt")
    )
    
    print(f"✅ Training completed!")
    print(f"📈 Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"📈 Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"📈 Final val acc: {history['val_acc'][-1]:.4f}")
    print(f"� History keys: {list(history.keys())}")
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model.predict(batch_x)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5].tolist()}")
            print(f"Actual labels: {batch_y[:5].tolist()}")
            break
    
    print("🎉 Quick training test successful!")

if __name__ == "__main__":
    quick_training_test()
