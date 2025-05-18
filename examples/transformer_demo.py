#!/usr/bin/python

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import the classifier from the project
from models.transformer import TransformerClassifier
from classifiers.transformer_classifier import transformer_classifier

def generate_synthetic_data(n_samples=1000, seq_len=50, n_features=5, n_classes=3):
    """Generate synthetic time series data for demonstration"""
    # Generate random features
    X = np.random.randn(n_samples, seq_len, n_features)
    
    # Generate class labels based on sequence patterns
    y = np.zeros(n_samples, dtype=np.int64)
    
    # Class 0: Sequences with upward trend
    mask_0 = np.random.choice(n_samples, n_samples // n_classes, replace=False)
    for i in mask_0:
        X[i, :, 0] = np.linspace(-1, 1, seq_len) + 0.1 * np.random.randn(seq_len)
        y[i] = 0
        
    # Class 1: Sequences with downward trend
    mask_1 = np.setdiff1d(np.arange(n_samples), mask_0)
    mask_1 = np.random.choice(mask_1, n_samples // n_classes, replace=False)
    for i in mask_1:
        X[i, :, 0] = np.linspace(1, -1, seq_len) + 0.1 * np.random.randn(seq_len)
        y[i] = 1
    
    # Class 2: Sequences with sinusoidal pattern
    mask_2 = np.setdiff1d(np.arange(n_samples), np.concatenate([mask_0, mask_1]))
    for i in mask_2:
        X[i, :, 0] = np.sin(np.linspace(0, 3*np.pi, seq_len)) + 0.1 * np.random.randn(seq_len)
        y[i] = 2
    
    return X, y

def demo_transformer_classifier():
    """Demonstrate the TransformerClassifier model with various configurations"""
    print("🚀 DEMONSTRATING TRANSFORMER CLASSIFIER WITH IMPROVED ARCHITECTURE")
    print("=" * 80)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=1200, seq_len=50, n_features=5, n_classes=3)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Standardize data
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_val_2d = scaler.transform(X_val_2d)
    X_test_2d = scaler.transform(X_test_2d)
    
    X_train = X_train_2d.reshape(X_train.shape)
    X_val = X_val_2d.reshape(X_val.shape)
    X_test = X_test_2d.reshape(X_test.shape)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)
    
    # Demo 1: Basic transformer with default settings
    print("\n🔍 Demo 1: Basic transformer with default settings")
    model1 = TransformerClassifier(
        input_size=5,  # Number of features
        d_model=64,    # Embedding dimension
        nhead=4,       # Number of attention heads
        num_layers=2,  # Number of transformer layers
        output_size=3, # Number of classes
    )
    model1.train_model(train_loader, val_loader, epochs=10, patience=5)
    test_acc1 = model1.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc1:.4f}")
    
    # Demo 2: Transformer with Conv1D preprocessing
    print("\n🔍 Demo 2: Transformer with Conv1D preprocessing")
    model2 = TransformerClassifier(
        input_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_size=3,
        use_conv1d=True,  # Enable Conv1D preprocessing
    )
    model2.train_model(train_loader, val_loader, epochs=10, patience=5)
    test_acc2 = model2.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc2:.4f}")
    
    # Demo 3: Transformer with mean pooling and gradient clipping
    print("\n🔍 Demo 3: Transformer with mean pooling and gradient clipping")
    model3 = TransformerClassifier(
        input_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_size=3,
        pooling_strategy='mean',  # Use mean pooling instead of last token
        gradient_clip=0.5,        # Apply gradient clipping
    )
    model3.train_model(train_loader, val_loader, epochs=10, patience=5)
    test_acc3 = model3.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc3:.4f}")
    
    # Demo 4: Transformer with learnable pooling
    print("\n🔍 Demo 4: Transformer with learnable pooling")
    model4 = TransformerClassifier(
        input_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_size=3,
        pooling_strategy='learnable',  # Use learnable weighted pooling
    )
    model4.train_model(train_loader, val_loader, epochs=10, patience=5)
    test_acc4 = model4.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc4:.4f}")
    
    # Demo 5: Full feature set with norm_first transformer and auxiliary task
    print("\n🔍 Demo 5: Full feature set with norm_first transformer and auxiliary task")
    model5 = TransformerClassifier(
        input_size=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        output_size=3,
        dropout_rate=0.3,
        optimizer='AdamW',
        use_conv1d=True,
        use_auxiliary=True,
        pooling_strategy='mean',
        gradient_clip=1.0,
    )
    model5.train_model(train_loader, val_loader, epochs=15, patience=5)
    test_acc5 = model5.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc5:.4f}")
    
    # Compare results
    print("\n📊 RESULTS COMPARISON")
    print("=" * 50)
    print(f"Basic Transformer: {test_acc1:.4f}")
    print(f"With Conv1D: {test_acc2:.4f}")
    print(f"With Mean Pooling: {test_acc3:.4f}")
    print(f"With Learnable Pooling: {test_acc4:.4f}")
    print(f"Advanced Configuration: {test_acc5:.4f}")
    
    # Use the high-level wrapper function
    print("\n🔍 Using the high-level wrapper function")
    test_acc = transformer_classifier(
        X_train, X_val, X_test, y_train, y_val, y_test,
        input_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_size=3,
        epochs=10,
        use_conv1d=True,
        pooling_strategy='mean',
        gradient_clip=0.5
    )
    print(f"Test Accuracy from wrapper: {test_acc:.4f}")

if __name__ == "__main__":
    demo_transformer_classifier()
