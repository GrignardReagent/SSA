#!/usr/bin/python

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.transformer import TransformerClassifier
from utils.set_seed import set_seed

def transformer_classifier(X_train, X_val, X_test, y_train, y_val, y_test, 
                          input_size=None, d_model=64, nhead=4, num_layers=2,
                          output_size=None, dropout_rate=0.2, learning_rate=0.001, 
                          batch_size=64, epochs=50, patience=10, optimizer='Adam',
                          use_conv1d=False, use_auxiliary=False, aux_weight=0.1, save_path=None):
    """ 
    Trains a Transformer model for classification and evaluates it. 
    This is a high-level wrapper for quick usage.
    
    Args:
        X_train, X_val, X_test: Training, validation and test input data
        y_train, y_val, y_test: Training, validation and test labels
        input_size: Number of features per time step (default: 1)
        d_model: Embedding dimension for transformer (default: 64)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        output_size: Number of classes (default: inferred from y_train)
        dropout_rate: Dropout probability (default: 0.2)
        learning_rate: Learning rate (default: 0.001)
        batch_size: Batch size for training (default: 64)
        epochs: Maximum number of training epochs (default: 50)
        patience: Early stopping patience (default: 10)
        optimizer: Optimizer type - 'Adam', 'SGD', or 'AdamW' (default: 'Adam')
        use_conv1d: Whether to use Conv1D preprocessing (default: False)
        use_auxiliary: Whether to use auxiliary task (default: False)
        aux_weight: Weight for auxiliary loss (default: 0.1)
        save_path: Path to save the best model (default: None)
        
    Returns:
        test_acc: Final test accuracy
    """

    set_seed(42)  # reproducibility

    # Set input/output sizes if not given
    if input_size is None:
        input_size = 1  # each time step is a single value
    if output_size is None:
        output_size = len(set(y_train))

    # === Standardize ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # === Reshape for Transformer: (batch, seq_len, features) ===
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], input_size))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], input_size))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], input_size))

    # === Convert to tensors and loaders ===
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # === Initialize and train model ===
    model = TransformerClassifier(
        input_size=input_size, 
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers, 
        output_size=output_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer=optimizer,
        use_conv1d=use_conv1d,
        use_auxiliary=use_auxiliary,
        aux_weight=aux_weight
    )

    model.train_model(train_loader, val_loader, epochs=epochs, patience=patience, save_path=save_path)

    # === Evaluate ===
    transformer_acc = model.evaluate(test_loader)

    # Print out results based on configuration
    if not use_conv1d and not use_auxiliary:
        print(f"=== Vanilla Transformer Accuracy: {transformer_acc:.2f} ===")
    elif use_conv1d and not use_auxiliary:
        print(f"=== Transformer with Conv1D Preprocessing Accuracy: {transformer_acc:.2f} ===")
    elif not use_conv1d and use_auxiliary:
        print(f"=== Transformer with Auxiliary Task Accuracy: {transformer_acc:.2f} ===")
    elif use_conv1d and use_auxiliary:
        print(f"=== Transformer with Conv1D and Auxiliary Task Accuracy: {transformer_acc:.2f} ===")

    return transformer_acc
