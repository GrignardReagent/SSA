#!/usr/bin/python

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.lstm import LSTMClassifier
from utils.set_seed import set_seed

def lstm_classifier(X_train, X_val, X_test, y_train, y_val, y_test, 
                    input_size=None, hidden_size=64, num_layers=2, 
                    output_size=None, dropout_rate=0.3, learning_rate=0.001, 
                    batch_size=64, epochs=50, patience=10, bidirectional=True, use_conv1d=False, use_attention=False, num_attention_heads=1, use_auxiliary=False):
    """ 
    Trains an LSTM model for classification and evaluates it. 
    This is a high-level wrapper for quick usage.
    
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

    # === Reshape for LSTM: (batch, seq_len, features) ===
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
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, output_size=output_size,
                           dropout_rate=dropout_rate,learning_rate=learning_rate,bidirectional=bidirectional,use_conv1d=use_conv1d, use_attention=use_attention, num_attention_heads=num_attention_heads, use_auxiliary=use_auxiliary)

    model.train_model(train_loader, val_loader, epochs=epochs, patience=patience)

    # === Evaluate ===
    lstm_acc = model.evaluate(test_loader)

    # print out vanilla LSTM accuracy
    if not use_conv1d and not use_attention:
        print(f"=== Vanilla Long-Short Term Memory (LSTM) Accuracy: {lstm_acc:.2f} ===")
    # LSTM with Conv1D accuracy
    elif use_conv1d and not use_attention:
        print(f"=== LSTM with Conv1D Accuracy: {lstm_acc:.2f} ===")
    # LSTM with Conv1D and Attention accuracy
    elif use_conv1d and use_attention and num_attention_heads == 1:
        print(f"=== LSTM with Conv1D and Attention Accuracy: {lstm_acc:.2f} ===")
    # LSTM with Multi-Head Attention accuracy
    elif use_conv1d and use_attention and num_attention_heads > 1:
        print(f"=== LSTM with Conv1D and {num_attention_heads}-Head Attention Accuracy: {lstm_acc:.2f} ===")
    # LSTM with auxiliary input accuracy
    elif use_auxiliary:
        print(f"=== LSTM with Auxiliary Input Accuracy: {lstm_acc:.2f} ===")
    return lstm_acc
