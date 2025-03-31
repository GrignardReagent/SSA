#!/usr/bin/python

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.MLP import MLP
from utils.set_seed import set_seed

def mlp_classifier(X_train, X_val, X_test, y_train, y_val, y_test, input_size=None, hidden_size:list=[300, 200], output_size=None, dropout_rate=0.3, learning_rate=0.001, batch_size=32, epochs=10):
    """ 
    Trains a basic MLP model for classification and evaluates the model. This is a high level function and doesn't allow you to change the model architecture. To modify model architecture, you need to modify the MLP class in models/MLP.py.
    
    Parameters:
        X_train, X_test, y_train, y_test: Split data
        input_size: The number of expected features in the input
        hidden_size: A list of integers specifying the number of neurons in each hidden layer
        output_size: The number of output features
        dropout_rate: The dropout rate (default: 0.3)
        learning_rate: The learning rate (default: 0.001)
        batch_size: The batch size for training (default: 32)
        epochs: The number of epochs to train the model (default: 10)
    Returns:
        accuracy: Classification accuracy of the MLP model
    """

    set_seed(42)  # Set random seed for reproducibility
    # Define model parameters
    input_size = X_train.shape[1]
    output_size = len(set(y_train))  # Number of classes

    # Standardize the data 
    # If your input features are too large (e.g., >1000) or too small (<0.0001), it can cause unstable training, so it's better to standardize the data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # prepare the data in terms of tensors and dataloaders so torch can use it
    train_loader = DataLoader(TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)),
    batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)),
    batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Train MLP model
    model = MLP(input_size, hidden_size, output_size, dropout_rate, learning_rate)
    model.train_model(train_loader, val_loader, epochs=epochs, patience=10)

    # Evaluate MLP model
    mlp_acc = model.evaluate(test_loader)
    print(f"=== Multilayer Perceptron (MLP) Accuracy: {mlp_acc:.2f} ===")

    return mlp_acc