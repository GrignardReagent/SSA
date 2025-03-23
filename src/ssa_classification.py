import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_and_split_data(mRNA_traj_file, split_test_size=0.2, split_val_size=None, split_random_state=42):
    """
    Loads the mRNA trajectories dataset, extracts features and labels,
    and splits the data into training, test, and optionally validation sets.

    Parameters:
        mRNA_traj_file: Path to the mRNA trajectories dataset
        split_test_size: Fraction of the dataset for the test split (default: 0.2)
        split_val_size: Fraction of the training data for the validation split (default: None) - **Required for Deep Learning**
        split_random_state: Seed for reproducibility (default: 42)

    Returns:
        X_train, X_test, y_train, y_test [, X_val, y_val]: Split data
    """
    # Load dataset
    df_results = pd.read_csv(mRNA_traj_file)

    # Extract features and labels
    X = df_results.iloc[:, 1:].values
    y = df_results["label"].values

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_test_size, random_state=split_random_state, stratify=y
    )

    # Further split training data if validation size is specified
    if split_val_size:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=split_val_size,
            random_state=split_random_state, stratify=y_train
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test

def svm_classifier(X_train, X_test, y_train, y_test, svm_C=1.0, svm_gamma='scale', svm_kernel='rbf'):
    """
    Trains a basic SVM model for classification and evaluates the model.
    
    Parameters:
        X_train, X_test, y_train, y_test: Split data
        svm_C: Regularization parameter (default: 1.0)
        svm_gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale')
        svm_kernel: Specifies the kernel type to be used in the algorithm (default: 'rbf')
    
    Returns:
        accuracy: Classification accuracy of the SVM
    """
    # Train a basic SVM model without grid search
    svm_model = SVC(C=svm_C, gamma=svm_gamma, kernel=svm_kernel)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    if svm_kernel == 'rbf':
        print(f"=== SVM (RBF Kernel) Classification Accuracy: {svm_acc:.2f} ===")
    elif svm_kernel == 'linear':
        print(f"=== SVM (Linear Kernel) Classification Accuracy: {svm_acc:.2f} ===")
    return svm_acc

def random_classifier(y_test):
    """ 
    Makes random predictions for classification and evaluates the model.
    
    Parameters:
        y_test: True labels for the test set
    
    Returns:
        accuracy: Classification accuracy of the random classifier
    """
    # Random Classifier
    random_preds = np.random.choice([0, 1], size=len(y_test))  # Random guesses
    random_acc = accuracy_score(y_test, random_preds)
    print(f"=== Random Classifier Accuracy: {random_acc:.2f} ===")

    return random_acc

def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    """ 
    Trains a basic logistic regression model for classification and evaluates the model.
    
    Parameters:
        X_train, X_test, y_train, y_test: Split data
    
    Returns:
        accuracy: Classification accuracy of the logistic regression model
    """
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_preds = log_reg.predict(X_test)
    log_reg_acc = accuracy_score(y_test, log_reg_preds)
    print(f"=== Logistic Regression Accuracy: {log_reg_acc:.2f} ===")

    return log_reg_acc

def random_forest_classifier(X_train, X_test, y_train, y_test, rf_n_estimators=100, rf_max_depth=None):
    """ 
    Trains a basic random forest model for classification and evaluates the model.
    
    Parameters:
        X_train, X_test, y_train, y_test: Split data
        rf_n_estimators: The number of trees in the forest (default: 100)
        rf_max_depth: The maximum depth of the tree (default: None)
    
    Returns:
        accuracy: Classification accuracy of the random forest model
    """
    # Random Forest
    rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"=== Random Forest Accuracy: {rf_acc:.2f} ===")

    return rf_acc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.MLP import MLP

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