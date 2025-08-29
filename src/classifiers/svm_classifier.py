#!/usr/bin/python

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
import pandas as pd
from typing import Dict, Optional


def svm_classifier(X_train, X_test, y_train, y_test, 
                   svm_C=1.0, 
                   svm_gamma='scale', 
                   svm_kernel='rbf',
                   print_classification_report=False,
                   print_confusion_matrix=False,
                   ):
    """
    Trains a basic SVM model for classification and evaluates the model.
    
    Parameters:
        X_train, X_test, y_train, y_test: Split data
        svm_C: Regularization parameter (default: 1.0)
        svm_gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale') 
        svm_kernel: Specifies the kernel type to be used in the algorithm (default: 'rbf')
    
    Returns:
        accuracy: Classification accuracy of the SVM
    
    Example:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        >>> accuracy = svm_classifier(X_train, X_test, y_train, y_test, svm_C=1.0, svm_kernel='rbf')
        === SVM (RBF Kernel) Classification Accuracy: 0.87 ===
        >>> print(f"Accuracy: {accuracy}")
        Accuracy: 0.87
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

    if print_classification_report:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    if print_confusion_matrix:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    return svm_acc


def grid_search_svm(
    df: pd.DataFrame,
    param_grid: Optional[Dict[str, list[float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Grid search SVM hyperparameters for RBF and linear kernels."""
   
    # Extract labels
    y = df["label"].values

    # Extract features (all columns except 'label')
    X = df.drop(columns=["label"]).values
    
    # Split data into training and testing sets with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define default hyperparameter grid if none provided
    # C: regularization parameter (controls overfitting)
    # gamma: kernel coefficient for RBF kernel (controls decision boundary complexity)
    param_grid = param_grid or {
        "C": [0.1, 1, 10],
        "gamma": [0.01, 0.1, 1],
    }
    
    results = {}
    
    # Test both RBF (radial basis function) and linear kernels
    for kernel in ("rbf",
                #    "linear"
                   ):
        # Perform 5-fold cross-validation grid search to find optimal hyperparameters
        search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=5)
        
        # Fit the grid search on training data
        search.fit(X_train, y_train)
        
        # Make predictions on test set using the best found parameters
        predictions = search.best_estimator_.predict(X_test)
        
        # Store results including accuracy and best hyperparameters
        results[kernel] = {
            "accuracy": accuracy_score(y_test, predictions),
            "C": search.best_params_["C"],
            "gamma": search.best_params_["gamma"],
        }
    return results