import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tqdm
import scipy.stats as st
import itertools
from sympy import sqrt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def svm_classification(mRNA_traj_file, split_test_size=0.2, split_random_state=42, svm_C=1.0, svm_gamma='scale', svm_kernel='rbf'):
    """
    1) Loads the mRNA trajectories dataset, extracts features and labels; 
    2) Splits the data into training and test sets;
    3) Trains a basic SVM model for classification;
    4) Evaluates the model, prints and returns the classification accuracy.
    
    Parameters:
        mRNA_traj_file: Path to the mRNA trajectories dataset
        split_test_size: Fraction of the dataset to include in the test split (default: 0.2)
        split_random_state: Seed for the random number generator (default: 42)
        svm_C: Regularization parameter (default: 1.0)
        svm_gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale')
        svm_kernel: Specifies the kernel type to be used in the algorithm (default: 'rbf')
    
    Returns:
        accuracy: Classification accuracy of the SVM
    """

    # Load the mRNA trajectories dataset
    df_results = pd.read_csv(mRNA_traj_file)

    # Extract features (mRNA trajectories) and labels
    X = df_results.iloc[:, 1:].values  # All time series data
    y = df_results["label"].values  # Labels: 0 (Stressed Condition) or 1 (Normal Condition)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=split_random_state)

    # Train a basic SVM model without grid search
    svm_model = SVC(C=svm_C, gamma=svm_gamma, kernel=svm_kernel)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"=== SVM Classification Accuracy: {accuracy:.2f} ===")

    return accuracy

def pca_plot(mRNA_traj_file):
    """
    Load the mRNA trajectories dataset and perform PCA for visualization.
    
    Parameters:
        mRNA_traj_file: Path to the mRNA trajectories dataset
    """
    # Load the mRNA trajectories dataset
    df_results = pd.read_csv(mRNA_traj_file)

    # Extract features (mRNA trajectories) and labels
    X = df_results.iloc[:, 1:].values  # All time series data
    y = df_results["label"].values  # Labels: 0 (Stressed Condition) or 1 (Normal Condition)

    # Scatter plot of two PCA components for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='Stressed Condition', alpha=0.5)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='green', label='Normal Condition', alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of mRNA Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()
