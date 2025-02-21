import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_and_split_data(mRNA_traj_file, split_test_size=0.2, split_random_state=42):
    """
    Loads the mRNA trajectories dataset, extracts features and labels, and splits the data into training and test sets.
    
    Parameters:
        mRNA_traj_file: Path to the mRNA trajectories dataset
        split_test_size: Fraction of the dataset to include in the test split (default: 0.2)
        split_random_state: Seed for the random number generator (default: 42)
    
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    # Load the mRNA trajectories dataset
    df_results = pd.read_csv(mRNA_traj_file)

    # Extract features (mRNA trajectories) and labels
    X = df_results.iloc[:, 1:].values  # All time series data
    y = df_results["label"].values  # Labels: 0 (Stressed Condition) or 1 (Normal Condition)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=split_random_state)
    
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"=== SVM Classification Accuracy: {accuracy:.2f} ===")

    return accuracy

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
