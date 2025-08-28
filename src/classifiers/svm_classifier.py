#!/usr/bin/python

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
    return svm_acc