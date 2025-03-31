#!/usr/bin/python

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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