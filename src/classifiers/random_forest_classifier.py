#!/usr/bin/python

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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