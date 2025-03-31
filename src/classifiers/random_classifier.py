#!/usr/bin/python

import numpy as np
from sklearn.metrics import accuracy_score

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