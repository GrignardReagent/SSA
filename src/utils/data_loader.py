import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    # Load dataset if given as CSV
    if mRNA_traj_file.endswith(".csv"):
        df_results = pd.read_csv(mRNA_traj_file)
        # Extract features and labels
        X = df_results.iloc[:, 1:].values
        y = df_results["label"].values
        
    # load from tsv
    elif mRNA_traj_file.endswith(".tsv"):
        df_results = pd.read_csv(mRNA_traj_file, sep="\t")
        # Extract features and labels
        X = df_results.iloc[:, 1:].values
        y = df_results["label"].values
        
    # load from npz
    elif mRNA_traj_file.endswith(".npz"):
        data = np.load(mRNA_traj_file)
        X = data['X']
        y = data['y']

    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .npz, or .tsv file.")

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