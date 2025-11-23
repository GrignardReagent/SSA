import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_processing import build_groups

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

# -----------------------------------------------------------------------------
# Siamese / Contrastive Learning Data Pipeline
# -----------------------------------------------------------------------------

class SiameseGroupDataset(Dataset):
    """
    Dataset wrapper that:
    1. Receives (seq_len, 2) pair groups.
    2. Performs Random Cropping (Training) or Center Cropping (Validation).
    3. Splits into (x1, x2, y).
    """
    def __init__(self, groups, crop_len=200, training=True):
        self.groups = groups
        self.crop_len = crop_len
        self.training = training

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # X has shape (Original_Seq_Len, 2)
        X, y = self.groups[idx]
        seq_len = X.shape[0]
        
        # --- Cropping Logic ---
        if seq_len > self.crop_len:
            if self.training:
                # Random crop for training (Data Augmentation)
                start = np.random.randint(0, seq_len - self.crop_len)
            else:
                # Center crop for deterministic validation
                start = (seq_len - self.crop_len) // 2
            
            X_crop = X[start : start + self.crop_len, :]
        else:
            # Pad if sequence is shorter than crop window
            pad_len = self.crop_len - seq_len
            # Pad with zeros at the end
            X_crop = np.pad(X, ((0, pad_len), (0, 0)), mode='constant')

        # --- Split into x1, x2, y ---
        return (
            torch.tensor(X_crop[:, 0:1], dtype=torch.float32), # x1
            torch.tensor(X_crop[:, 1:2], dtype=torch.float32), # x2
            torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # y (target)
        )

def siamese_data_prep(
    all_file_paths, 
    batch_size=64, 
    num_groups_train=20000, 
    num_groups_val=2000, 
    num_groups_test=2000, 
    crop_len=200, 
    seed=42
):
    """
    Full pipeline for Siamese preparation with Log-Scaling and File-based splitting.
    """
    # 1. Split FILES (Simulations) to prevent leakage
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # 2. Build Groups (Pairs)
    #    We generate MANY pairs (e.g. 20,000) because random cropping makes them distinct.
    print(f"Generating {num_groups_train} training pairs...")
    train_groups = build_groups(train_files, num_groups=num_groups_train, num_traj=2, seed=seed)
    
    print(f"Generating validation/test pairs...")
    val_groups   = build_groups(val_files,   num_groups=num_groups_val,   num_traj=2, seed=seed)
    test_groups  = build_groups(test_files,  num_groups=num_groups_test,  num_traj=2, seed=seed)

    # 3. Global Log-Scaling (important for training convergence)
    # We apply Log1p FIRST to handle the massive range (mu=1 vs mu=10,000), 
    # THEN fit the StandardScaler.
    scaler = StandardScaler()
    
    # Helper to collect all training values for fitting
    def get_log_values(group_list):
        # Extract X, apply log1p, and stack
        all_X = [np.log1p(X) for X, y in group_list] 
        return np.vstack(all_X)

    if len(train_groups) > 0:
        print("Fitting scaler on Log-Transformed training data...")
        # Stack all training data to find global Mean/Std
        train_stack = get_log_values(train_groups)
        scaler.fit(train_stack.reshape(-1, 1))
    
    def transform_groups(group_list):
        transformed = []
        for X, y in group_list:
            # A. Log Transform
            X_log = np.log1p(X)
            
            # B. Standard Scale
            shape = X_log.shape
            # Flatten -> Scale -> Reshape
            X_sc = scaler.transform(X_log.reshape(-1, 1)).reshape(shape)
            
            transformed.append((X_sc, y))
        return transformed

    # Apply transformations
    print("Applying Log-Scaling to all groups...")
    train_groups = transform_groups(train_groups)
    val_groups   = transform_groups(val_groups)
    test_groups  = transform_groups(test_groups)

    # 4. Create Datasets with Random Cropping
    train_ds = SiameseGroupDataset(train_groups, crop_len=crop_len, training=True)
    val_ds   = SiameseGroupDataset(val_groups,   crop_len=crop_len, training=False)
    test_ds  = SiameseGroupDataset(test_groups,  crop_len=crop_len, training=False)

    # 5. Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader