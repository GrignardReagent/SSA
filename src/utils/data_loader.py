from pathlib import Path
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
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
# Baseline Data Pipeline
# -----------------------------------------------------------------------------

def _load_and_concat(files, num_traj, preprocess_fn, rng, separator_len=1, separator_val=-10.0):
    """
    Helper to load files and concatenate 'num_traj' trajectories.
    """
    pool = []
    for path in files:
        try:
            data = np.load(path, allow_pickle=True)
            if "trajectories" in data:
                pool.extend(list(data["trajectories"]))
        except:
            pass
            
    if len(pool) < num_traj:
        # Fallback: replicate if not enough data
        if len(pool) == 0: return None # Should catch downstream
        while len(pool) < num_traj:
            pool += pool
            
    # Sample without replacement if possible
    selected_trajs = rng.sample(pool, num_traj)

    # === CONCATENATE ALONG TIME DIMENSION ===
    # Preprocess -> Stack
    proc = []
    for traj in selected_trajs:
        # Reshape
        t = preprocess_fn(traj) # (Time, 1)
        proc.append(t)

    # Crop to same length before stacking (optional, keeps segments regular)
    L = min(p.shape[0] for p in proc)
    proc = [p[:L] for p in proc]
    
    # Insert separator if requested
    if separator_len > 0:
        # Create separator of shape (sep_len, num_features)
        # Assuming channel dim is 1 based on default preprocess_fn
        n_features = proc[0].shape[1]
        separator = np.full((separator_len, n_features), separator_val, dtype=proc[0].dtype)
        
        proc_with_separators = []
        for i, p in enumerate(proc):
            proc_with_separators.append(p)
            # add separator except after last
            proc_with_separators.append(separator) if i < len(proc) - 1 else None
        return np.concatenate(proc_with_separators, axis=0)
        
    # Concatenate along TIME (Axis 0)
    # Result: (num_traj * L, 1)
    return np.concatenate(proc, axis=0)

def _get_params_from_filename(filepath):
    """
    Extracts parameters (mu, cv, t_ac) from the filename.
    Expected filepath format: 'mRNA_trajectories_mu_cv_t_ac.npz'
    """
    try:
        name = Path(filepath).stem
        parts = name.split('_')
        # Expected format: ['mRNA', 'trajectories', mu, cv, t_ac]
        if len(parts) >= 5:
            mu = float(parts[2])
            cv = float(parts[3])
            t_ac = float(parts[4])
            return np.array([mu, cv, t_ac])
    except Exception:
        pass
    return None

def make_groups(
    traj_paths,
    num_traj=2, 
    pos_ratio=0.5,
    preprocess_fn=lambda traj: traj.reshape(-1, 1),
    rng=None,
    verbose=False,
    separator_len=1,
    separator_val=-100.0,
    param_dist_threshold=0.7 
):
    """
    Generates A SINGLE group (set of trajectories).
    
    Args:
        param_dist_threshold: Minimum LOG Euclidean distance required between parameters for negative pairs. Default to roughly 0.7 (which corresponds to a ~2x change in any single parameter, since ln(2) =~ 0.69)
    """
    if rng is None:
        rng = random.Random()
        
    # --- 1. Roll the Dice ---
    is_positive = rng.random() < pos_ratio
    
    # --- 2. Select Files ---
    if is_positive:
        if verbose: print("Generating POSITIVE group.")
        file_a = rng.choice(traj_paths)
        label = 1.0
        X = _load_and_concat([file_a], num_traj, preprocess_fn, rng, separator_len=separator_len, separator_val=separator_val)
        
    else:
        if verbose: print("Generating NEGATIVE group.")
        
        # Pick first file
        file_a = rng.choice(traj_paths)
        params_a = _get_params_from_filename(file_a)
        
        # Pick second file with sufficient parameter distance away from file_a
        file_b = None
        max_attempts = 100
        
        for _ in range(max_attempts):
            candidate = rng.choice(traj_paths)
            if candidate == file_a:
                continue
                
            # Check physical distance
            params_b = _get_params_from_filename(candidate)
            
            if params_a is not None and params_b is not None:
                # Calculate Log-Euclidean Distance
                # 1. Take log of both (add small epsilon to avoid log(0))
                log_a = np.log(params_a + 1e-9)
                log_b = np.log(params_b + 1e-9)

                # 2. Euclidean Distance in Log-Space
                # This yields a single scalar "distance"
                dist = np.linalg.norm(log_a - log_b)
                
                if dist > param_dist_threshold:
                    file_b = candidate
                    break # Found a separable pair
            else:
                # Fallback if filename parsing fails: just ensure different files
                file_b = candidate
                break
        
        # Fallback if loop exhausted (unlikely with sparse data, but safe)
        if file_b is None:
            if verbose: print(f"⚠️ Warning: Could not find distinct pair for {file_a.name} after {max_attempts} tries. Picking random.")
            
            while True:
                file_b = rng.choice(traj_paths)
                if file_b != file_a: break

        if verbose: print(f"Selected files:\n A: {file_a.name}\n B: {file_b.name}")
        label = 0.0
        
        num_traj_a = num_traj // 2
        num_traj_b = num_traj - num_traj_a
        
        X_a = _load_and_concat([file_a], num_traj_a, preprocess_fn, rng, separator_len=separator_len, separator_val=separator_val)
        X_b = _load_and_concat([file_b], num_traj_b, preprocess_fn, rng, separator_len=separator_len, separator_val=separator_val)
        
        if X_a is not None and X_b is not None:
            if separator_len > 0:
                n_features = X_a.shape[1]
                separator = np.full((separator_len, n_features), separator_val, dtype=X_a.dtype)
                X = np.concatenate([X_a, separator, X_b], axis=0)
            else:
                X = np.concatenate([X_a, X_b], axis=0)
        else:
            X = None
            
    return (X, label)

class BaselineDataset(Dataset):
    def __init__(self, groups):
        self.groups = groups
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        # === INSTANCE NORMALIZATION ===
        # X shape: (Time_Steps, num_traj)
        X, y = self.groups[idx]

        # Convert to Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        # Calculate mean/std for EACH trajectory (column) independently
        # dim=0 is Time, so we want statistics per column.
        mean = X_tensor.mean(dim=0, keepdim=True)
        std = X_tensor.std(dim=0, keepdim=True) + 1e-8  # Add epsilon to avoid div by zero

        X_norm = (X_tensor - mean) / std
        # print(f"Instance Normalisation - Mean: {mean.flatten().tolist()}, Std: {std.flatten().tolist()}")
        # ==============================

        return X_norm, y_tensor

def baseline_data_prep(
    all_file_paths, 
    batch_size=64, 
    num_groups_train=10000, 
    num_groups_val=1000, 
    num_groups_test=1000, 
    num_traj=2,
    pos_ratio=0.5,
    seed=42,
    separator_len=1,
    separator_val=-100.0,
    verbose=False
):
    # 1. Split FILES first (Zero Leakage)
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # Setup Random Generator (Shared seed for reproducibility)
    rng = random.Random(seed)

    # 2. Iteratively Generate Groups
    print(f"Generating {num_groups_train} training groups...")
    train_groups = [
        make_groups(train_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, separator_len=separator_len, separator_val=separator_val) 
        for _ in tqdm(range(num_groups_train))
    ]
    
    print(f"Generating validation groups...")
    val_groups = [
        make_groups(val_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, separator_len=separator_len, separator_val=separator_val) 
        for _ in tqdm(range(num_groups_val))
    ]
    
    print(f"Generating test groups...")
    test_groups = [
        make_groups(test_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, separator_len=separator_len, separator_val=separator_val) 
        for _ in tqdm(range(num_groups_test))
    ]
    
    # 3. DataLoaders
    train_loader = DataLoader(BaselineDataset(train_groups), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(BaselineDataset(val_groups),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(BaselineDataset(test_groups),  batch_size=batch_size, shuffle=False)
    
    batch_X, _ = next(iter(train_loader))
    print(f"Batch Mean (should be ~0): {batch_X.mean():.4f}")
    print(f"Batch Std  (should be ~1): {batch_X.std():.4f}")
    print(f"Min Value: {batch_X.min():.2f}, Max Value: {batch_X.max():.2f}")
    
    return train_loader, val_loader, test_loader, None


def save_loader_to_disk(loader, save_path):
    """
    Iterates through a dynamic loader (which generates data on-the-fly),
    collects ALL samples, and saves them as a static .pt file.
    
    Args:
        loader: The dynamic DataLoader instance.
        save_path: Path to save the .pt file (e.g., 'data/train_data.pt').
    """
    print(f"❄️  Freezing and saving loader to {save_path}...")
    all_X = []
    all_y = []
    
    # Disable gradient tracking for speed
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(loader, desc="Materializing Data")):
            all_X.append(X)
            all_y.append(y)
    
    # Concatenate all batches into large tensors
    # X shape: (Total_Samples, Time, 1)
    # y shape: (Total_Samples, 1)
    full_X = torch.cat(all_X, dim=0)
    full_y = torch.cat(all_y, dim=0)
    
    # Save dictionary
    torch.save({
        'X': full_X,
        'y': full_y
    }, save_path)
    
    print(f"✅ Saved {full_X.shape[0]} samples to {save_path}")


def load_loader_from_disk(load_path, batch_size=64, shuffle=False):
    """
    Loads a saved .pt file and returns a standard DataLoader.
    This reconstructs the exact dataset saved by save_loader_to_disk.
    
    Args:
        load_path: Path to the .pt file.
        batch_size: Batch size for the new loader.
        shuffle: Whether to shuffle the data (set False for exact reproducibility).
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(f"❌ File not found: {load_path}")
        
    print(f"📂 Loading static data from {load_path}...")
    data = torch.load(load_path)
    
    # Create standard TensorDataset
    dataset = TensorDataset(data['X'], data['y'])
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


# -----------------------------------------------------------------------------
# SSL Data Pipeline
# -----------------------------------------------------------------------------

class SSL_Dataset(Dataset):
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

def ssl_data_prep(
    all_file_paths, 
    batch_size=64, 
    num_groups_train=20000, 
    num_groups_val=2000, 
    num_groups_test=2000, 
    crop_len=200,
    log_scale=False, # this is required if data spans across a large range
    seed=42
):
    """
    Full pipeline for Siamese preparation with Log-Scaling and File-based splitting.
    """
    # 1. Split FILES (Simulations) to prevent leakage
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # 2. Build Pairs
    # We generate MANY pairs (e.g. 20,000) because random cropping makes them distinct.
    # CHANGE: num_groups now refers to positive pairs only.
    # We DO NOT need to build negative groups explicitly.
    print(f"Generating {num_groups_train} training pairs...")
    train_groups = build_groups(train_files, num_groups=num_groups_train, num_traj=2, pos_ratio=1.0, seed=seed) # 100% Positives
    
    print(f"Generating validation/test pairs...")
    val_groups   = build_groups(val_files,   num_groups=num_groups_val,   num_traj=2, pos_ratio=1.0, seed=seed)
    test_groups  = build_groups(test_files,  num_groups=num_groups_test,  num_traj=2, pos_ratio=1.0, seed=seed)
    # 3. Global Log-Scaling (important for training convergence)
    # We apply Log1p FIRST to handle the massive range (mu=1 vs mu=10,000), 
    # THEN fit the StandardScaler.
    scaler = StandardScaler()
    
    if log_scale:
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

    else: 
        print("Using standard scaling as per configuration.")
        
        # Helper to collect all training values for fitting
        def get_raw_values(group_list):
            # Extract X and stack
            all_X = [X for X, y in group_list] 
            return np.vstack(all_X)

        if len(train_groups) > 0:
            print("Fitting scaler on raw training data...")
            # Stack all training data to find global Mean/Std
            train_stack = get_raw_values(train_groups)
            scaler.fit(train_stack.reshape(-1, 1))
        
        def transform_groups(group_list):
            transformed = []
            for X, y in group_list:
                # Standard Scale directly
                shape = X.shape
                # Flatten -> Scale -> Reshape
                X_sc = scaler.transform(X.reshape(-1, 1)).reshape(shape)
                
                transformed.append((X_sc, y))
            return transformed

    # Apply transformations
    train_groups = transform_groups(train_groups)
    val_groups   = transform_groups(val_groups)
    test_groups  = transform_groups(test_groups)

    # 4. Data Augmentation: Create Datasets with Random Cropping
    train_ds = SSL_Dataset(train_groups, crop_len=crop_len, training=True)
    val_ds   = SSL_Dataset(val_groups,   crop_len=crop_len, training=False)
    test_ds  = SSL_Dataset(test_groups,  crop_len=crop_len, training=False)

    # 5. Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, scaler