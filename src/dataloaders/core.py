import numpy as np
import pandas as pd
import random
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

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

def _load_and_concat(files, num_traj, preprocess_fn, rng, separator_len=1, separator_val=-100.0, stack_axis=0, sample_len=None):
    """
    Helper to load files and concatenate 'num_traj' trajectories.
    
    Args:
        stack_axis: 
            0 = Concatenate along Time (for Baseline/Classifier). Shape: (Time * num_traj, 1)
            1 = Stack along Channels (for SSL/Siamese). Shape: (Time, num_traj)
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
        if len(pool) == 0: return None 
        while len(pool) < num_traj:
            pool += pool
            
    # Sample without replacement if possible
    selected_trajs = rng.sample(pool, num_traj)

    # Preprocess
    proc = []
    for traj in selected_trajs:
        t = preprocess_fn(traj) # (Time, 1)
        
        # --- Random Crop if sample_len is set ---
        if sample_len is not None:
            curr_len = t.shape[0]
            if curr_len > sample_len:
                start = rng.randint(0, curr_len - sample_len)
                t = t[start : start + sample_len]
        
        proc.append(t)

    # Crop to min length to allow stacking/concatenation
    L = min(p.shape[0] for p in proc)
    proc = [p[:L] for p in proc]
    
    # === Stack Axis 1 (SSL Mode) ===
    if stack_axis == 1:
        # Result: (L, num_traj)
        return np.concatenate(proc, axis=1)

    # === Stack Axis 0 (Baseline Mode) ===
    # Insert separator if requested
    if separator_len > 0:
        n_features = proc[0].shape[1]
        separator = np.full((separator_len, n_features), separator_val, dtype=proc[0].dtype)
        
        proc_with_separators = []
        for i, p in enumerate(proc):
            proc_with_separators.append(p)
            # add separator except after last
            proc_with_separators.append(separator) if i < len(proc) - 1 else None
        return np.concatenate(proc_with_separators, axis=0)
        
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
    param_dist_threshold=0.7,
    stack_axis=0,
    sample_len=None
):
    """
    Generates A SINGLE group (set of trajectories).
    
    Args:
        stack_axis: 0 for single-channel time series (concatenation), 1 for multi-channel/pair (stacking).
    """
    if rng is None:
        rng = random.Random()
        
    is_positive = rng.random() < pos_ratio
    
    if is_positive:
        if verbose: print("Generating POSITIVE group.")
        file_a = rng.choice(traj_paths)
        label = 1.0
        X = _load_and_concat(
            [file_a], num_traj, preprocess_fn, rng, 
            separator_len=separator_len, separator_val=separator_val, 
            stack_axis=stack_axis, sample_len=sample_len
        )
        
    else:
        if verbose: print("Generating NEGATIVE group.")
        
        file_a = None
        file_b = None
        
        # Try up to 10 different 'File A' candidates
        max_pair_attempts = 10 
        for _ in range(max_pair_attempts):
            candidate_a = rng.choice(traj_paths)
            params_a = _get_params_from_filename(candidate_a)
            
            found_b = False
            max_b_attempts = 100 
            for _ in range(max_b_attempts):
                candidate_b = rng.choice(traj_paths)
                if candidate_b == candidate_a: continue
                
                params_b = _get_params_from_filename(candidate_b)
                
                # Check Log-Euclidean Distance
                if params_a is not None and params_b is not None:
                    log_a = np.log(params_a + 1e-9)
                    log_b = np.log(params_b + 1e-9)
                    dist = np.linalg.norm(log_a - log_b)
                    
                    if dist > param_dist_threshold:
                        file_b = candidate_b
                        found_b = True
                        break 
                else:
                    if candidate_b != candidate_a:
                        file_b = candidate_b
                        found_b = True
                        break
            
            if found_b:
                file_a = candidate_a
                break
        
        # Fallback
        if file_a is None or file_b is None:
            if verbose: print("⚠️ Warning: Could not find distinct pair. Picking random.")
            file_a = rng.choice(traj_paths)
            while True:
                file_b = rng.choice(traj_paths)
                if file_b != file_a: break
    
        label = 0.0
        
        num_traj_a = num_traj // 2
        num_traj_b = num_traj - num_traj_a
        
        X_a = _load_and_concat(
            [file_a], num_traj_a, preprocess_fn, rng, 
            separator_len=separator_len, separator_val=separator_val, 
            stack_axis=stack_axis, sample_len=sample_len
        )
        X_b = _load_and_concat(
            [file_b], num_traj_b, preprocess_fn, rng, 
            separator_len=separator_len, separator_val=separator_val, 
            stack_axis=stack_axis, sample_len=sample_len
        )
        
        if X_a is not None and X_b is not None:
            if stack_axis == 0:
                # Concatenate along time with separator
                if separator_len > 0:
                    n_features = X_a.shape[1]
                    separator = np.full((separator_len, n_features), separator_val, dtype=X_a.dtype)
                    X = np.concatenate([X_a, separator, X_b], axis=0)
                else:
                    X = np.concatenate([X_a, X_b], axis=0)
            else:
                # Stack along channels (Time must match, which _load_and_concat handles via cropping)
                # But X_a and X_b might have different lengths if they came from different files with different raw lengths
                # So we must unify lengths here.
                L = min(X_a.shape[0], X_b.shape[0])
                X_a = X_a[:L]
                X_b = X_b[:L]
                X = np.concatenate([X_a, X_b], axis=1)
        else:
            X = None
            
    return (X, label)

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
    try:
        torch.save({
            'X': full_X,
            'y': full_y
        }, save_path)
    except RuntimeError as e:
        print(f"⚠️ Failed to save directly to {save_path}: {e}")
        print("Attempting to save using sudo...")
        import tempfile
        import subprocess
        import os
        
        fd, temp_path = tempfile.mkstemp(suffix='.pt')
        os.close(fd)
        
        torch.save({
            'X': full_X,
            'y': full_y
        }, temp_path)
        
        subprocess.run(['sudo', 'mv', temp_path, str(save_path)], check=True)
    
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