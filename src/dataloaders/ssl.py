import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .core import make_groups 

# -----------------------------------------------------------------------------
# SSL Data Pipeline
# -----------------------------------------------------------------------------

class SSL_Dataset(Dataset):
    """
    Dataset wrapper that:
    1. Receives (seq_len, 2) pairs from make_groups.
    2. Performs Independent Random Cropping.
    3. Applies Instance Normalization (per channel).
    """
    def __init__(self, groups, sample_len=200, training=True):
        self.groups = groups
        self.sample_len = sample_len
        self.training = training

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # X has shape (Original_Seq_Len, 2) - Stacked along columns
        X, y = self.groups[idx]
        seq_len = X.shape[0]
        
        # --- Independent Cropping Logic (Translation Invariance) ---
        # We crop x1 and x2 from DIFFERENT timestamps to force physics learning
        if seq_len > self.sample_len:
            if self.training:
                start1 = np.random.randint(0, seq_len - self.sample_len)
                start2 = np.random.randint(0, seq_len - self.sample_len)
                
                x1 = X[start1 : start1 + self.sample_len, 0:1]
                x2 = X[start2 : start2 + self.sample_len, 1:2]
            else:
                # Center crop for validation (deterministic)
                start = (seq_len - self.sample_len) // 2
                x1 = X[start : start + self.sample_len, 0:1]
                x2 = X[start : start + self.sample_len, 1:2]
        else:
            # Pad if too short
            pad_len = self.sample_len - seq_len
            x1 = np.pad(X[:, 0:1], ((0, pad_len), (0, 0)), mode='constant')
            x2 = np.pad(X[:, 1:2], ((0, pad_len), (0, 0)), mode='constant')

        # Convert to Tensor
        x1_t = torch.tensor(x1, dtype=torch.float32)
        x2_t = torch.tensor(x2, dtype=torch.float32)
        y_t  = torch.tensor(y,  dtype=torch.float32).unsqueeze(0)

        # === Instance Normalization (Per View) ===
        # Apply (X - Mean) / Std independently for each view
        # Dimensions are (Time, 1) -> Mean/Std over Time (Dim 0)
        
        m1, s1 = x1_t.mean(dim=0, keepdim=True), x1_t.std(dim=0, keepdim=True) + 1e-8
        x1_norm = (x1_t - m1) / s1
        
        m2, s2 = x2_t.mean(dim=0, keepdim=True), x2_t.std(dim=0, keepdim=True) + 1e-8
        x2_norm = (x2_t - m2) / s2

        return x1_norm, x2_norm, y_t

def ssl_data_prep(
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
    param_dist_threshold=0.7,
    sample_len=200,
    verbose=False
):
    """
    Updated pipeline using make_groups logic + Instance Normalization.
    """
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
    rng = random.Random(seed)

    # Note: stack_axis=1 creates (Time, num_traj) output for SSL
    print(f"Generating {num_groups_train} training pairs...")
    train_groups = [
        make_groups(train_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, stack_axis=1, separator_len=separator_len, separator_val=separator_val, param_dist_threshold=param_dist_threshold) # stack along channels, so that shape: (time, num_traj)
        for _ in tqdm(range(num_groups_train))
    ]
    
    print(f"Generating validation pairs...")
    val_groups = [
        make_groups(val_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, stack_axis=1, separator_len=separator_len, separator_val=separator_val, param_dist_threshold=param_dist_threshold) # stack along channels, so that shape: (time, num_traj)
        for _ in tqdm(range(num_groups_val))
    ]
    
    print(f"Generating test pairs...")
    test_groups = [
        make_groups(test_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, stack_axis=1, separator_len=separator_len, separator_val=separator_val,  param_dist_threshold=param_dist_threshold) # stack along channels, so that shape: (time, num_traj)
        for _ in tqdm(range(num_groups_test))
    ]
    
    train_ds = SSL_Dataset(train_groups, sample_len=sample_len, training=True)
    val_ds   = SSL_Dataset(val_groups,   sample_len=sample_len, training=False)
    test_ds  = SSL_Dataset(test_groups,  sample_len=sample_len, training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, None