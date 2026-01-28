import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .core import make_groups, load_and_split_data

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
    param_dist_threshold=0.7,
    sample_len=None,
    verbose=False
):
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
    rng = random.Random(seed)

    print(f"Generating {num_groups_train} training groups...")
    train_groups = [
        make_groups(train_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, 
                    separator_len=separator_len, separator_val=separator_val, 
                    param_dist_threshold=param_dist_threshold, sample_len=sample_len, stack_axis=0) 
        for _ in tqdm(range(num_groups_train))
    ]
    
    print(f"Generating validation groups...")
    val_groups = [
        make_groups(val_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, 
                    separator_len=separator_len, separator_val=separator_val, 
                    param_dist_threshold=param_dist_threshold, sample_len=sample_len, stack_axis=0) 
        for _ in tqdm(range(num_groups_val))
    ]
    
    print(f"Generating test groups...")
    test_groups = [
        make_groups(test_files, num_traj=num_traj, pos_ratio=pos_ratio, rng=rng, verbose=verbose, 
                    separator_len=separator_len, separator_val=separator_val, 
                    param_dist_threshold=param_dist_threshold, sample_len=sample_len, stack_axis=0) 
        for _ in tqdm(range(num_groups_test))
    ]
    
    train_loader = DataLoader(BaselineDataset(train_groups), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(BaselineDataset(val_groups),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(BaselineDataset(test_groups),  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, None

