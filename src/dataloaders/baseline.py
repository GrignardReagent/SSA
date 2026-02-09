import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .core import make_groups, load_and_split_data

class BaselineDataset(Dataset):
    """
    Generates pairs dynamically on-the-fly during training.
    This drastically reduces RAM usage compared to pre-generating lists.
    """
    def __init__(self, file_paths, num_samples, num_traj=2, pos_ratio=0.5, 
                 separator_len=1, separator_val=-100.0, param_dist_threshold=0.7, 
                 sample_len=None, stack_axis=0):
        self.file_paths = file_paths
        self.num_samples = num_samples # This defines the "Epoch Length"
        self.num_traj = num_traj
        self.pos_ratio = pos_ratio
        self.separator_len = separator_len
        self.separator_val = separator_val
        self.param_dist_threshold = param_dist_threshold
        self.sample_len = sample_len
        self.stack_axis = stack_axis
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # We ignore 'idx' because generation is stochastic.
        # We instantiate a local RNG to ensure randomness across workers.
        # random.Random() seeds from os.urandom or time, ensuring diversity.
        rng = random.Random()
        
        X, y = None, None
        
        # Retry loop: make_groups might return None if it fails to find a distinct pair
        # We loop until we get a valid sample.
        while X is None:
            X, y = make_groups(
                self.file_paths, 
                num_traj=self.num_traj, 
                pos_ratio=self.pos_ratio, 
                rng=rng, 
                verbose=False,
                separator_len=self.separator_len, 
                separator_val=self.separator_val, 
                param_dist_threshold=self.param_dist_threshold, 
                sample_len=self.sample_len, 
                stack_axis=self.stack_axis
            )
            
        # === INSTANCE NORMALIZATION ===
        # Convert to Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        # Calculate mean/std for EACH trajectory (column) independently
        # dim=0 is Time, so we want statistics per column.
        mean = X_tensor.mean(dim=0, keepdim=True)
        std = X_tensor.std(dim=0, keepdim=True) + 1e-8 

        X_norm = (X_tensor - mean) / std

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
    # 1. Split FILES (Zero Leakage)
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    if verbose:
        print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # 2. Create Dynamic Datasets (Lazy Generation)
    # Instead of generating a list, we just pass the file paths.
    
    train_ds = BaselineDataset(
        train_files, num_samples=num_groups_train,
        num_traj=num_traj, pos_ratio=pos_ratio, 
        separator_len=separator_len, separator_val=separator_val,
        param_dist_threshold=param_dist_threshold, sample_len=sample_len,
        stack_axis=0 # Baseline always stacks in time (0)
    )
    
    val_ds = BaselineDataset(
        val_files, num_samples=num_groups_val,
        num_traj=num_traj, pos_ratio=pos_ratio, 
        separator_len=separator_len, separator_val=separator_val,
        param_dist_threshold=param_dist_threshold, sample_len=sample_len,
        stack_axis=0
    )
    
    test_ds = BaselineDataset(
        test_files, num_samples=num_groups_test,
        num_traj=num_traj, pos_ratio=pos_ratio, 
        separator_len=separator_len, separator_val=separator_val,
        param_dist_threshold=param_dist_threshold, sample_len=sample_len,
        stack_axis=0
    )
    
    # 3. DataLoaders
    # Using multiple workers is safe because we create a new random.Random() in __getitem__
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, None