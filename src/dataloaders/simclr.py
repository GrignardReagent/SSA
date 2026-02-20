import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SimCLR_Dataset(Dataset):
    """
    Lazy-loading Dataset for SimCLR.
    1. Stores only file paths (low memory).
    2. Loads simulation data on-the-fly in __getitem__.
    3. Handles corrupt/empty files by retrying with a random file.
    """
    def __init__(self, file_paths, labels=None, training=True, sample_len=400, 
                 log_scale=False, instance_norm=True, num_traj=1, 
                 separator_len=1, separator_val=-100.0):
        self.file_paths = file_paths
        # If labels aren't provided, just use dummy 0s (SimCLR doesn't use them for training)
        self.labels = labels if labels is not None else [0] * len(file_paths)
        self.training = training
        self.sample_len = sample_len
        self.log_scale = log_scale
        self.instance_norm = instance_norm
        self.num_traj = num_traj
        self.separator_len = separator_len
        self.separator_val = separator_val

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # --- 1. Lazy Load Logic with Retry ---
        # We loop until we find a valid file (in case of corruption or empty files)
        trajectories = None
        label = 0.0
        
        current_idx = idx
        attempts = 0
        
        while trajectories is None:
            try:
                path = self.file_paths[current_idx]
                data = np.load(path, allow_pickle=True)
                if "trajectories" in data:
                    loaded = list(data["trajectories"])
                    if len(loaded) > 0:
                        trajectories = loaded
                        # Use provided label or 0 if dummy
                        label = self.labels[current_idx]
            except Exception:
                pass
            
            if trajectories is None:
                # Pick a random new file if the current one failed
                current_idx = np.random.randint(0, len(self.file_paths))
                attempts += 1
                if attempts > 10:
                    # Failsafe for very broken datasets
                    raise RuntimeError("Failed to load valid data after 10 attempts.")

        # --- 2. Select Trajectories for Views ---
        num_available = len(trajectories)
        total_needed = 2 * self.num_traj
        
        if num_available >= total_needed:
            # Pick distinct trajectories if possible
            indices = np.random.choice(num_available, total_needed, replace=False)
        else:
            # Duplicate if not enough
            indices = np.random.choice(num_available, total_needed, replace=True)
            
        indices_v1 = indices[:self.num_traj]
        indices_v2 = indices[self.num_traj:]

        # --- 3. View Creation Helper (Preserving Normalization Fix) ---
        def create_view(subset_indices):
            segments = []
            for i in subset_indices:
                t_raw = trajectories[i]
                if t_raw.ndim == 1: t_raw = t_raw.reshape(-1, 1)
                
                # A. To Tensor
                t_tensor = torch.tensor(t_raw, dtype=torch.float32)

                # B. Log Scale
                if self.log_scale:
                    t_tensor = torch.log1p(t_tensor)
                
                # C. Instance Norm (CRITICAL: Normalize FIRST)
                if self.instance_norm:
                    m = t_tensor.mean(dim=0, keepdim=True)
                    s = t_tensor.std(dim=0, keepdim=True) + 1e-8
                    t_tensor = (t_tensor - m) / s
                
                # D. Crop or Pad (Pad with ZEROS SECOND)
                curr = t_tensor.shape[0]
                target = self.sample_len
                
                if curr > target:
                    if self.training:
                        start = np.random.randint(0, curr - target)
                        t_proc = t_tensor[start : start + target]
                    else:
                        t_proc = t_tensor[:target]
                else:
                    diff = target - curr
                    # Pad dim 0 (Time)
                    pad_tensor = torch.zeros((diff, t_tensor.shape[1]), dtype=t_tensor.dtype)
                    t_proc = torch.cat([t_tensor, pad_tensor], dim=0)

                segments.append(t_proc)

            # E. Concatenate if num_traj > 1
            if self.num_traj > 1:
                # Create separator (e.g. -10.0 is safer than -100.0 for normalized data)
                sep_val = self.separator_val
                # Ensure separator matches dtype
                sep = torch.full((self.separator_len, segments[0].shape[1]), sep_val, dtype=segments[0].dtype)
                
                final_parts = []
                for i, seg in enumerate(segments):
                    final_parts.append(seg)
                    if i < len(segments) - 1:
                        final_parts.append(sep)
                return torch.cat(final_parts, dim=0)
            else:
                return segments[0]

        t1_tensor = create_view(indices_v1)
        t2_tensor = create_view(indices_v2)
        y_tensor  = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return t1_tensor, t2_tensor, y_tensor

def ssl_data_prep(
    all_file_paths, 
    batch_size=64, 
    seed=42,
    verbose=False,
    log_scale=True,
    instance_norm=True,
    sample_len=None,  # Important: Must provide or we guess from 1st file
    num_traj=1,
    separator_len=1,
    separator_val=-10.0, # Changed default to -10.0 for stability with Instance Norm
):
    """
    Prepares data for SimCLR-style training using Lazy Loading.
    """
    # 1. Split Files (Simulations)
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    if verbose:
        print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
    
    # 2. Handle Sample Length (Auto-detect from FIRST file if None)
    if sample_len is None:
        try:
            # Peek at the first file to guess length
            temp_data = np.load(train_files[0], allow_pickle=True)
            traj = list(temp_data["trajectories"])[0]
            sample_len = len(traj)

        except Exception as e:
            sample_len = 200 # Fallback
            print(f"⚠️  Could not detect sample_len. Defaulting to {sample_len}")

    # 3. Create Dynamic Datasets
    # We pass the list of files directly. No pre-loading.
    
    train_ds = SimCLR_Dataset(
        train_files, training=True, sample_len=sample_len, 
        log_scale=log_scale, instance_norm=instance_norm,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )
    
    val_ds = SimCLR_Dataset(
        val_files, training=False, sample_len=sample_len, 
        log_scale=log_scale, instance_norm=instance_norm,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )
    
    test_ds = SimCLR_Dataset(
        test_files, training=False, sample_len=sample_len, 
        log_scale=log_scale, instance_norm=instance_norm,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    # 4. Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader