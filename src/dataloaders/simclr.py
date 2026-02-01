import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# SimCLR Data Pipeline (Chen et al., 2020) adapted for Time Series
# -----------------------------------------------------------------------------

def organize_files_into_groups(files, rng=None):
    """
    Loads files and organizes them into groups.
    Each group corresponds to ONE simulation file (one physical system).
    
    Returns:
        grouped_data: List of lists. Each inner list contains all trajectories from one file.
                      Example: [ [traj1_fileA, traj2_fileA...], [traj1_fileB, ...], ... ]
        labels: List of dummy labels or parameter arrays corresponding to each group.
    """
    grouped_data = []
    group_labels = [] # Optional: Store metadata if needed for downstream tasks
    
    if rng is None:
        rng = random.Random(42)

    print(f"Loading and grouping {len(files)} files...")
    
    for filepath in tqdm(files, desc="Loading Files"):
        try:
            data = np.load(filepath, allow_pickle=True)
            if "trajectories" in data:
                trajs = list(data["trajectories"])
                
                # We need at least 1 trajectory to make a pair (can duplicate if needed)
                if len(trajs) > 0:
                    # If we need strict distinct pairs, we prefer files with >= 2 trajs.
                    # But we can handle single-traj files by duplicating views.
                    grouped_data.append(trajs)
                    
                    # Store filename or params as label (placeholder)
                    group_labels.append(0) 
                    
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            pass
            
    return grouped_data, group_labels

class SimCLR_Dataset(Dataset):
    """
    Dataset aligning with SimCLR (Chen et al. 2020):
    1. Index -> Corresponds to one Simulation File.
    2. GetItem -> Returns two 'Augmented Views' (Two Trajectories t1, t2) from that file.
    
    Input shape: Complete Time Series (padded to max length in dataset if variable).
    Output shape: (t1, t2, label)
    """
    def __init__(self, grouped_data, labels, training=True, sample_len=None, log_scale=True, instance_norm=True, 
                 num_traj=1, separator_len=1, separator_val=-100.0):
        self.grouped_data = grouped_data
        self.labels = labels
        self.training = training
        self.log_scale = log_scale
        self.instance_norm = instance_norm
        self.num_traj = num_traj if num_traj is not None else 1
        self.separator_len = separator_len
        self.separator_val = separator_val
        
        # Determine strict sample length for padding/cropping (if not provided)
        if sample_len is None:
            # Scan a subset to estimate max length to avoid OOM on massive datasets
            # Or scan all if feasible. Here we scan all since data is in memory.
            max_found = 0
            for group in grouped_data:
                for traj in group:
                    if len(traj) > max_found: max_found = len(traj)
            self.sample_len = max_found
        else:
            self.sample_len = sample_len

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        # 1. Select the "System" (File)
        trajectories = self.grouped_data[idx]
        num_available = len(trajectories)
        
        # 2. Select Trajectories for Views
        # We need num_traj for View 1 and num_traj for View 2
        total_needed = 2 * self.num_traj
        
        if num_available >= total_needed:
            # Pick distinct trajectories without replacement
            indices = np.random.choice(num_available, total_needed, replace=False)
        else:
            # Sample with replacement if not enough
            indices = np.random.choice(num_available, total_needed, replace=True)
            
        indices_v1 = indices[:self.num_traj]
        indices_v2 = indices[self.num_traj:]
        
        # Helper to process a set of indices into a single view
        def create_view(subset_indices):
            segments = []
            for i in subset_indices:
                t_raw = trajectories[i]
                if t_raw.ndim == 1: t_raw = t_raw.reshape(-1, 1)
                
                # --- Random Crop / Pad to sample_len ---
                # This acts as augmentation similar to core.py
                curr = t_raw.shape[0]
                target = self.sample_len
                
                if curr > target:
                    if self.training:
                        start = np.random.randint(0, curr - target)
                        t_proc = t_raw[start : start + target]
                    else:
                        # Center crop or start crop for validation? Using start for consistency
                        t_proc = t_raw[:target]
                else:
                    # Pad
                    diff = target - curr
                    t_proc = np.pad(t_raw, ((0, diff), (0, 0)), mode='constant')
                
                # --- Convert to Tensor ---
                t_tensor = torch.tensor(t_proc, dtype=torch.float32)

                # --- Log Scaling ---
                if self.log_scale:
                    t_tensor = torch.log1p(t_tensor)
                
                # --- Instance Norm ---
                if self.instance_norm:
                    m, s = t_tensor.mean(dim=0, keepdim=True), t_tensor.std(dim=0, keepdim=True) + 1e-8
                    t_tensor = (t_tensor - m) / s
                
                segments.append(t_tensor)

            # --- Concatenate with Separators ---
            if self.num_traj > 1 and self.separator_len > 0:
                dtype = segments[0].dtype
                # Create separator tensor (sep_len, features)
                sep = torch.full((self.separator_len, segments[0].shape[1]), self.separator_val, dtype=dtype)
                
                final_parts = []
                for i, seg in enumerate(segments):
                    final_parts.append(seg)
                    if i < len(segments) - 1:
                        final_parts.append(sep)
                return torch.cat(final_parts, dim=0)
            else:
                return torch.cat(segments, dim=0)

        t1_tensor = create_view(indices_v1)
        t2_tensor = create_view(indices_v2)
        y_tensor  = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        
        return t1_tensor, t2_tensor, y_tensor

def ssl_data_prep(
    all_file_paths, 
    batch_size=64, 
    seed=42,
    verbose=False,
    log_scale=True,
    instance_norm=True,
    sample_len=None, 
    num_traj=None,
    separator_len=1,
    separator_val=-100.0,
):
    """
    Prepares data for SimCLR-style training.
    Organizes data by File, then samples pairs from each file.
    
    Args:
        sample_len: Length of EACH trajectory segment (before concatenation).
        num_traj: Number of trajectories to concatenate per view.
    """
    # 1. Split Files (Simulations)
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)
    
    print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
    
    rng = random.Random(seed)

    # 2. Load Groups (File-based grouping)
    # Each element in train_groups is a LIST of trajectories from ONE file.
    train_groups, train_y = organize_files_into_groups(train_files, rng=rng)
    val_groups, val_y     = organize_files_into_groups(val_files, rng=rng)
    test_groups, test_y   = organize_files_into_groups(test_files, rng=rng)

    print(f"Total Unique Systems (Files): {len(train_groups)} Train, {len(val_groups)} Val")

    # 3. Create Datasets
    # We allow the dataset to auto-detect max_len from the training data
    # to ensure all batches have consistent shape.
    
    # Init training set first to find global max_len
    train_ds = SimCLR_Dataset(
        train_groups, train_y, training=True, sample_len=sample_len, 
        log_scale=log_scale, instance_norm=instance_norm,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )
    global_sample_len = train_ds.sample_len
    print(f"Detected/Set Sample Length (per traj): {global_sample_len}")
    
    val_ds = SimCLR_Dataset(
        val_groups, val_y, training=False, sample_len=global_sample_len, 
        log_scale=log_scale, instance_norm=instance_norm,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )
    test_ds = SimCLR_Dataset(
        test_groups, test_y, training=False, sample_len=global_sample_len, 
        log_scale=log_scale, instance_norm=instance_norm,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    # 4. Loaders
    # SimCLR requires large batch sizes for good negative sampling.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, None