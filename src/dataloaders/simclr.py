import torch
import numpy as np
import pandas as pd
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

    normalisation: None | 'instance' | 'global'
        None     - no normalisation
        instance - per-trajectory z-score normalisation
        global   - dataset-level z-score using pre-computed mean/std
    """
    def __init__(self,
                 file_paths,
                 labels=None,
                 training=True,
                 sample_len=500,
                 log_scale=False,
                 normalisation=None,
                 global_mean=0.0,
                 global_std=1.0,
                 num_traj=1,
                 separator_len=1,
                 separator_val=-10.0
                 ):
        self.file_paths = file_paths
        # If labels aren't provided, just use dummy 0s (SimCLR doesn't use them for training)
        self.labels = labels if labels is not None else [0] * len(file_paths)
        self.training = training
        self.sample_len = sample_len
        self.log_scale = log_scale
        self.normalisation = normalisation
        self.global_mean = global_mean
        self.global_std = global_std
        self.num_traj = num_traj
        self.separator_len = separator_len
        self.separator_val = separator_val

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # --- 1. Lazy Load Logic with Retry ---
        trajectories = None
        label = 0.0

        current_idx = idx
        attempts = 0

        while trajectories is None:
            try:
                path = self.file_paths[current_idx]
                if str(path).endswith(".csv"):
                    df = pd.read_csv(path)
                    trajectories = list(df.values)
                    label = self.labels[current_idx]
                else:
                    data = np.load(path, allow_pickle=True)
                    if "trajectories" in data:
                        loaded = list(data["trajectories"])
                        if len(loaded) > 0:
                            trajectories = loaded
                            label = self.labels[current_idx]
            except Exception:
                pass

            if trajectories is None:
                current_idx = np.random.randint(0, len(self.file_paths))
                attempts += 1
                if attempts > 10:
                    raise RuntimeError("Failed to load valid data after 10 attempts.")

        # --- 2. Select Trajectories for Views ---
        num_available = len(trajectories)
        total_needed = 2 * self.num_traj

        if num_available >= total_needed:
            indices = np.random.choice(num_available, total_needed, replace=False)
        else:
            indices = np.random.choice(num_available, total_needed, replace=True)

        indices_v1 = indices[:self.num_traj]
        indices_v2 = indices[self.num_traj:]

        # --- 3. View Creation Helper ---
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

                # C. Normalisation
                if self.normalisation == 'instance':
                    m = t_tensor.mean(dim=0, keepdim=True)
                    s = t_tensor.std(dim=0, keepdim=True) + 1e-8
                    t_tensor = (t_tensor - m) / s
                elif self.normalisation == 'global':
                    t_tensor = (t_tensor - self.global_mean) / self.global_std

                # D. Crop or Pad
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
                    pad_tensor = torch.zeros((diff, t_tensor.shape[1]), dtype=t_tensor.dtype)
                    t_proc = torch.cat([t_tensor, pad_tensor], dim=0)

                segments.append(t_proc)

            # E. Concatenate if num_traj > 1
            if self.num_traj > 1:
                sep = torch.full((self.separator_len, segments[0].shape[1]), self.separator_val, dtype=segments[0].dtype)

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
    log_scale=False,
    normalisation=None,
    sample_len=None,
    num_traj=1,
    separator_len=1,
    separator_val=-10.0,
):
    """
    Prepares data for SimCLR-style training using Lazy Loading.

    normalisation: None | 'instance' | 'global'
        None     - no normalisation
        instance - per-trajectory z-score normalisation
        global   - dataset-level z-score (stats computed from training set)
    """
    # 1. Split Files
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)

    if verbose:
        print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # 2. Compute Global Stats (only needed for 'global' normalisation)
    global_mean, global_std = 0.0, 1.0

    if normalisation == 'global':
        if verbose: print("Calculating Global Normalisation Stats...")
        # Sample up to 200 files for a fast, accurate estimation
        subset_files = np.random.choice(train_files, min(200, len(train_files)), replace=False)
        all_vals = []
        for f in subset_files:
            try:
                if str(f).endswith(".csv"):
                    df = pd.read_csv(f)
                    trajs = list(df.values)
                    for traj in trajs:
                        all_vals.append(traj)
                else:
                    data = np.load(f, allow_pickle=True)
                    if "trajectories" in data and len(data["trajectories"]) > 0:
                        traj_array = np.concatenate(list(data["trajectories"]))
                        all_vals.append(traj_array)
            except Exception:
                pass

        if all_vals:
            all_vals_cat = np.concatenate(all_vals)
            if log_scale:
                all_vals_cat = np.log1p(all_vals_cat)

            global_mean = float(np.mean(all_vals_cat))
            global_std = float(np.std(all_vals_cat)) + 1e-8

            if verbose: print(f"Global Stats -> Mean: {global_mean:.4f}, Std: {global_std:.4f}")

    # 3. Handle Sample Length
    if sample_len is None:
        try:
            if str(train_files[0]).endswith(".csv"):
                temp_data = pd.read_csv(train_files[0])
                traj = list(temp_data.values)[0]
                sample_len = len(traj)
            else:
                temp_data = np.load(train_files[0], allow_pickle=True)
                traj = list(temp_data["trajectories"])[0]
                sample_len = len(traj)
        except Exception as e:
            sample_len = 200
            print(f"Could not detect sample_len. Defaulting to {sample_len}")

    # 4. Create Dynamic Datasets
    train_ds = SimCLR_Dataset(
        train_files, training=True, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    val_ds = SimCLR_Dataset(
        val_files, training=False, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    test_ds = SimCLR_Dataset(
        test_files, training=False, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    # 5. Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
