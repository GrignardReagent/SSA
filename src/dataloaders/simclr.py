import hashlib
import json
import os
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Normalisation helpers ──────────────────────────────────────────────────────

def _norm_instance(t: torch.Tensor) -> torch.Tensor:
    """Per-trajectory z-score: mean/std computed along the time axis."""
    m = t.mean(dim=0, keepdim=True)
    s = t.std(dim=0, keepdim=True)
    return (t - m) / s


def _norm_global(t: torch.Tensor, global_mean: float, global_std: float) -> torch.Tensor:
    """Dataset-level z-score using pre-computed mean/std."""
    return (t - global_mean) / global_std


def _norm_joint(t1: torch.Tensor, t2: torch.Tensor):
    """Z-score computed from the concatenated pair so both views share one mean/std."""
    combined = torch.cat([t1, t2], dim=0)
    m = combined.mean(dim=0, keepdim=True)
    s = combined.std(dim=0, keepdim=True)
    return (t1 - m) / s, (t2 - m) / s


def _dataset_fingerprint(file_paths, log_scale: bool) -> str:
    """
    Short SHA-256 fingerprint of the sorted file list + log_scale flag.
    Used to name the stats cache file so different datasets never collide.
    """
    key = json.dumps(
        {"files": sorted(str(p) for p in file_paths), "log_scale": log_scale},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _compute_welford_stats(train_files, log_scale: bool, verbose: bool, cache_dir=None):
    """
    Stream through every training file once and return (mean, std) using
    Chan's parallel update formula.  Memory usage is O(1).

    If cache_dir is given, the result is saved to
    ``<cache_dir>/global_stats_<fingerprint>.json`` on first run and
    loaded from there on every subsequent call with the same file list
    and log_scale setting.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        fingerprint = _dataset_fingerprint(train_files, log_scale)
        cache_path  = os.path.join(cache_dir, f"global_stats_{fingerprint}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                cached = json.load(fh)
            if verbose:
                print(f"Loaded cached global stats from {cache_path} "
                      f"(mean={cached['mean']:.4f}, std={cached['std']:.4f})")
            return cached["mean"], cached["std"]

    # ── Welford / Chan's parallel algorithm ───────────────────────────────────
    welford_n    = 0
    welford_mean = 0.0
    welford_M2   = 0.0

    for f in tqdm(train_files, desc="Welford global stats", disable=not verbose):
        try:
            if str(f).endswith(".csv"):
                df = pd.read_csv(f)
                trajs = [df.values]
            else:
                data = np.load(f, allow_pickle=True)
                if "trajectories" not in data or len(data["trajectories"]) == 0:
                    continue
                trajs = list(data["trajectories"])

            for traj in trajs:
                vals = np.asarray(traj, dtype=np.float64).flatten()
                if log_scale:
                    vals = np.log1p(vals)

                n_batch    = len(vals)
                mean_batch = np.mean(vals)
                M2_batch   = np.var(vals) * n_batch

                delta        = mean_batch - welford_mean
                n_total      = welford_n + n_batch
                welford_mean += delta * n_batch / n_total
                welford_M2   += M2_batch + delta**2 * welford_n * n_batch / n_total
                welford_n     = n_total

        except Exception:
            pass

    if welford_n > 1:
        global_mean = float(welford_mean)
        global_std  = float(np.sqrt(welford_M2 / welford_n))
    else:
        global_mean, global_std = 0.0, 1.0

    # ── Cache save ────────────────────────────────────────────────────────────
    if cache_path is not None:
        with open(cache_path, "w") as fh:
            json.dump(
                {"mean": global_mean, "std": global_std,
                 "n_values": welford_n, "n_files": len(train_files),
                 "log_scale": log_scale},
                fh, indent=2,
            )
        if verbose:
            print(f"Cached global stats to {cache_path}")

    return global_mean, global_std


def batch_norm_collate_fn(batch):
    """
    Custom collate function for 'batch-wise' normalisation mode.
    After stacking the batch, computes mean/std across the batch dimension
    (dim=0) and z-score normalises z1 and z2 using those shared statistics.
    """
    z1_list, z2_list, y_list = zip(*batch)
    z1 = torch.stack(z1_list, dim=0)  # (B, T, C)
    z2 = torch.stack(z2_list, dim=0)
    y  = torch.stack(y_list,  dim=0)

    combined = torch.cat([z1, z2], dim=0)  # (2B, T, C)
    m = combined.mean(dim=0, keepdim=True)  # (1, T, C)
    s = combined.std(dim=0, keepdim=True).clamp(min=1e-8)  # clamp prevents div-by-zero when both views are identical

    z1 = (z1 - m) / s
    z2 = (z2 - m) / s
    return z1, z2, y

# ──────────────────────────────────────────────────────────────────────────────


def _load_trajectories_with_retry(
    file_paths, 
    labels, 
    start_idx
    ):
    """Lazily load the trajectory bag + label for file_paths[start_idx],
    retrying at a random file (up to 10 times) if it is missing/corrupt/empty.
    Shared by SimCLR_Dataset and SimCLR_ExpDataset.
    """
    trajectories = None
    label = 0.0
    current_idx = start_idx
    attempts = 0

    while trajectories is None:
        try:
            path = file_paths[current_idx]
            if str(path).endswith(".csv"):
                df = pd.read_csv(path)
                trajectories = list(df.values)
                label = labels[current_idx]
            else:
                data = np.load(path, allow_pickle=True)
                if "trajectories" in data:
                    loaded = list(data["trajectories"])
                    if len(loaded) > 0:
                        trajectories = loaded
                        label = labels[current_idx]
        except Exception:
            pass

        if trajectories is None:
            current_idx = np.random.randint(0, len(file_paths))
            attempts += 1
            if attempts > 10:
                raise RuntimeError("Failed to load valid data after 10 attempts.")

    return trajectories, label


def _make_pair_from_trajectories(
    trajectories, 
    label, 
    *, 
    num_traj, 
    sample_len, 
    training,
    log_scale, 
    normalisation, 
    global_mean, 
    global_std,
    separator_len, 
    separator_val,
):
    """Draw one random (v1, v2) pair (each made of num_traj trajectories,
    without replacement within the pair) from a bag of trajectories loaded
    from one file/context, apply subsampling/log-scale/normalisation, and
    return (t1_tensor, t2_tensor, y_tensor). Shared by SimCLR_Dataset (called
    once per file per epoch) and SimCLR_ExpDataset (called many times per
    file per epoch, scaled to that file's trajectory count).
    """
    num_available = len(trajectories)
    total_needed = 2 * num_traj

    if num_available >= total_needed:
        indices = np.random.choice(num_available, total_needed, replace=False)
    else:
        indices = np.random.choice(num_available, total_needed, replace=True)

    indices_v1 = indices[:num_traj]
    indices_v2 = indices[num_traj:]

    def create_view(subset_indices, apply_norm=True):
        segments = []
        for i in subset_indices:
            t_raw = trajectories[i]
            if t_raw.ndim == 1: t_raw = t_raw.reshape(-1, 1)

            # A. To Tensor
            t_tensor = torch.tensor(t_raw, dtype=torch.float32)

            # B. Log Scale
            if log_scale:
                t_tensor = torch.log1p(t_tensor)

            # C. Normalisation (skipped for 'joint'/'batch-wise', applied after both views exist)
            if apply_norm:
                if normalisation == 'instance':
                    t_tensor = _norm_instance(t_tensor)
                elif normalisation == 'global':
                    t_tensor = _norm_global(t_tensor, global_mean, global_std)

            # D. Subsample to sample_len
            curr = t_tensor.shape[0]
            target = sample_len
            stride = curr / target

            if curr >= target and training:
                # Random phase offset within the first stride window for augmentation
                phase = np.random.uniform(0, stride)
                idx = np.clip(
                    np.round(phase + np.arange(target) * stride).astype(int),
                    0, curr - 1
                )
            else:
                # Deterministic uniform spacing; also handles curr < target by
                # repeating points (avoids padding)
                idx = np.round(np.linspace(0, curr - 1, target)).astype(int)
            t_proc = t_tensor[idx]

            segments.append(t_proc)

        # E. Concatenate if num_traj > 1
        if num_traj > 1:
            sep = torch.full((separator_len, segments[0].shape[1]), separator_val, dtype=segments[0].dtype)

            final_parts = []
            for i, seg in enumerate(segments):
                final_parts.append(seg)
                if i < len(segments) - 1:
                    final_parts.append(sep)
            return torch.cat(final_parts, dim=0)
        else:
            return segments[0]

    if normalisation == 'joint':
        # Build both views without normalisation, then compute shared stats
        t1_tensor = create_view(indices_v1, apply_norm=False)
        t2_tensor = create_view(indices_v2, apply_norm=False)
        t1_tensor, t2_tensor = _norm_joint(t1_tensor, t2_tensor)
    elif normalisation == 'batch-wise':
        # No per-sample normalisation; batch_norm_collate_fn handles it at batch level
        t1_tensor = create_view(indices_v1, apply_norm=False)
        t2_tensor = create_view(indices_v2, apply_norm=False)
    else:
        t1_tensor = create_view(indices_v1)
        t2_tensor = create_view(indices_v2)

    y_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

    return t1_tensor, t2_tensor, y_tensor


class SimCLR_Dataset(Dataset):
    """
    Lazy-loading Dataset for SimCLR.
    1. Stores only file paths (low memory).
    2. Loads simulation data on-the-fly in __getitem__.
    3. Handles corrupt/empty files by retrying with a random file.

    normalisation: None | 'instance' | 'global' | 'joint' | 'batch-wise'
        None     - no normalisation
        instance - per-trajectory z-score normalisation
        global   - dataset-level z-score using pre-computed mean/std
        joint    - z-score computed from the concatenated pair (v1 + v2),
                   so both views share a single mean/std derived from each other
        batch-wise    - z-score computed across the full batch (both views combined),
                   applied via a custom collate_fn in the DataLoader
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
        trajectories, label = _load_trajectories_with_retry(self.file_paths, self.labels, idx)
        t1_tensor, t2_tensor, y_tensor = _make_pair_from_trajectories(
            trajectories, label,
            num_traj=self.num_traj, sample_len=self.sample_len, training=self.training,
            log_scale=self.log_scale, normalisation=self.normalisation,
            global_mean=self.global_mean, global_std=self.global_std,
            separator_len=self.separator_len, separator_val=self.separator_val,
        )
        return t1_tensor, t2_tensor, y_tensor

# IY035 ONLY, DO NOT USE:
# Taking a set number of pairs makes the data drawn from datasets irreproducible, as sampling is random each time. 
class SimCLR_ExpDataset(Dataset):
    """
    Same lazy-loading / pair-construction behaviour as SimCLR_Dataset, but
    __len__ is a directly-stated pair count (n_pairs) instead of exactly one
    pair per file. SimCLR_Dataset reduces a file with hundreds of
    trajectories to a single pair per epoch, leaving most of its
    trajectories unused; here every __getitem__ call -- regardless of idx --
    independently picks a random file (uniformly, same as make_groups()'s
    rng.choice(traj_paths)) and draws a random pair from it, so n_pairs
    directly controls how many such draws happen per epoch.

    This mirrors dataloaders.ssl.make_groups/ssl_data_prep's "just state how
    many pairs you want" ergonomics, but stays lazy: make_groups pre-
    generates its whole group list up front (num_groups_train samples fully
    materialized in memory before training starts, then reused unchanged
    every epoch), which gets slow and memory-heavy at large counts. Here
    nothing is precomputed -- __getitem__ loads and builds one pair on
    demand each call, exactly like SimCLR_Dataset already did, so n_pairs can
    be raised freely without an upfront generation pass or a resident copy of
    the whole dataset in memory. Every epoch also gets a fresh set of
    (file, cell) draws, not the same fixed n_pairs replayed every time.

    All other constructor parameters and normalisation modes are identical
    to SimCLR_Dataset -- see its docstring.
    """
    def __init__(self,
                 file_paths,
                 labels=None,
                 training=True,
                 sample_len=None,
                 log_scale=False,
                 normalisation=None,
                 global_mean=0.0,
                 global_std=1.0,
                 num_traj=1,
                 separator_len=1,
                 separator_val=-10.0,
                 n_pairs=10000,
                 ):
        self.file_paths = file_paths
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
        self.n_pairs = n_pairs

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # idx is ignored on purpose: every call is an independent uniform
        # random file draw, matching make_groups()'s rng.choice(traj_paths).
        file_idx = np.random.randint(0, len(self.file_paths))
        trajectories, label = _load_trajectories_with_retry(self.file_paths, self.labels, file_idx)
        t1_tensor, t2_tensor, y_tensor = _make_pair_from_trajectories(
            trajectories, label,
            num_traj=self.num_traj, sample_len=self.sample_len, training=self.training,
            log_scale=self.log_scale, normalisation=self.normalisation,
            global_mean=self.global_mean, global_std=self.global_std,
            separator_len=self.separator_len, separator_val=self.separator_val,
        )
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
    stats_cache_dir=".stats_cache",
):
    """
    Prepares data for SimCLR-style training using Lazy Loading.

    normalisation: None | 'instance' | 'global' | 'joint' | 'batch-wise'
        None       - no normalisation
        instance   - per-trajectory z-score normalisation
        global     - dataset-level z-score (stats computed from training set)
        joint      - z-score computed from the concatenated pair (v1 + v2) per sample
        batch-wise - z-score computed across the full batch (both views combined)

    stats_cache_dir: directory where the computed global mean/std is saved as a JSON
        file (keyed by a fingerprint of the file list + log_scale).  Defaults to
        '.stats_cache' in the current working directory.  Only pass this explicitly
        if you want the cache somewhere specific (e.g. a shared filesystem accessible
        to all cluster nodes).  Set to None to disable caching entirely.
    """
    # 1. Split Files
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)

    if verbose:
        print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # 2. Compute Global Stats (only needed for 'global' normalisation)
    global_mean, global_std = 0.0, 1.0

    if normalisation == 'global':
        if verbose: print("Calculating Global Normalisation Stats (Welford streaming over ALL training files)...")
        global_mean, global_std = _compute_welford_stats(train_files, log_scale, verbose, cache_dir=stats_cache_dir)
        if verbose:
            print(f"Welford Global Stats -> Mean: {global_mean:.4f}, Std: {global_std:.4f}")

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
        val_files, training=True, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    test_ds = SimCLR_Dataset(
        test_files, training=True, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val
    )

    # 5. Loaders
    collate_fn = batch_norm_collate_fn if normalisation == 'batch-wise' else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

# IY035 ONLY, DO NOT USE:
# Taking a set number of pairs makes the data drawn from datasets irreproducible, as sampling is random each time. 
def ssl_exp_data_prep(
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
    stats_cache_dir=".stats_cache",
    num_groups_train=10000,
    num_groups_val=1000,
    num_groups_test=1000,
):
    """
    Same as ssl_data_prep, built on SimCLR_ExpDataset instead of
    SimCLR_Dataset: __len__ is a directly-stated pair count
    (num_groups_train/val/test, same naming as dataloaders.ssl.ssl_data_prep)
    instead of exactly one pair per file. Use this when there are few files
    but each holds many trajectories/cells (e.g. labelled experimental data),
    where the one-pair-per-file scheme leaves most trajectories in each file
    unused every epoch.

    Unlike dataloaders.ssl.ssl_data_prep (make_groups), which pre-generates
    its whole group list up front and replays the same fixed set every
    epoch, SimCLR_ExpDataset stays lazy: nothing is precomputed, each
    __getitem__ call draws a fresh random file + pair on demand, and a new
    set of draws happens every epoch -- so num_groups_train can be raised
    freely without an upfront generation pass or holding the whole dataset in
    memory.

    All other parameters and normalisation modes are identical to
    ssl_data_prep -- see its docstring for details.
    """
    # 1. Split Files
    train_files, test_files = train_test_split(all_file_paths, test_size=0.2, random_state=seed)
    train_files, val_files  = train_test_split(train_files, test_size=0.2, random_state=seed)

    if verbose:
        print(f"Files split: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")

    # 2. Compute Global Stats (only needed for 'global' normalisation)
    global_mean, global_std = 0.0, 1.0

    if normalisation == 'global':
        if verbose: print("Calculating Global Normalisation Stats (Welford streaming over ALL training files)...")
        global_mean, global_std = _compute_welford_stats(train_files, log_scale, verbose, cache_dir=stats_cache_dir)
        if verbose:
            print(f"Welford Global Stats -> Mean: {global_mean:.4f}, Std: {global_std:.4f}")

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
    train_ds = SimCLR_ExpDataset(
        train_files, training=True, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val,
        n_pairs=num_groups_train,
    )

    val_ds = SimCLR_ExpDataset(
        val_files, training=True, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val,
        n_pairs=num_groups_val,
    )

    test_ds = SimCLR_ExpDataset(
        test_files, training=True, sample_len=sample_len,
        log_scale=log_scale, normalisation=normalisation,
        global_mean=global_mean, global_std=global_std,
        num_traj=num_traj, separator_len=separator_len, separator_val=separator_val,
        n_pairs=num_groups_test,
    )

    if verbose:
        print(f"Pairs/epoch (dense): {len(train_ds)} Train, {len(val_ds)} Val, {len(test_ds)} Test")

    # 5. Loaders
    collate_fn = batch_norm_collate_fn if normalisation == 'batch-wise' else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
