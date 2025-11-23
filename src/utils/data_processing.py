"""
Data processing utilities for labelling and processing time series data.
"""
import re
from collections import defaultdict
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pathlib import Path

################## Helper function to capture type mismatch errors between pd.DataFrame and np.ndarray ############
def _ensure_numpy(data):
    """Convert pandas DataFrame to numpy array if needed."""
    if isinstance(data, pd.DataFrame):
        return data.values
    return data

def _safe_slice(data, start_idx=None, end_idx=None, axis=1):
    """Safely slice data whether it's DataFrame or numpy array."""
    if isinstance(data, pd.DataFrame):
        if axis == 1:  # column-wise slicing
            return data.iloc[:, start_idx:end_idx].values
        else:  # row-wise slicing
            return data.iloc[start_idx:end_idx, :].values
    else:
        if axis == 1:
            return data[:, start_idx:end_idx]
        else:
            return data[start_idx:end_idx, :]

def _return_no_label_df(df: pd.DataFrame):
    """Print warning if 'label' column exists in DataFrame, and drop the column if it does."""
    if 'label' in df.columns:
        print("Warning: 'label' column exists in DataFrame.")
        df = df.drop(columns=['label'], errors='ignore')
    return df

################## Helper function to capture type mismatch errors between pd.DataFrame and np.ndarray ############

# Labelling functions
def add_binary_labels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of ``df`` with a new ``label`` column.

    Parameters
    ----------
    df:
        DataFrame containing the simulation parameters & results.
    column:
        Column of the csv file with all the statistical properties used to simulate df, on which to base the 50/50 split.  Values greater than the
        halfway point of the sorted column are labelled ``1``; the rest are
        labelled ``0``.
    Returns
    -------
    pd.DataFrame
        DataFrame with the new ``label`` column added.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'mu_target': [1.0, 1.2, 0.8, 1.1],
    ...     'cv_target': [0.10, 0.15, 0.12, 0.14],
    ...     't_ac_target': [5, 5, 5, 5],
    ... })
    >>> labelled = add_binary_labels(df, 'mu_target')
       mu_target  cv_target  t_ac_target  label
    0        1.0       0.10               5      0
    1        1.2       0.15               5      1
    2        0.8       0.12               5      0
    3        1.1       0.14               5      1      
    """
    labelled = df.copy()
    
    # Check if all values are the same
    if labelled[column].nunique() == 1:
        print(f"Warning: All values in column '{column}' are identical.")
        # Return random 50/50 split
        labels = np.random.choice([0, 1], size=len(labelled))
        labelled["label"] = labels
        return labelled
    
    # Sort the values in the specified column to find the median split point
    order = np.argsort(labelled[column].values)
    
    # Calculate the midpoint index for a 50/50 binary classification split
    midpoint = len(labelled) // 2
    
    # Initialize all labels as 0 (lower half of sorted values)
    labels = np.zeros(len(labelled), dtype=int)
    
    # Assign label 1 to the upper half (values above median)
    labels[order[midpoint:]] = 1
    labelled["label"] = labels
    return labelled

def add_nearest_neighbour_labels(
    df: pd.DataFrame,
    mu_col: str = "mu_target",
    cv_col: str = "cv_target",
    tac_col: str = "t_ac_target",
    *,
    positive_on: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` labelled by nearest neighbour pairing.

    This function groups rows into disjoint pairs based on proximity in the
    three‑dimensional parameter space spanned by ``mu_col``, ``cv_col`` and
    ``tac_col``.  Pairing is performed greedily: the lowest‑index unused row is
    matched with the unused row to which it has the smallest Euclidean distance.
    Within each pair, whichever row has the larger ``positive_on`` value is
    labelled ``1`` and its partner ``0``.  If the number of rows is odd, the
    leftover row is labelled by comparing its ``positive_on`` value to the
    median of that column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results with parameter columns.
    mu_col, cv_col, tac_col : str, optional
        Names of the columns representing the parameter space.  The defaults
        correspond to ``"mu_target"``, ``"cv_target"`` and
        ``"t_ac_target"``.
    positive_on : str, optional
        Column used to decide which element of a pair gets the positive label.
        If ``None`` (default), ``mu_col`` is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``label`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'mu_target': [1.0, 1.2, 0.8, 1.1],
    ...     'cv_target': [0.10, 0.15, 0.12, 0.14],
    ...     't_ac_target': [5, 5, 5, 5],
    ... })
    >>> add_nearest_neighbour_labels(df)
       mu_target  cv_target  t_ac_target  label
    0        1.0       0.10               5      0
    1        1.2       0.15               5      1
    2        0.8       0.12               5      0
    3        1.1       0.14               5      1

    In this example the rows are paired by their nearest neighbours in
    ``(mu_target, cv_target, t_ac_target)`` space.  Within each pair the
    higher ``mu_target`` value receives the positive label ``1``.
    """

    if positive_on is None:
        # Default to using the first coordinate as the label comparator
        positive_on = mu_col

    labelled = df.copy()  # operate on a copy to avoid mutating caller's DataFrame
    coords = labelled[[mu_col, cv_col, tac_col]].to_numpy(dtype=float)
    n = len(coords)

    if n == 0:
        # Nothing to label; return empty column for consistency
        labelled["label"] = np.array([], dtype=int)
        return labelled

    # Precompute pairwise Euclidean distances in parameter space
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)  # ignore self-distances when searching for neighbours

    unused = set(range(n))  # indices not yet paired
    labels = np.zeros(n, dtype=int)
    pos_vals = labelled[positive_on].to_numpy()

    # Greedily build pairs: choose the smallest unused index and match to its nearest neighbour
    while len(unused) >= 2:
        i = min(unused)
        remaining = list(unused - {i})
        j = remaining[np.argmin(dist[i, remaining])]

        # Assign labels within the pair based on the positive_on column
        if pos_vals[i] > pos_vals[j]:
            labels[i], labels[j] = 1, 0
        elif pos_vals[i] < pos_vals[j]:
            labels[i], labels[j] = 0, 1
        else:  # deterministic tie‑break keeps function stable across runs
            labels[i], labels[j] = (0, 1) if i < j else (1, 0)

        unused.remove(i)
        unused.remove(j)

    # If there's an unpaired item, label it relative to the column median
    if unused:
        k = unused.pop()
        median_val = float(np.median(pos_vals))
        labels[k] = 1 if pos_vals[k] >= median_val else 0

    labelled["label"] = labels
    return labelled


def _sample_files_for_groups(rng, files, n_pos, n_neg):
    """
    Helper to select files for positive and negative groups.
    Handles cases where requested groups > available files by switching to replacement.
    """
    # --- 1. Select files for Positive Groups ---
    if n_pos <= len(files):
        pos_files = rng.sample(files, n_pos) if n_pos > 0 else []
    else:
        print(f"Warning: Requested {n_pos} positive groups but only {len(files)} files available. "
              f"Sampling files with replacement.")
        pos_files = rng.choices(files, k=n_pos) if n_pos > 0 else []

    # --- 2. Select files for Negative Groups ---
    neg_files = []
    if n_neg > 0:
        needed_for_disjoint = 2 * n_neg
        
        if needed_for_disjoint <= len(files):
            # Unique disjoint pairs (no file reused across any pair)
            pool = rng.sample(files, needed_for_disjoint)
            neg_files = [(pool[i], pool[i + n_neg]) for i in range(n_neg)]
        else:
            print(f"Warning: Requested {n_neg} negative groups (needs {needed_for_disjoint} unique files) "
                  f"but only {len(files)} files available. Sampling pairs with replacement.")
            # Sample pairs one by one, ensuring (a != b) inside each pair
            for _ in range(n_neg):
                neg_files.append(tuple(rng.sample(files, 2)))
                
    return pos_files, neg_files


# Helper: stack trajs preprocessed trajectories into (seq_len, num_traj)
def _stack_trajs(trajs, preprocess_fn):
    """
    Stack a list of raw trajectories into a single array (seq_len, K).
    """
    proc = [preprocess_fn(t) for t in trajs]  # each -> (seq_len, 1)
    seq_lens = [p.shape[0] for p in proc]
    # unify length (truncate to min length to avoid padding headaches)
    L = min(seq_lens)
    proc = [p[:L] for p in proc]
    X = np.concatenate(proc, axis=1)  # (L, num_traj) 
    return X

def _sample_trajs(rng, pool, k, label, file_id):
    """
    Prefer no-replacement; fallback to replacement with warning.
    """
    if len(pool) >= k:
        return rng.sample(pool, k)
    # failsafe
    print(f"Warning: {label} file {file_id} has only {len(pool)} trajectories; "
            f"sampling {k} with replacement.")
    return [rng.choice(pool) for _ in range(k)]

def build_groups(
    traj_paths,
    num_groups=2, # a pair of positive and negative groups
    num_traj=2, # trajectories per group, 2 for pairs by default
    pos_ratio=0.5,
    preprocess_fn=lambda traj: traj.reshape(-1, 1),  # -> (seq_len, 1)
    seed=42,
):
    """
    Returns: list of (X, y) where
      - X: (seq_len, K)   # K trajectories stacked as K-channels
      - y: 1 for 'all from same file' (positive), 0 for 'mixed files' (negative)
    Assumes all trajectories have the same seq_len (pad/trim if needed).
    """
    from collections import defaultdict
    from tqdm import tqdm
    import random
    
    # reproducibility
    rng = random.Random(seed)
    np.random.seed(seed)
    
    files = list(traj_paths)
    assert len(files) >= 2, "Need at least two files to build negative groups."
    
    # decide counts
    n_pos = int(round(num_groups * pos_ratio))
    n_neg = num_groups - n_pos
    
    # --- Step 1: Select Files (using helper) ---    
    pos_files, neg_files = _sample_files_for_groups(rng, files, n_pos, n_neg)

    # --- Step 2: Load Data ---
    needed_files = set(pos_files)
    for fa, fb in neg_files:
        needed_files.add(fa)
        needed_files.add(fb)

    # Load only the needed files
    grouped = defaultdict(list)
    for path in sorted(needed_files):
            try:
                data = np.load(path, allow_pickle=True)
                if "trajectories" not in data:
                    raise KeyError(f"{path} missing key 'trajectories'.")
                grouped[path] = list(data["trajectories"])
            except Exception as e:
                raise RuntimeError(f"Failed to load {path}: {e}")
            
    groups = []
    
    # --- Step 3: Build Positive Groups using the pre-sampled pos_files ---
    for f in tqdm(pos_files, desc="Building positive groups"):
        pool = grouped[f]
        if len(pool) == 0:
            raise ValueError(f"Positive file {f} has 0 trajectories.")
        trajs = _sample_trajs(rng, pool, num_traj, "positive", f)
        X = _stack_trajs(trajs, preprocess_fn)
        groups.append((X, 1)) # label 1 for positive

    # --- Step 4: Build Negative Groups using the pre-sampled neg_files ---
    for fa, fb in tqdm(neg_files, desc="Building negative groups"):
        # For each negative group, draw a random feasible split k_a∈[1, num_traj-1] (subject to pool sizes) so proportions vary
        # That gives the model more realistic variety and prevents it from memorising trivial cues like “negatives always contain exactly half from each file.”
        pool_a, pool_b = grouped[fa], grouped[fb]
        na, nb = len(pool_a), len(pool_b)
        if na == 0 or nb == 0:
            raise ValueError(f"Empty file in negative pair: {fa}, {fb}")

        # feasible split range for k_a
        k_min = max(1, num_traj - nb)
        k_max = min(na, num_traj - 1)
        if k_min > k_max:
            # if impossible without replacement, allow replacement via k_a in [1, num_traj-1]
            k_min, k_max = 1, num_traj - 1
        k_a = rng.randint(k_min, k_max)
        k_b = num_traj - k_a

        traj_a = _sample_trajs(rng, pool_a, k_a, "negative(A)", fa)
        traj_b = _sample_trajs(rng, pool_b, k_b, "negative(B)", fb)
        
        X = _stack_trajs(traj_a + traj_b, preprocess_fn)
        groups.append((X, 0))  # label 0 for negative

    rng.shuffle(groups)
    return groups


def handle_missing_values(time_series_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a time‑series DataFrame using IterativeImputer.

    - Input must be a pandas DataFrame where each row is one time series and
      each column is a time step/feature.
    - Returns a DataFrame with the same index and columns where missing values
      have been imputed. If no values are missing, a float copy of the input is
      returned.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from utils.data_processing import handle_missing_values
    >>> df = pd.DataFrame(
    ...     {
    ...         "t0": [1.0, np.nan, 3.0],
    ...         "t1": [2.0, 2.5, np.nan],
    ...         "t2": [np.nan, 3.5, 1.5],
    ...     },
    ...     index=["trajA", "trajB", "trajC"],
    ... )
    >>> imputed = handle_missing_values(df)
    >>> imputed.shape == df.shape
    True
    >>> imputed.index.equals(df.index) and imputed.columns.equals(df.columns)
    True
    >>> imputed.isna().sum().sum()  # all missing values imputed
    0
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    if not isinstance(time_series_df, pd.DataFrame):
        raise TypeError(
            "time_series_df must be a pandas DataFrame with rows as samples and columns as time steps"
        )

    df_in = time_series_df
    index = df_in.index
    columns = df_in.columns

    # Convert to matrix (samples × features)
    data_matrix = df_in.to_numpy(dtype=float, copy=False)

    print(f"  Data matrix shape before imputation: {data_matrix.shape}")
    print(f"  Missing values: {np.isnan(data_matrix).sum()} ({np.isnan(data_matrix).mean()*100:.1f}%)")

    if np.isnan(data_matrix).any():
        print("  Applying IterativeImputer for missing value imputation...")
        imputer = IterativeImputer(
            estimator=None,
            missing_values=np.nan,
            max_iter=10,
            tol=1e-3,
            n_nearest_features=None,
            initial_strategy='mean',
            imputation_order='ascending',
            skip_complete=False,
            min_value=None,
            max_value=None,
            verbose=0,
            random_state=42,
        )
        imputed_matrix = imputer.fit_transform(data_matrix)
        print(f"  Imputation completed. Remaining NaN values: {np.isnan(imputed_matrix).sum()}")
    else:
        print("  No missing values detected, using original data")
        imputed_matrix = data_matrix.astype(float, copy=True)

    df_out = pd.DataFrame(imputed_matrix, index=index, columns=columns)

    # Final check and fallback interpolation across time (axis=1)
    if df_out.isna().any().any():
        print("  Warning: Remaining NaNs detected, applying row-wise interpolation...")
        df_interp = df_out.interpolate(method='linear', axis=1, limit_direction='both')
        df_interp = df_interp.ffill(axis=1).bfill(axis=1)
        if df_interp.isna().any().any():
            global_mean = np.nanmean(df_out.to_numpy())
            df_interp = df_interp.fillna(global_mean)
        df_out = df_interp

    return df_out
