import numpy as np
import pandas as pd
from typing import Union, Optional, List


def shuffle_time_series(
    data: Union[pd.DataFrame, np.ndarray],
    preserve_columns: Optional[List[str]] = None,
    axis: int = 1,
    random_state: Optional[int] = None,
    inplace: bool = False,
    strategy: str = "per_sample",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Shuffle time series data along the temporal axis for each sample.
    
    This function shuffles the temporal order within each time series while 
    preserving the sample structure. Useful for testing whether temporal 
    order matters for classification tasks.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data where each row represents a time series sample.
        For DataFrames, columns represent time points or features.
        For arrays, shape should be (n_samples, n_timepoints).
    preserve_columns : list of str, optional
        Column names to preserve (not shuffle). Typically used for 
        label columns. Only applicable when data is a DataFrame.
    axis : int, default=1
        Axis along which to shuffle:
        - 1: shuffle columns (time points) within each row (sample)
        - 0: shuffle rows (samples) within each column (time point)
    random_state : int, optional
        Random seed for reproducible shuffling.
    inplace : bool, default=False
        If True, modify the original data. If False, return a copy.
    strategy : {"per_sample", "global"}, default="per_sample"
        - "per_sample": independently permute each sample (row) when axis=1,
          or each column vector when axis=0. This destroys cross-sample
          alignment and temporal structure; good for testing sequence models.
        - "global": apply a single, consistent permutation across all samples
          along the chosen axis. For axis=1, this is equivalent to reordering
          feature columns uniformly for all rows; good for testing that a
          non-sequence model (e.g., SVM) is invariant to a relabeling of
          feature indices.
        
    Returns
    -------
    pd.DataFrame or np.ndarray
        Shuffled data with same shape as input.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Example with DataFrame
    >>> df = pd.DataFrame({
    ...     't_0': [1, 4, 7],
    ...     't_1': [2, 5, 8], 
    ...     't_2': [3, 6, 9],
    ...     'label': [0, 1, 0]
    ... })
    >>> shuffled_df = shuffle_time_series(df, preserve_columns=['label'])
    >>> 
    >>> # Example with numpy array
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> shuffled_arr = shuffle_time_series(arr, random_state=42)
    >>> 
    >>> # Shuffle samples within each time point (axis=0)
    >>> shuffled_samples = shuffle_time_series(arr, axis=0, random_state=42)
    """
    
    # Local RNG for reproducibility without touching global state
    rng = np.random.default_rng(random_state)
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        # Create copy if not inplace
        result = data if inplace else data.copy()
        
        # Determine columns to shuffle
        if preserve_columns is None:
            preserve_columns = []
        
        shuffle_columns = [col for col in result.columns if col not in preserve_columns]
        
        if axis == 1:
            if strategy == "per_sample":
                # Shuffle columns (time points) within each row (sample)
                for i in range(len(result)):
                    # Extract row values for columns to shuffle (positional)
                    row_values = result.loc[result.index[i], shuffle_columns].values
                    # Apply per-row permutation
                    permuted = rng.permutation(row_values)
                    # Assign back to dataframe
                    result.loc[result.index[i], shuffle_columns] = permuted
            elif strategy == "global":
                # Apply the same permutation of columns to all rows.
                # Compute a permutation of the shuffle_columns order, and then
                # reorder those columns consistently while leaving preserved
                # columns untouched and in place.
                col_order = list(result.columns)
                # Indices of the columns we are shuffling (by name)
                shuffle_idx = [col_order.index(c) for c in shuffle_columns]
                permuted_names = [shuffle_columns[k] for k in rng.permutation(len(shuffle_columns))]
                new_order = col_order[:]
                for idx, new_name in zip(shuffle_idx, permuted_names):
                    new_order[idx] = new_name
                result = result[new_order]
            else:
                raise ValueError("strategy must be 'per_sample' or 'global'")
        elif axis == 0:
            if strategy == "per_sample":
                # Shuffle rows (samples) within each column (time point)
                for col in shuffle_columns:
                    col_values = result[col].values
                    result[col] = rng.permutation(col_values)
            elif strategy == "global":
                # Apply the same permutation of rows to all columns
                perm = rng.permutation(len(result))
                result = result.iloc[perm].reset_index(drop=True)
            else:
                raise ValueError("strategy must be 'per_sample' or 'global'")
        else:
            raise ValueError("axis must be 0 or 1")
            
        return result
    
    # Handle numpy array input
    elif isinstance(data, np.ndarray):
        # Create copy if not inplace
        result = data if inplace else data.copy()
        
        if axis == 1:
            if strategy == "per_sample":
                # Shuffle columns within each row independently
                for i in range(result.shape[0]):
                    result[i, :] = rng.permutation(result[i, :])
            elif strategy == "global":
                # Apply the same column permutation for all rows
                perm = rng.permutation(result.shape[1])
                result = result[:, perm]
            else:
                raise ValueError("strategy must be 'per_sample' or 'global'")
        elif axis == 0:
            if strategy == "per_sample":
                # Shuffle rows within each column independently
                for j in range(result.shape[1]):
                    result[:, j] = rng.permutation(result[:, j])
            elif strategy == "global":
                # Apply the same row permutation for all columns
                perm = rng.permutation(result.shape[0])
                result = result[perm, :]
            else:
                raise ValueError("strategy must be 'per_sample' or 'global'")
        else:
            raise ValueError("axis must be 0 or 1")
            
        return result
    
    else:
        raise TypeError("data must be a pandas DataFrame or numpy array")


def create_temporal_control_experiment(
    df: pd.DataFrame,
    label_column: str = 'label',
    random_state: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create original and shuffled datasets for temporal order experiments.
    
    This is a convenience function that creates both the original dataset
    and a temporally-shuffled version for comparing whether temporal order
    matters in classification tasks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data and labels.
    label_column : str, default='label'
        Name of the column containing labels to preserve.
    random_state : int, optional
        Random seed for reproducible results.
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (original_data, shuffled_data) - both ready for train/test splitting.
        
    Examples
    --------
    >>> df_original, df_shuffled = create_temporal_control_experiment(
    ...     df, label_column='label', random_state=42
    ... )
    >>> # Now you can compare classification performance:
    >>> # Train on df_original vs df_shuffled
    """
    original = df.copy()
    shuffled = shuffle_time_series(
        df, 
        preserve_columns=[label_column], 
        random_state=random_state
    )
    
    return original, shuffled
