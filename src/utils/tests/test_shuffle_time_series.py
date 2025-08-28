import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the src directory is on the Python path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.shuffle_time_series import shuffle_time_series


def test_shuffle_dataframe_axis1_preserve_columns_and_reproducible():
    df = pd.DataFrame({
        't0': [1, 4, 7],
        't1': [2, 5, 8],
        't2': [3, 6, 9],
        'label': [0, 1, 0],
    })

    shuffled = shuffle_time_series(df, preserve_columns=['label'], random_state=42)

    # label column should be unchanged
    assert shuffled['label'].tolist() == df['label'].tolist()

    # each row of time values should be a permutation of the original
    for idx in df.index:
        original_row = df.loc[idx, ['t0', 't1', 't2']].tolist()
        shuffled_row = shuffled.loc[idx, ['t0', 't1', 't2']].tolist()
        assert sorted(original_row) == sorted(shuffled_row)

    # at least one row should be different from the original order
    assert not shuffled[['t0', 't1', 't2']].equals(df[['t0', 't1', 't2']])

    # shuffling with the same seed should be reproducible
    shuffled_again = shuffle_time_series(df, preserve_columns=['label'], random_state=42)
    assert shuffled[['t0', 't1', 't2']].equals(shuffled_again[['t0', 't1', 't2']])


def test_shuffle_numpy_array_axis1_and_axis0():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    shuffled = shuffle_time_series(arr, random_state=0)

    # each row should be a permutation of the original row
    for i in range(arr.shape[0]):
        assert sorted(arr[i].tolist()) == sorted(shuffled[i].tolist())
    assert not np.array_equal(arr, shuffled)

    # reproducibility
    shuffled_again = shuffle_time_series(arr, random_state=0)
    assert np.array_equal(shuffled, shuffled_again)

    # axis=0 shuffling - each column permuted
    arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    shuffled_axis0 = shuffle_time_series(arr2, axis=0, random_state=0)
    for j in range(arr2.shape[1]):
        assert sorted(arr2[:, j].tolist()) == sorted(shuffled_axis0[:, j].tolist())
    assert not np.array_equal(arr2, shuffled_axis0)