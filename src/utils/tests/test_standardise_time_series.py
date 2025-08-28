import numpy as np
import pandas as pd

from utils.standardise_time_series import standardise_time_series


def test_standardise_different_lengths_no_nan():
    rng = np.random.default_rng(0)
    df1 = pd.DataFrame(rng.normal(size=(2, 5)))
    df2 = pd.DataFrame(rng.normal(size=(2, 7)))
    df3 = pd.DataFrame(rng.normal(size=(2, 6)))

    result = standardise_time_series([df1, df2, df3], labels=[0, 1, 1])

    expected_cols = min(df.shape[1] for df in (df1, df2, df3)) + 1
    assert result.shape == (df1.shape[0] + df2.shape[0] + df3.shape[0], expected_cols)
    assert result.isna().sum().sum() == 0


def test_standardise_without_labels():
    df1 = pd.DataFrame(np.arange(6).reshape(1, 6))
    df2 = pd.DataFrame(np.arange(8).reshape(1, 8))

    result = standardise_time_series([df1, df2])

    expected_cols = min(df1.shape[1], df2.shape[1])
    assert list(result.columns) == [f"t_{i}" for i in range(expected_cols)]
    assert result.isna().sum().sum() == 0