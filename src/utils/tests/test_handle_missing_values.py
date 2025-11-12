import numpy as np
import pandas as pd

from utils.data_processing import handle_missing_values


def test_imputation_preserves_shape_index_and_columns():
    # Create a small DataFrame with NaNs
    idx = ["a", "b", "c"]
    cols = ["t0", "t1", "t2", "t3"]
    df = pd.DataFrame(
        [
            [1.0, np.nan, 3.0, 4.0],
            [np.nan, 2.0, np.nan, 1.0],
            [0.5, 0.7, 0.9, np.nan],
        ],
        index=idx,
        columns=cols,
    )

    out = handle_missing_values(df)

    # Shape and labels preserved
    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.index.equals(df.index)
    assert out.columns.equals(df.columns)

    # No missing values remain
    assert out.isna().sum().sum() == 0


def test_no_missing_values_returns_equal_values():
    idx = ["x", "y"]
    cols = ["t0", "t1", "t2"]
    df = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        index=idx,
        columns=cols,
    )

    out = handle_missing_values(df)

    # Values should be numerically identical
    np.testing.assert_allclose(out.to_numpy(), df.to_numpy(dtype=float))
    assert out.index.equals(df.index)
    assert out.columns.equals(df.columns)


def test_raising_on_non_dataframe_input():
    data = [np.array([1.0, np.nan, 2.0])]
    try:
        handle_missing_values(data)  # type: ignore[arg-type]
    except TypeError as e:
        assert "pandas DataFrame" in str(e)
    else:
        raise AssertionError("TypeError was not raised for non-DataFrame input")

