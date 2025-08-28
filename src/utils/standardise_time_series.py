import pandas as pd
from typing import Sequence, Optional, List


def standardise_time_series(
    data_frames: Sequence[pd.DataFrame],
    labels: Optional[Sequence[int]] = None,
    prefix: str = "t_",
) -> pd.DataFrame:
    """Return concatenated time series with equal length.

    This utility converts a sequence of time-series DataFrames of different
    temporal lengths into a single fixed-size feature matrix. The number of
    columns in the result equals the minimum number of columns across the
    provided time series. Columns are taken from the *end* of each DataFrame
    and renamed to ``{prefix}0`` .. ``{prefix}N``. Optionally a label can be
    appended to each row.

    Parameters
    ----------
    data_frames : sequence of DataFrame
        Each DataFrame contains samples in rows and temporal points in columns.
    labels : sequence of int, optional
        Label for each DataFrame.  When provided, a ``label`` column will be
        appended to the result and each row from the corresponding DataFrame
        will take that label.
    prefix : str, default "t_"
        Prefix used when renaming the temporal columns.

    Returns
    -------
    DataFrame
        All input DataFrames concatenated with consistent column names and no
        missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame([[1, 2, 3]])
    >>> df2 = pd.DataFrame([[4, 5]])
    >>> standardise_time_series([df1, df2], labels=[0, 1])
       t_0  t_1  label
    0    2    3      0
    1    4    5      1
    """
    if not data_frames:
        raise ValueError("`data_frames` must contain at least one DataFrame")

    if labels is not None and len(labels) != len(data_frames):
        raise ValueError("Length of `labels` must match `data_frames`")

    # Find the minimum number of columns across all DataFrames
    min_columns = min(df.shape[1] for df in data_frames)
    column_names = [f"{prefix}{i}" for i in range(min_columns)]

    standardised: List[pd.DataFrame] = []
    for idx, df in enumerate(data_frames):
        ts = df.iloc[:, -min_columns:].copy()
        ts.columns = column_names
        if labels is not None:
            ts["label"] = labels[idx]
        standardised.append(ts)

    result = pd.concat(standardised, ignore_index=True)

    return result