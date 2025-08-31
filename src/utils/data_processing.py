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


def add_binary_labels(df: pd.DataFrame, column: str = "mu_target") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``label`` column using a median split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results.
    column : str, default ``"mu_target"``
        Column used to determine the split; the lower half receives ``0`` and
        the upper half ``1``. If all values are identical, labels are assigned
        randomly to keep the 50/50 balance.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a ``label`` column added.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'mu_target': [1.0, 2.0, 3.0, 4.0]})
    >>> add_binary_labels(df)['label'].tolist()
    [0, 0, 1, 1]
    """

    labelled = df.copy()
    if labelled[column].nunique() == 1:
        print(f"Warning: All values in column '{column}' are identical.")
        labelled["label"] = np.random.choice([0, 1], size=len(labelled))
        return labelled

    order = np.argsort(labelled[column].values)
    midpoint = len(labelled) // 2
    labels = np.zeros(len(labelled), dtype=int)
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
    """Label rows by pairing nearest neighbours in parameter space.

    Rows are greedily paired based on Euclidean distance in the
    ``(mu_col, cv_col, tac_col)`` parameter space. Within each pair, the row with
    the larger ``positive_on`` value receives label ``1``. If there is an odd
    row out, it is compared against the median of ``positive_on``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation results.
    mu_col, cv_col, tac_col : str, optional
        Column names representing the parameter space. ``tac_col`` defaults to
        ``"t_ac_target"``.
    positive_on : str, optional
        Column used to determine which element of a pair receives label ``1``.
        Defaults to ``mu_col``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a ``label`` column added.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'mu_target': [1.0, 1.1, 2.0, 2.05],
    ...     'cv_target': [0.1, 0.1, 0.2, 0.2],
    ...     't_ac_target': [0.5, 0.5, 0.6, 0.6],
    ... })
    >>> add_nearest_neighbour_labels(df)['label'].tolist()
    [0, 1, 0, 1]
    """

    labelled = df.copy()
    if positive_on is None:
        positive_on = mu_col

    coords = labelled[[mu_col, cv_col, tac_col]].to_numpy(dtype=float)
    n = len(coords)
    if n == 0:
        labelled["label"] = np.array([], dtype=int)
        return labelled

    # Precompute pairwise Euclidean distances in parameter space
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    unused = set(range(n))
    labels = np.zeros(n, dtype=int)
    pos_vals = labelled[positive_on].to_numpy()

    # Greedily build pairs: pick smallest unused index and match to its nearest neighbour
    while len(unused) >= 2:
        i = min(unused)
        remaining = list(unused - {i})
        j = remaining[np.argmin(dist[i, remaining])]

        # Assign labels within the pair based on positive_on column
        if pos_vals[i] > pos_vals[j]:
            labels[i], labels[j] = 1, 0
        elif pos_vals[i] < pos_vals[j]:
            labels[i], labels[j] = 0, 1
        else:  # deterministic tie-break for stability
            labels[i], labels[j] = (0, 1) if i < j else (1, 0)

        unused.remove(i)
        unused.remove(j)

    # If one row remains, label relative to column median
    if unused:
        k = unused.pop()
        median_val = float(np.median(pos_vals))
        labels[k] = 1 if pos_vals[k] >= median_val else 0

    labelled["label"] = labels
    return labelled


def add_labels(
    df: pd.DataFrame,
    labeling_regime: str = "binary",
    *,
    column: str = "mu_target",
    mu_col: str = "mu_target",
    cv_col: str = "cv_target",
    tac_col: str = "t_ac_target",
    positive_on: str | None = None,
) -> pd.DataFrame:
    """Wrapper selecting between :func:`add_binary_labels` and
    :func:`add_nearest_neighbour_labels`.

    Parameters mirror those of the underlying functions. ``labeling_regime``
    chooses which labelling strategy to apply.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'mu_target': [1.0, 1.1, 2.0, 2.05],
    ...     'cv_target': [0.1, 0.1, 0.2, 0.2],
    ...     't_ac_target': [0.5, 0.5, 0.6, 0.6],
    ... })
    >>> add_labels(df, labeling_regime='binary')['label'].tolist()
    [0, 0, 1, 1]
    >>> add_labels(df, labeling_regime='nearest_neighbour')['label'].tolist()
    [0, 1, 0, 1]
    """

    if labeling_regime == "binary":
        return add_binary_labels(df, column=column)
    if labeling_regime == "nearest_neighbour":
        return add_nearest_neighbour_labels(
            df, mu_col=mu_col, cv_col=cv_col, tac_col=tac_col, positive_on=positive_on
        )
    raise ValueError(
        f"Unknown labeling_regime '{labeling_regime}'. Choose 'binary' or 'nearest_neighbour'."
    )
# helper function to extract the path stem and append as a new column in the df
# def _


# def add_pair_labels(
#     df: pd.DataFrame,
#     regime: str = "nearest_neighbour",
#     *,
#     high_is_positive: bool = True,
#     inplace: bool = True,
#     overwrite: bool = False,
#     pattern: str = r"mRNA_trajectories_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)",
#     source_col: str = "source",
# ) -> pd.DataFrame:
#     """
#     Append a binary `label` column to each row of `df` using filename-encoded parameters.

#     Expected `df` columns
#     ---------------------
#     - source_col (default 'source'): filename stem like
#       'mRNA_trajectories_<mu>_<cv>_<t_ac>' ('.csv' suffix allowed but not required).
#       Every row in a given file inherits the same label.

#     Regimes
#     -------
#     1) 'nearest_neighbour' (default):
#        For each (mu, t_ac) group, pair files by *closest cv* (implemented as
#        sorted-adjacent greedy pairing). Within each pair:
#          - label = 1 for the higher cv if high_is_positive=True, else 0
#          - label = 0 for the lower cv if high_is_positive=True, else 1
#        If a group has an odd number of files, the unpaired file is labelled by
#        comparing its cv to the group median cv (above/at median gets the
#        'positive' label).

#     2) 'mu':
#        A global split on mu using the median across all files:
#          - label = 1 if mu >= median(mu) when high_is_positive=True (else 0)
#          - ties on the median are broken by cv (higher cv is 'positive').

#     Parameters
#     ----------
#     regime : {'nearest_neighbour', 'mu'}
#         Labelling strategy.
#     high_is_positive : bool, default True
#         Controls which side of the split gets label 1.
#     inplace : bool, default True
#         If False, operate on a copy and return it.
#     overwrite : bool, default False
#         If False and 'label' already exists, raise; if True, overwrite it.
#     pattern : str
#         Regex used to extract (mu, cv, t_ac) from `source`.
#         Must contain three capturing groups in the order (mu, cv, t_ac).
#     source_col : str, default 'source'
#         Column name that stores the filename stem.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with a `label` column added (or updated if overwrite=True).

#     Raises
#     ------
#     KeyError
#         If `source_col` is missing.
#     ValueError
#         If a source does not match the expected pattern.
#     """
    
#     if source_col not in df.columns:
#         raise KeyError(f"'{source_col}' column is required but not found.")

#     if "label" in df.columns and not overwrite:
#         # Do nothing unless explicitly told to overwrite
#         return df if inplace else df.copy()

#     work = df if inplace else df.copy()
    
#     # Extract parameters from filenames like "mRNA_trajectories_1.0_0.5_2.0.csv" where the numbers are mu, cv, and t_ac values
#     pattern = re.compile(r"mRNA_trajectories_([0-9.]+)_([0-9.]+)_([0-9.]+)")
    
#     # Group files by their mu and t_ac values (ignoring cv for now). This creates groups of files that have the same mu and t_ac but different cv values
#     groups: dict[tuple[float, float], list[tuple[float, str]]] = defaultdict(list)
    
#     for source in df["source"].unique():
#         # ``source`` stores the filename stem of the synthetic CSV each row originated from. Grouping by it lets us pair trajectories from the same parameter sweep.
#         match = pattern.match(source)
#         if not match:
#             raise ValueError(f"unrecognised filename pattern: {source}")
#         mu, cv, t_ac = map(float, match.groups())
#         # Group files by (mu, t_ac) and store their cv values with filenames
#         groups[(mu, t_ac)].append((cv, source))
    
#     # Use nearest neighbour for labelling data
#     if regime == 'nearest_neighbour':
        
    
#     # use the magnitude of mu of the trajectory as a regime for labelling data
#     if regime == 'mu':
 

# import re
# from collections import defaultdict
# from typing import Dict, List, Tuple
# import numpy as np
# import pandas as pd

# def add_pair_labels(
#     df: pd.DataFrame,
#     regime: str = "sobol_nn",
#     *,
#     high_is_positive: bool = True,
#     inplace: bool = True,
#     overwrite: bool = False,
#     pattern: str = r"mRNA_trajectories_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)",
#     source_col: str = "source",
# ) -> pd.DataFrame:
#     """
#         Append a binary `label` column to each row of `df` using filename-encoded parameters.

#     Expected `df` columns
#     ---------------------
#     - source_col (default 'source'): filename stem like
#       'mRNA_trajectories_<mu>_<cv>_<t_ac>' ('.csv' suffix allowed but not required).
#       Every row in a given file inherits the same label.

#     Regimes
#     -------
#     1) 'sobol_nn' (default): build disjoint pairs by greedy nearest-neighbour
#       matching in (mu, t_ac) space. Within each pair, higher CV gets label 1
#       if high_is_positive=True (else 0). If one item is left unmatched (odd N),
#       compare its CV to the global median CV to assign a label.
      
#     2) 'nearest_neighbour':
#        For each (mu, t_ac) group, pair files by *closest cv* (implemented as
#        sorted-adjacent greedy pairing). Within each pair:
#          - label = 1 for the higher cv if high_is_positive=True, else 0
#          - label = 0 for the lower cv if high_is_positive=True, else 1
#        If a group has an odd number of files, the unpaired file is labelled by
#        comparing its cv to the group median cv (above/at median gets the
#        'positive' label).

#     3) 'mu':
#        A global split on mu using the median across all files:
#          - label = 1 if mu >= median(mu) when high_is_positive=True (else 0)
#          - ties on the median are broken by cv (higher cv is 'positive').

#     Parameters
#     ----------
#     regime : {'nearest_neighbour', 'mu'}
#         Labelling strategy.
#     high_is_positive : bool, default True
#         Controls which side of the split gets label 1.
#     inplace : bool, default True
#         If False, operate on a copy and return it.
#     overwrite : bool, default False
#         If False and 'label' already exists, raise; if True, overwrite it.
#     pattern : str
#         Regex used to extract (mu, cv, t_ac) from `source`.
#         Must contain three capturing groups in the order (mu, cv, t_ac).
#     source_col : str, default 'source'
#         Column name that stores the filename stem.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with a `label` column added (or updated if overwrite=True).

#     Raises
#     ------
#     KeyError
#         If `source_col` is missing.
#     ValueError
#         If a source does not match the expected pattern.
    
#     """
#     if source_col not in df.columns:
#         raise KeyError(f"'{source_col}' column is required but not found.")
#     if "label" in df.columns and not overwrite:
#         return df if inplace else df.copy()

#     work = df if inplace else df.copy()

#     # Parse parameters from source strings (or, better, replace this with
#     # direct numeric columns if you have them).
#     rx = re.compile(pattern + r"(?:\.csv)?$")
#     mu_map: Dict[str, float] = {}
#     cv_map: Dict[str, float] = {}
#     tac_map: Dict[str, float] = {}

#     for src in work[source_col].astype(str).unique():
#         m = rx.match(src)
#         if not m:
#             raise ValueError(f"Unrecognised filename pattern: {src!r}")
#         mu, cv, t_ac = map(float, m.groups())
#         mu_map[src] = mu
#         cv_map[src] = cv
#         tac_map[src] = t_ac

#     sources = list(mu_map.keys())
#     mu = np.array([mu_map[s] for s in sources], dtype=float)
#     cv = np.array([cv_map[s] for s in sources], dtype=float)
#     tac = np.array([tac_map[s] for s in sources], dtype=float)

#     label_by_source: Dict[str, int] = {}

#     if regime == "sobol_nn":
#         # Greedy disjoint matching by nearest neighbour in (mu, t_ac).
#         n = len(sources)
#         unused = set(range(n))
#         # Precompute distances (μ, τ_ac) Euclidean
#         X = np.column_stack([mu, tac])
#         D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
#         np.fill_diagonal(D, np.inf)

#         # To make deterministic, break ties by index
#         while len(unused) >= 2:
#             # Pick a seed deterministically (smallest index still unused)
#             i = min(unused)
#             # Find nearest neighbour among remaining
#             j_candidates = list(unused - {i})
#             j = j_candidates[np.argmin(D[i, j_candidates])]
#             # Assign labels within the pair by CV
#             if (cv[i] >= cv[j]) == high_is_positive:
#                 pos, neg = i, j
#             else:
#                 pos, neg = j, i
#             label_by_source[sources[pos]] = 1
#             label_by_source[sources[neg]] = 0
#             unused.remove(i)
#             unused.remove(j)

#         # Handle odd one out
#         if len(unused) == 1:
#             k = next(iter(unused))
#             cv_med = float(np.median(cv))
#             is_pos = (cv[k] >= cv_med) if high_is_positive else (cv[k] < cv_med)
#             label_by_source[sources[k]] = 1 if is_pos else 0

#     elif regime == "nearest_neighbour":
#         # Legacy behaviour: group by exact (mu, t_ac) keys and pair by adjacent CV.
#         # With Sobol this will often do nothing; kept for compatibility.
#         groups: Dict[Tuple[float, float], List[Tuple[float, str]]] = defaultdict(list)
#         for s in sources:
#             groups[(mu_map[s], tac_map[s])].append((cv_map[s], s))

#         for (_, _), items in groups.items():
#             items = sorted(items, key=lambda x: x[0])
#             i = 0
#             while i + 1 < len(items):
#                 (cv_lo, src_lo), (cv_hi, src_hi) = items[i], items[i + 1]
#                 if high_is_positive:
#                     label_by_source[src_hi] = 1
#                     label_by_source[src_lo] = 0
#                 else:
#                     label_by_source[src_hi] = 0
#                     label_by_source[src_lo] = 1
#                 i += 2
#             if i < len(items):
#                 # odd one — use group median CV
#                 med = float(np.median([c for c, _ in items]))
#                 cv_last, src_last = items[-1]
#                 is_pos = (cv_last >= med) if high_is_positive else (cv_last < med)
#                 label_by_source[src_last] = 1 if is_pos else 0

#     elif regime == "mu":
#         mu_med = float(np.median(mu))
#         cv_med = float(np.median(cv))
#         for s, mu_v, cv_v in zip(sources, mu, cv):
#             if mu_v > mu_med or (mu_v == mu_med and cv_v >= cv_med):
#                 is_pos = True
#             else:
#                 is_pos = False
#             if not high_is_positive:
#                 is_pos = not is_pos
#             label_by_source[s] = 1 if is_pos else 0

#     else:
#         raise ValueError("Unsupported regime. Use 'sobol_nn', 'nearest_neighbour', or 'mu'.")

#     # Broadcast to rows
#     work["label"] = work[source_col].map(label_by_source).astype(np.int8)
#     if work["label"].isna().any():
#         missing = work.loc[work["label"].isna(), source_col].unique()
#         raise RuntimeError(
#             "Failed to assign labels for some sources. "
#             f"Check parsing or pattern. Missing examples: {missing[:5]!r}"
#         )
#     return work

