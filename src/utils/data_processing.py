"""
Backwards-compatibility shim for utils.data_processing.

All symbols now live in utils/processing/. This file re-exports them so that
existing experiment scripts and tests that do
    from utils.data_processing import <name>
continue to work without modification.
"""

from utils.processing.helpers import _ensure_numpy, _safe_slice, _return_no_label_df
from utils.processing.imputation import handle_missing_values
from utils.processing.labelling import add_binary_labels, add_nearest_neighbour_labels

__all__ = [
    "_ensure_numpy",
    "_safe_slice",
    "_return_no_label_df",
    "handle_missing_values",
    "add_binary_labels",
    "add_nearest_neighbour_labels",
]
