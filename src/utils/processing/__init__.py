"""
Processing utilities for experimental time-series data.

Submodules
----------
helpers
    Internal dtype-conversion helpers (_ensure_numpy, _safe_slice,
    _return_no_label_df). Used by stats and visualisation modules.
imputation
    NaN imputation: fill_nans (numpy), handle_missing_values (DataFrame).
normalisation
    Normalisation: batch_wise_normalise, timepoint_zscore.
balancing
    Class balancing: balance_classes, balance_by_label.
labelling
    Label assignment: add_binary_labels, add_nearest_neighbour_labels.
pipeline
    End-to-end preparation: prepare_dataset.
"""

from utils.processing.helpers import _ensure_numpy, _safe_slice, _return_no_label_df
from utils.processing.imputation import fill_nans, handle_missing_values
from utils.processing.feature_frames import fit_fill_feature_frame
from utils.processing.normalisation import batch_wise_normalise, instance_norm_np
from utils.processing.balancing import balance_classes, balance_by_label
from utils.processing.labelling import add_binary_labels, add_nearest_neighbour_labels
from utils.processing.pipeline import prepare_dataset

__all__ = [
    "_ensure_numpy", "_safe_slice", "_return_no_label_df",
    "fill_nans", "handle_missing_values", "fit_fill_feature_frame",
    "batch_wise_normalise", "instance_norm_np",
    "balance_classes", "balance_by_label",
    "add_binary_labels", "add_nearest_neighbour_labels",
    "prepare_dataset",
]
