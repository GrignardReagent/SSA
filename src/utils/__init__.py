"""
Utility modules for the stochastic_simulations project.

Submodules
----------
processing/
    All data-processing utilities, organised by concern:
      imputation    — fill_nans (numpy), handle_missing_values (DataFrame)
      normalisation — instance_norm_np
      balancing     — balance_classes, balance_by_label
      labelling     — add_binary_labels, add_nearest_neighbour_labels
      pipeline      — prepare_dataset (full end-to-end prep)
      helpers       — internal helpers (_ensure_numpy, _safe_slice, …)

metrics
    Embedding quality metrics: discriminability_d_score, fisher_d_score,
    clustering_agreement_metrics.

embeddings
    SimCLR model loading / encoding and dimensionality reduction:
    load_simclr_model, encode_channel, reduce_embeddings, projection_frame.

Shims (backwards compatible)
-----------------------------
data_processing.py  — re-exports from processing/ (legacy import path)
"""
