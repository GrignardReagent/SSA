"""SVM data-preparation utilities."""

from __future__ import annotations

import numpy as np
import torch


def extract_data_for_svm(loader, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Flatten all batches from a DataLoader into arrays for SVM input.

    Converts each batch from ``(batch, time, features)`` to
    ``(batch, time * features)`` and concatenates batches along the sample axis.
    """
    X_list = []
    y_list = []

    if verbose:
        print("Extracting data from loader for SVM...")

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_np = X_batch.cpu().numpy()
            y_np = y_batch.cpu().numpy().ravel()

            # SVMs expect fixed-length feature vectors, not sequence tensors.
            X_flat = X_np.reshape(X_np.shape[0], -1)

            X_list.append(X_flat)
            y_list.append(y_np)

    return np.vstack(X_list), np.concatenate(y_list)
