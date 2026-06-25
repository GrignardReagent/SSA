"""
Embedding extraction and dimensionality reduction utilities.

Covers SimCLR model loading / encoding and 2-D projection helpers used across
experiment notebooks (IY028, IY029, IY030, IY031, …).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ── SimCLR model utilities ────────────────────────────────────────────────────

def parse_arch_from_name(name: str) -> dict:
    """Infer SSL_Transformer architecture parameters from a checkpoint filename.

    Checkpoint filenames encode model width (D), attention heads (H) and depth
    (L) as ``_D<int>``, ``_H<int>``, ``_L<int>`` substrings.

    Parameters
    ----------
    name:
        Filename (or full path) of the ``.pth`` checkpoint.

    Returns
    -------
    dict
        Keyword arguments accepted by :class:`models.ssl_transformer.SSL_Transformer`.

    Raises
    ------
    AttributeError
        If any of the D / H / L patterns are missing from the filename.

    Examples
    --------
    >>> parse_arch_from_name("IY024_simCLR_b64_lr0.01_L2_H4_D16_model.pth")
    {'input_size': 1, 'd_model': 16, 'nhead': 4, 'num_layers': 2, 'dropout': 0.01, 'use_conv1d': False}
    """
    name = Path(name).name
    return {
        "input_size": 1,
        "d_model": int(re.search(r"_D(\d+)", name).group(1)),
        "nhead": int(re.search(r"_H(\d+)", name).group(1)),
        "num_layers": int(re.search(r"_L(\d+)", name).group(1)),
        "dropout": 0.01,
        "use_conv1d": False,
    }


def load_simclr_model(checkpoint_path: str | Path, device):
    """Load a frozen SSL_Transformer from a ``.pth`` checkpoint.

    Architecture parameters are inferred from the checkpoint filename via
    :func:`parse_arch_from_name`.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint saved during SimCLR training.
    device:
        ``torch.device`` to load the model onto.

    Returns
    -------
    SSL_Transformer
        Model in eval mode with frozen weights.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    """
    import torch
    from models.ssl_transformer import SSL_Transformer

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SimCLR checkpoint: {checkpoint_path}")

    model = SSL_Transformer(**parse_arch_from_name(checkpoint_path.name))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def encode_channel(model, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """Encode a single channel matrix using ``model.backbone.encode``.

    Parameters
    ----------
    model:
        Loaded SSL_Transformer (frozen, eval mode).
    X:
        Array of shape (n_samples, n_timepoints).
    device:
        ``torch.device``.
    batch_size:
        Number of samples per forward pass.

    Returns
    -------
    np.ndarray
        Embeddings of shape (n_samples, d_model).

    Examples
    --------
    >>> Z_m = encode_channel(simclr_model, X_m_norm, device)
    >>> Z_g = encode_channel(simclr_model, X_g_norm, device)
    >>> Z_dual = np.concatenate([Z_m, Z_g], axis=1)
    """
    import torch
    from tqdm.auto import tqdm

    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(X), batch_size), desc="SimCLR encode", leave=False):
            batch = (
                torch.from_numpy(X[start : start + batch_size])
                .float()
                .unsqueeze(-1)
                .to(device)
            )
            embeddings.append(model.backbone.encode(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_embeddings(
    X: np.ndarray,
    name: str | None = None,
    random_state: int = 42,
    tsne_perplexity: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute PCA and t-SNE 2-D projections for a feature matrix.

    Perplexity is auto-clamped to ``[5, 30]`` if not specified, preventing
    t-SNE failures on small datasets.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).
    name:
        Display label used in the printed PCA variance line (optional).
    random_state:
        Seed for PCA and t-SNE.
    tsne_perplexity:
        Override the automatic perplexity. Use when you need a fixed value
        across multiple datasets for comparability.

    Returns
    -------
    dict
        ``{"PCA": coords_2d, "t-SNE": coords_2d}`` — each array is
        (n_samples, 2).

    Examples
    --------
    >>> projs = reduce_embeddings(simclr_scaled, name="SimCLR", random_state=42)
    >>> projs["PCA"].shape
    (200, 2)
    """
    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X)
    if name:
        print(f"{name} PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")

    # t-SNE
    if tsne_perplexity is None:
        # perplexity must be < n_samples (sklearn hard constraint).
        # n/3 scales naturally with dataset size; clamp to [min(5, n-1), 30]
        # so tiny datasets don't breach the strict upper bound.
        tsne_perplexity = int(min(30, max(min(5, len(X) - 1), (len(X) - 1) // 3)))

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    tsne_coords = tsne.fit_transform(X)
    return {"PCA": pca_coords, "t-SNE": tsne_coords}


# ── Plotting helpers ──────────────────────────────────────────────────────────

def projection_frame(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    representation: str,
    method: str,
) -> pd.DataFrame:
    """Combine 2-D projection coordinates with cell metadata.

    Parameters
    ----------
    coords:
        Array of shape (n_samples, 2) from :func:`reduce_embeddings`.
    metadata:
        Per-cell metadata DataFrame aligned row-wise with ``coords``.
    representation:
        Name of the feature representation (e.g. ``"SimCLR"``).
    method:
        Projection method name (e.g. ``"t-SNE"``).

    Returns
    -------
    pd.DataFrame
        Copy of ``metadata`` with additional columns:
        ``x``, ``y``, ``representation``, ``method``.

    Examples
    --------
    >>> df = projection_frame(projs["t-SNE"], metadata, "SimCLR", "t-SNE")
    >>> sns.scatterplot(data=df, x="x", y="y", hue="class_name")
    """
    df = metadata.copy()
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["representation"] = representation
    df["method"] = method
    return df
