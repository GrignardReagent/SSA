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


def make_checkpoint_short_label(
    path: str | Path,
    norm_overrides: dict[str, str] | None = None,
) -> str:
    """Create a compact SimCLR checkpoint label for tables and plots.

    The label encodes experiment id, normalisation strategy, batch size and
    embedding width when these are present in the checkpoint filename.
    """
    path = Path(path)
    stem = path.stem
    exp_match = re.search(r"IY\d+", path.parts[-2] if len(path.parts) >= 2 else stem)
    exp_id = exp_match.group() if exp_match else ""

    batch_match = re.search(r"_b(\d+)", stem)
    batch = f"b{batch_match.group(1)}" if batch_match else ""
    dim_match = re.search(r"_D(\d+)", stem)
    dim = f"D{dim_match.group(1)}" if dim_match else ""

    norm_overrides = norm_overrides or {}
    if "batch-wise" in stem:
        norm = "batch"
    elif "global" in stem:
        norm = "global"
    elif "joint" in stem:
        norm = "joint"
    elif "mixed" in stem:
        norm = "mixed"
    elif stem in norm_overrides:
        norm = norm_overrides[stem]
    else:
        norm = "inst"

    return f"{exp_id}-{norm} {batch} {dim}".strip()


def make_unique_checkpoint_label(
    path: str | Path,
    index: int,
    norm_overrides: dict[str, str] | None = None,
) -> str:
    """Append a timestamp/index suffix so duplicate checkpoint labels stay unique."""
    path = Path(path)
    timestamp = re.search(r"_(20\d{6}_\d{6})_model$", path.stem)
    suffix = timestamp.group(1)[9:] if timestamp else f"#{index:02d}"
    return f"{make_checkpoint_short_label(path, norm_overrides)} {suffix}"


def make_checkpoint_arch_label(
    path: str | Path,
    norm_overrides: dict[str, str] | None = None,
) -> str:
    """Create a detailed checkpoint label with architecture and timestamp."""
    path = Path(path)
    stem = path.stem
    exp_match = re.search(r"(IY\d+)", stem)
    exp_id = exp_match.group(1) if exp_match else ""

    arch_match = re.search(r"(b\d+_lr[\d.]+(?:_d[\d.]+)?_L\d+_H\d+_D\d+)", stem)
    if arch_match:
        arch = arch_match.group(1)
    else:
        fallback = re.search(r"(L\d+_H\d+_D\d+)", stem)
        arch = fallback.group(1) if fallback else ""

    timestamp = re.search(r"(\d{8}_\d{6})", stem)
    timestamp_label = timestamp.group(1)[-6:] if timestamp else ""

    norm_overrides = norm_overrides or {}
    norm = (
        norm_overrides.get(stem)
        or (re.search(r"_(instance|global|joint)_", stem) or [None, ""])[1]
        or "instance"
    )
    return f"{exp_id}-{arch}-{norm}-{timestamp_label}"


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
    phate_knn: int = 5,
) -> dict[str, np.ndarray]:
    """Compute PCA, t-SNE, and PHATE 2-D projections for a feature matrix.

    Perplexity is auto-clamped to ``[5, 30]`` if not specified, preventing
    t-SNE failures on small datasets. PHATE requires the ``phate`` package.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).
    name:
        Display label used in the printed PCA variance line (optional).
    random_state:
        Seed for PCA, t-SNE, and PHATE.
    tsne_perplexity:
        Override the automatic perplexity. Use when you need a fixed value
        across multiple datasets for comparability.
    phate_knn:
        Number of nearest neighbours for PHATE graph construction.
        Default 5 works well for datasets of ~500 cells.

    Returns
    -------
    dict
        ``{"PCA": coords_2d, "t-SNE": coords_2d, "PHATE": coords_2d}`` —
        each array is (n_samples, 2).

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

    # PHATE — heat-diffusion manifold embedding; good at preserving both local
    # and global trajectory structure in biological data.
    import phate
    phate_op = phate.PHATE(
        n_components=2,
        knn=phate_knn,
        random_state=random_state,
        verbose=0,
    )
    phate_coords = phate_op.fit_transform(X)

    return {"PCA": pca_coords, "t-SNE": tsne_coords, "PHATE": phate_coords}


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
