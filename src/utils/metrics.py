"""
Embedding quality metrics for evaluating learned representations.

Two complementary D-score definitions are provided — choose based on context:

  discriminability_d_score  (pairwise-distance Cohen's d)
    Use when: multiclass data, or when you want a full breakdown of within- vs
    between-class distance distributions. O(n²) — avoid on very large datasets.

  fisher_d_score  (Fisher linear discriminant ratio)
    Use when: binary classification only, and speed matters (e.g. per-epoch or
    per-panel monitoring during sweeps). O(n·d) — fast even at scale.

Also provides clustering agreement metrics (ARI, NMI).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)


# ── D-score variants ──────────────────────────────────────────────────────────

def discriminability_d_score(X: np.ndarray, labels: np.ndarray) -> dict:
    """Pairwise-distance Cohen's d separability score.

    Use this for multiclass data or when you want the full distance breakdown.
    O(n²) — prefer fisher_d_score for fast binary monitoring.

    Computes the pooled-SD-normalised difference between mean between-class and
    mean within-class pairwise Euclidean distances (upper-triangle pairs only).

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).
    labels:
        Integer or string class labels (n_samples,).

    Returns
    -------
    dict with keys:
        D_score, mean_within_distance, mean_between_distance,
        pooled_distance_sd, n_within_pairs, n_between_pairs.

    Raises
    ------
    ValueError
        If fewer than two classes or no within/between pairs exist.

    Examples
    --------
    >>> from utils.metrics import discriminability_d_score
    >>> X = np.random.randn(100, 16)
    >>> y = np.array([0] * 50 + [1] * 50)
    >>> result = discriminability_d_score(X, y)
    >>> "D_score" in result
    True
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")
    if np.unique(labels).size < 2:
        raise ValueError("At least two classes are required.")

    distances = pairwise_distances(X, metric="euclidean")
    # Upper triangle (k=1) gives each pair exactly once and drops the zero diagonal.
    upper = np.triu_indices_from(distances, k=1)
    pair_dist = distances[upper]
    same_class = labels[upper[0]] == labels[upper[1]]
    within = pair_dist[same_class]    # distances between same-class pairs
    between = pair_dist[~same_class]  # distances between different-class pairs

    if within.size == 0 or between.size == 0:
        raise ValueError("Both within-class and between-class pairs are required.")

    pooled_sd = np.sqrt(0.5 * (within.var(ddof=1) + between.var(ddof=1)))
    d = np.nan if pooled_sd == 0 else (between.mean() - within.mean()) / pooled_sd
    return {
        "D_score": d,
        "mean_within_distance": float(within.mean()),
        "mean_between_distance": float(between.mean()),
        "pooled_distance_sd": float(pooled_sd),
        "n_within_pairs": int(within.size),
        "n_between_pairs": int(between.size),
    }


def fisher_d_score(emb: np.ndarray, y: np.ndarray) -> float:
    """Fisher multivariate discriminability: ||μ₁−μ₀||² / (tr(Σ₁)+tr(Σ₀)).

    Binary labels only (0/1). O(n·d) — use this for fast per-epoch or per-panel
    monitoring. For multiclass or a full distance breakdown, use
    discriminability_d_score instead.

    Parameters
    ----------
    emb:
        Embedding matrix (n_samples, n_dims).
    y:
        Binary labels (0 / 1).

    Returns
    -------
    float
        Fisher D-score (≥ 0; higher is more separable).
    """
    mu1 = emb[y == 1].mean(0)
    mu0 = emb[y == 0].mean(0)
    between = float(np.linalg.norm(mu1 - mu0) ** 2)
    within = float(emb[y == 1].var(0).sum() + emb[y == 0].var(0).sum())
    return between / (within + 1e-12)


# ── Clustering agreement ──────────────────────────────────────────────────────

def clustering_agreement_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> dict:
    """Cluster embeddings with KMeans and compare to true labels.

    Uses k equal to the number of unique true classes. Scores unsupervised
    cluster assignments with Adjusted Rand Index and Normalised Mutual Info.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).
    labels:
        True class labels (n_samples,).
    random_state:
        KMeans seed.

    Returns
    -------
    dict with keys:
        ARI, NMI, n_clusters, kmeans_inertia.

    Raises
    ------
    ValueError
        If fewer than two unique classes.

    Examples
    --------
    >>> from utils.metrics import clustering_agreement_metrics
    >>> metrics = clustering_agreement_metrics(X_scaled, y)
    >>> metrics["ARI"]  # 1.0 = perfect agreement
    """
    labels = np.asarray(labels)
    n_clusters = np.unique(labels).size
    if n_clusters < 2:
        raise ValueError("At least two classes are required.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X)

    return {
        "ARI": float(adjusted_rand_score(labels, cluster_labels)),
        "NMI": float(normalized_mutual_info_score(labels, cluster_labels)),
        "n_clusters": n_clusters,
        "kmeans_inertia": float(kmeans.inertia_),
    }
