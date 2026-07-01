"""
Shared plotting helpers for IY013 pairwise (one-vs-one) classification results.

build_accuracy_matrix    — long-format results -> symmetric class x class matrix
plot_pairwise_heatmaps   — variant x classifier grid of accuracy heatmaps
plot_pairwise_summary    — mean +/- std accuracy per classifier per variant
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def build_accuracy_matrix(results_df, classes: list[str], classifier: str, dataset_variant: str) -> np.ndarray:
    """Symmetric class x class accuracy matrix (NaN on the diagonal and missing pairs)."""
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.full((n, n), np.nan)
    sub = results_df[
        (results_df["Classifier"] == classifier) & (results_df["Dataset"] == dataset_variant)
    ]
    for _, row in sub.iterrows():
        i, j = idx[row["Class_A"]], idx[row["Class_B"]]
        mat[i, j] = row["Accuracy"]
        mat[j, i] = row["Accuracy"]
    return mat


def plot_pairwise_heatmaps(results_df, classes: list[str], variants: list[str], classifiers: list[str],
                            title: str, fig_path, annot_fontsize: int = 8):
    """variants x classifiers grid of pairwise-accuracy heatmaps, shared 0.5-1.0 colour scale."""
    n_rows, n_cols = len(variants), len(classifiers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    mask_diag = np.eye(len(classes), dtype=bool)

    last_im = None
    for r, variant in enumerate(variants):
        for c, clf in enumerate(classifiers):
            ax = axes[r, c]
            mat = build_accuracy_matrix(results_df, classes, clf, variant)
            last_im = sns.heatmap(
                mat, mask=mask_diag, ax=ax, vmin=0.5, vmax=1.0, cmap="viridis",
                annot=True, fmt=".2f", annot_kws={"fontsize": annot_fontsize},
                xticklabels=classes, yticklabels=classes, cbar=(c == n_cols - 1),
                cbar_kws={"label": "Test accuracy"} if c == n_cols - 1 else None,
                square=True,
            )
            ax.set_title(f"{clf} -- {variant}", fontsize=14)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=9)
            for label in ax.get_xticklabels():
                label.set_ha("right")

    fig.suptitle(title, fontsize=14)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    return fig


def plot_pairwise_summary(results_df, variants: list[str], classifiers: list[str], title: str, fig_path):
    """Grouped bar chart of mean +/- std accuracy (across all pairs) per classifier per variant."""
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    x = np.arange(len(variants))
    width = 0.8 / len(classifiers)
    for i, clf in enumerate(classifiers):
        means, stds = [], []
        for variant in variants:
            sub = results_df[(results_df["Classifier"] == clf) & (results_df["Dataset"] == variant)]["Accuracy"]
            means.append(sub.mean())
            stds.append(sub.std())
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4, label=clf,
                       color=palette[i], edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.text(0.02, 0.96, "Chance = 0.50", transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="grey")
    ax.set_xticks(x + width * (len(classifiers) - 1) / 2)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dataset variant")
    ax.set_ylabel("Mean pairwise test accuracy (+/- std across pairs)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    return fig
