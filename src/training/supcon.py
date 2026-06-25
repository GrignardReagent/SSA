"""
Supervised Contrastive (SupCon) loss.

Reference
---------
Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn.functional as F


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive loss.

    For each anchor i, positives are all other samples j with the same class
    label. The loss pushes anchors toward their positives and away from all
    other samples, directly optimising the embedding geometry exploited by
    downstream linear / SVM classifiers.

    This is the SupCon loss from Khosla et al. (2020), Equation 3:

        L_sup = Σ_i  -1/|P(i)| * Σ_{p ∈ P(i)}
                log[ exp(z_i · z_p / τ) / Σ_{a ≠ i} exp(z_i · z_a / τ) ]

    where P(i) is the set of indices with the same label as i (excluding i
    itself), and τ is the temperature.

    Parameters
    ----------
    features : (N, D) float tensor
        Projected embeddings from the model's projection head.
        L2-normalisation is applied internally.
    labels : (N,) integer tensor
        Class label for each embedding.
    temperature : float
        Softmax temperature τ (default 0.07, as in the SupCon paper).

    Returns
    -------
    Scalar loss tensor with gradients attached to ``features``.

    Notes
    -----
    - Anchors that have no same-class partner in the batch are skipped
      (their contribution would be undefined / NaN).
    - Self-comparison (anchor appearing as its own key) is excluded with a
      -inf mask before the softmax denominator, matching the InfoNCE convention
      used in ``cross_view_info_nce``.
    """
    N = features.size(0)
    device = features.device

    # L2 normalise: cosine similarity == dot product for unit vectors.
    features = F.normalize(features, dim=-1)  # (N, D)

    # Pairwise cosine-similarity matrix scaled by temperature: (N, N)
    logits = torch.matmul(features, features.T) / temperature

    # --- Self-comparison mask ---
    # Exclude diagonal entries so an anchor cannot act as its own positive
    # or inflate the softmax denominator via sim(z, z) = 1.
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    # --- Positive mask ---
    # True at position (i, j) when labels[i] == labels[j] AND i != j.
    labels_col = labels.unsqueeze(0)  # (1, N)
    labels_row = labels.unsqueeze(1)  # (N, 1)
    pos_mask = (labels_row == labels_col) & ~self_mask  # (N, N) bool

    num_positives = pos_mask.sum(dim=1).float()  # (N,)

    # Anchors without a same-class partner in the batch are excluded.
    # (This can happen for rare classes or very small batches.)
    valid = num_positives > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Stable log-sum-exp over all non-self entries (−inf positions are
    # excluded by logsumexp, since exp(−inf) = 0).
    log_denom = torch.logsumexp(logits, dim=1)  # (N,)

    # log p(j | i) = sim(i, j)/τ − log Σ_{a≠i} exp(sim(i,a)/τ)
    log_probs = logits - log_denom.unsqueeze(1)  # (N, N)

    # Sum log-probs over positive pairs.
    # IMPORTANT: use masked_fill instead of multiplying by pos_mask.float() to
    # avoid -inf × 0 = NaN at self-comparison positions (the diagonal was set
    # to -inf, so log_probs contains -inf there, and -inf * 0.0 = NaN in
    # IEEE 754 floating-point arithmetic).
    pos_log_probs = log_probs.masked_fill(~pos_mask, 0.0).sum(dim=1)  # (N,)
    loss = -(pos_log_probs[valid] / num_positives[valid]).mean()
    return loss
