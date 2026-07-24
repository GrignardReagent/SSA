#!/usr/bin/python3
"""
Integration tests for pytorch_metric_learning.losses.SupConLoss as used in
EXP-26-IY036 (IY036_supcon_allclass_training.py, IY036_msn2_cross_session.py).

This is a third-party, independently-tested library, so these tests do not
re-derive the SupCon math (unlike test_cross_view_info_nce.py's manual
reference). They instead pin down the two properties the IY036 call sites
actually rely on:

  1. test_matches_repo_call_convention
       SupConLoss(temperature=0.07) runs on input built exactly the way both
       IY036 scripts build it -- two projected views concatenated along the
       batch dim, labels concatenated the same way -- and returns a finite,
       positive, differentiable scalar.

  2. test_all_unique_labels_returns_zero_not_nan
       When every anchor in a batch has zero same-label partners (no
       positives to pull towards), the loss must be exactly 0, not NaN. This
       is the edge case IY036_msn2_cross_session.py's smaller 3-class batches
       are most likely to hit, and is the one behavioural property
       (AvgNonZeroReducer skipping zero-positive anchors) worth locking down
       with a test rather than trusting silently.
"""

import torch

from pytorch_metric_learning.losses import SupConLoss


def test_matches_repo_call_convention():
    """SupConLoss on IY036-shaped (2-view, concatenated) input must be a
    finite, positive, differentiable scalar."""
    torch.manual_seed(0)
    B, D, n_classes = 16, 16, 4
    criterion = SupConLoss(temperature=0.07)

    z1 = torch.randn(B, D, requires_grad=True)
    z2 = torch.randn(B, D, requires_grad=True)
    yb = torch.randint(0, n_classes, (B,))

    feats = torch.cat([z1, z2], dim=0)
    labels = torch.cat([yb, yb], dim=0)

    loss = criterion(feats, labels)

    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
    assert loss.item() > 0, f"Loss must be positive, got {loss.item()}"

    loss.backward()
    assert z1.grad is not None and z2.grad is not None, \
        "Both views must receive gradients after backward()"
    assert not torch.all(z1.grad == 0), "z1 gradients must be non-zero"
    assert not torch.all(z2.grad == 0), "z2 gradients must be non-zero"


def test_all_unique_labels_returns_zero_not_nan():
    """A batch where every sample has a unique label has no positives for any
    anchor. SupConLoss must return 0 (via AvgNonZeroReducer skipping
    zero-positive anchors), never NaN."""
    torch.manual_seed(1)
    criterion = SupConLoss(temperature=0.07)

    feats = torch.randn(6, 16, requires_grad=True)
    labels = torch.arange(6)  # every label unique -> no same-label partners

    loss = criterion(feats, labels)

    assert torch.isfinite(loss), f"Loss must not be NaN/inf, got {loss.item()}"
    assert loss.item() == 0.0, f"Expected exactly 0 for an all-unique-label batch, got {loss.item()}"
