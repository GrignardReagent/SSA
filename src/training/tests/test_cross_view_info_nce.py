#!/usr/bin/python3
"""
Unit tests for cross_view_info_nce() defined in training.train.

Four tests, each targeting a distinct correctness property:

  1. test_output_is_scalar_and_finite
       Basic sanity: the function returns a finite positive scalar.

  2. test_matches_manual_reference
       Numerical agreement with an independent reference implementation.
       Validates that the concatenation layout and mask positions are correct.

  3. test_self_comparisons_are_excluded
       The key behavioural test.  For L2-normalised embeddings sim(z_i, z_i) = 1,
       so unmasked self terms contribute exp(1/τ) to the softmax denominator —
       a constant that inflates the loss without providing any gradient signal.
       After masking, the denominator is strictly smaller, so the masked loss
       must be strictly lower than the unmasked loss.

  4. test_gradients_flow_through_both_views
       Confirms that loss.backward() propagates non-zero gradients into both
       input tensors, i.e. the function is end-to-end differentiable.
"""

import torch
import torch.nn.functional as F

from training.train import cross_view_info_nce


# ── reference helpers ─────────────────────────────────────────────────────────

def _manual_cross_view_info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Independent reference implementation.

    Builds the (2B, 2B) logit matrix identically to cross_view_info_nce(),
    applies the self-comparison mask, and calls F.cross_entropy.
    Written separately from the function under test so that test_matches_manual_reference
    is a genuine cross-check rather than a tautology.
    """
    B = z1.size(0)
    N = 2 * B

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    all_q = torch.cat([z1, z2], dim=0)   # (2B, E)
    all_k = torch.cat([z2, z1], dim=0)   # (2B, E)

    logits = torch.matmul(all_q, all_k.T) / temperature   # (2B, 2B)

    # Self-comparison positions: query i appears as key at (i + B) % 2B
    self_cols = (torch.arange(N) + B) % N
    logits[torch.arange(N), self_cols] = float('-inf')

    labels = torch.arange(N)
    return F.cross_entropy(logits, labels)


def _unmasked_cross_view_info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Same as cross_view_info_nce but WITHOUT the self-comparison mask.

    Used in test_self_comparisons_are_excluded to confirm that masking strictly
    reduces the loss (because the self term exp(1/τ) inflates the denominator).
    """
    B = z1.size(0)
    N = 2 * B

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    all_q = torch.cat([z1, z2], dim=0)
    all_k = torch.cat([z2, z1], dim=0)

    logits = torch.matmul(all_q, all_k.T) / temperature
    labels = torch.arange(N)
    return F.cross_entropy(logits, labels)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_output_is_scalar_and_finite():
    """Loss must be a finite, positive scalar (0-dim tensor)."""
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)

    loss = cross_view_info_nce(z1, z2, temperature=0.1)

    assert loss.shape == torch.Size([]), \
        f"Expected scalar (shape []), got shape {loss.shape}"
    assert torch.isfinite(loss), \
        f"Loss must be finite, got {loss.item()}"
    assert loss.item() > 0, \
        f"Loss must be positive, got {loss.item()}"


def test_matches_manual_reference():
    """
    Numerical agreement with the independent reference implementation.

    Checks that the concatenation layout (all_q = [z1; z2], all_k = [z2; z1])
    and the self-mask column formula ((i + B) % 2B) are both correct.
    """
    torch.manual_seed(42)
    B, E = 6, 32
    temperature = 0.2

    z1 = torch.randn(B, E)
    z2 = torch.randn(B, E)

    expected = _manual_cross_view_info_nce(z1.clone(), z2.clone(), temperature)
    actual   = cross_view_info_nce(z1.clone(), z2.clone(), temperature)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Loss mismatch: expected {expected.item():.6f}, got {actual.item():.6f}"
    )


def test_self_comparisons_are_excluded():
    """
    Self-comparison positions must be masked out of the softmax denominator.

    For L2-normalised embeddings, sim(z_i, z_i) = 1 always, so the unmasked
    self term contributes exp(1/τ) to the denominator — at τ=0.2 that is
    exp(5) ≈ 148, equivalent to ~90 average negatives.  The masked version
    excludes these positions via -inf, making its denominator strictly smaller
    and therefore its loss strictly lower.
    """
    torch.manual_seed(7)
    B, E = 8, 64
    temperature = 0.2   # small τ maximises the exp(1/τ) ≈ 148 self-term effect

    z1 = torch.randn(B, E)
    z2 = torch.randn(B, E)

    masked_loss   = cross_view_info_nce(z1, z2, temperature)
    unmasked_loss = _unmasked_cross_view_info_nce(z1.clone(), z2.clone(), temperature)

    assert masked_loss.item() < unmasked_loss.item(), (
        f"Masked loss ({masked_loss.item():.4f}) should be strictly less than "
        f"unmasked loss ({unmasked_loss.item():.4f}): self-comparison terms "
        f"exp(1/τ)=exp({1/temperature:.1f})≈{(1/temperature).__class__.__name__} "
        f"inflate the denominator when not masked."
    )


def test_gradients_flow_through_both_views():
    """
    loss.backward() must produce non-zero gradients in both z1 and z2.

    This confirms the function is end-to-end differentiable and that neither
    view is inadvertently detached from the computation graph.
    """
    torch.manual_seed(3)
    z1 = torch.randn(4, 16, requires_grad=True)
    z2 = torch.randn(4, 16, requires_grad=True)

    loss = cross_view_info_nce(z1, z2, temperature=0.1)
    loss.backward()

    assert z1.grad is not None, "z1 must receive gradients after backward()"
    assert z2.grad is not None, "z2 must receive gradients after backward()"
    assert not torch.all(z1.grad == 0), "z1 gradients must be non-zero"
    assert not torch.all(z2.grad == 0), "z2 gradients must be non-zero"
