"""
Tests that batch_norm_collate_fn never produces NaN, and that the downstream
InfoNCE loss is also finite, under the two conditions that previously caused NaN:

  1. Identical views (single-trajectory file, val mode before the fix)
  2. Small last batch (no drop_last on val/test loaders)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn.functional as F
from info_nce import InfoNCE

from dataloaders.simclr import batch_norm_collate_fn
from models.ssl_transformer import SSL_Transformer
from training.train import cross_view_info_nce


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_batch(batch_size, time_steps=500, channels=1):
    """Return a list of (z1, z2, y) tuples as the Dataset would yield."""
    z1 = torch.randn(batch_size, time_steps, channels)
    z2 = torch.randn(batch_size, time_steps, channels)
    y  = torch.zeros(batch_size, 1)
    return list(zip(z1, z2, y))


def _make_identical_batch(batch_size, time_steps=500, channels=1):
    """Both views are the same tensor — reproduces the single-trajectory val bug."""
    traj = torch.randn(batch_size, time_steps, channels)
    y    = torch.zeros(batch_size, 1)
    return list(zip(traj, traj.clone(), y))  # v1 == v2 for every sample


def _loss_from_embeddings(z1, z2, temperature=0.2):
    loss_fn = InfoNCE(negative_mode='unpaired', temperature=temperature)
    return loss_fn(z1, z2)


# ── collate tests ─────────────────────────────────────────────────────────────

def test_collate_no_nan_normal_batch():
    """Standard batch: both views differ → collate should be finite."""
    batch = _make_batch(64)
    z1, z2, _ = batch_norm_collate_fn(batch)

    assert not torch.isnan(z1).any(), "z1 contains NaN for a normal batch"
    assert not torch.isnan(z2).any(), "z2 contains NaN for a normal batch"
    assert not torch.isinf(z1).any(), "z1 contains Inf for a normal batch"
    assert not torch.isinf(z2).any(), "z2 contains Inf for a normal batch"
    print("✅ test_collate_no_nan_normal_batch PASSED")


def test_collate_no_nan_identical_views():
    """
    Single-trajectory files in val mode produce identical views.
    Before the fix, std=0 → division-by-zero → NaN everywhere.
    After the fix (.clamp(min=1e-8)), output must be finite.
    """
    batch = _make_identical_batch(64)
    z1, z2, _ = batch_norm_collate_fn(batch)

    assert not torch.isnan(z1).any(), "z1 contains NaN when views are identical"
    assert not torch.isnan(z2).any(), "z2 contains NaN when views are identical"
    assert not torch.isinf(z1).any(), "z1 contains Inf when views are identical"
    assert not torch.isinf(z2).any(), "z2 contains Inf when views are identical"
    print("✅ test_collate_no_nan_identical_views PASSED")


def test_collate_no_nan_small_batch():
    """Last-batch with a single sample (no drop_last on val loader)."""
    batch = _make_batch(1)
    z1, z2, _ = batch_norm_collate_fn(batch)

    assert not torch.isnan(z1).any(), "z1 contains NaN for a batch of size 1"
    assert not torch.isnan(z2).any(), "z2 contains NaN for a batch of size 1"
    print("✅ test_collate_no_nan_small_batch PASSED")


# ── end-to-end loss tests ─────────────────────────────────────────────────────

def test_loss_not_nan_after_collate_normal():
    """Collate + InfoNCE: loss must be finite for a normal batch."""
    batch = _make_batch(64)
    z1_raw, z2_raw, _ = batch_norm_collate_fn(batch)

    model = SSL_Transformer(input_size=1, d_model=8, nhead=4, num_layers=2)
    model.eval()
    with torch.no_grad():
        z1, z2 = model(z1_raw, z2_raw)

    loss = _loss_from_embeddings(z1, z2)
    assert not torch.isnan(loss), f"InfoNCE loss is NaN for a normal batch (got {loss})"
    assert not torch.isinf(loss), f"InfoNCE loss is Inf for a normal batch (got {loss})"
    print(f"✅ test_loss_not_nan_after_collate_normal PASSED  (loss={loss.item():.4f})")


def test_loss_not_nan_after_collate_identical_views():
    """
    Collate + InfoNCE with identical views: loss must be finite after the fix.
    This is the exact scenario that caused val_loss=nan throughout training.
    """
    batch = _make_identical_batch(64)
    z1_raw, z2_raw, _ = batch_norm_collate_fn(batch)

    model = SSL_Transformer(input_size=1, d_model=8, nhead=4, num_layers=2)
    model.eval()
    with torch.no_grad():
        z1, z2 = model(z1_raw, z2_raw)

    loss = _loss_from_embeddings(z1, z2)
    assert not torch.isnan(loss), f"InfoNCE loss is NaN with identical views (got {loss})"
    assert not torch.isinf(loss), f"InfoNCE loss is Inf with identical views (got {loss})"
    print(f"✅ test_loss_not_nan_after_collate_identical_views PASSED  (loss={loss.item():.4f})")


def test_cross_view_loss_not_nan_identical_views():
    """Same check for the cross_view_info_nce loss used as the default."""
    batch = _make_identical_batch(64)
    z1_raw, z2_raw, _ = batch_norm_collate_fn(batch)

    model = SSL_Transformer(input_size=1, d_model=8, nhead=4, num_layers=2)
    model.eval()
    with torch.no_grad():
        z1, z2 = model(z1_raw, z2_raw)

    loss = cross_view_info_nce(z1, z2, temperature=0.2)
    assert not torch.isnan(loss), f"cross_view_info_nce loss is NaN with identical views (got {loss})"
    assert not torch.isinf(loss), f"cross_view_info_nce loss is Inf with identical views (got {loss})"
    print(f"✅ test_cross_view_loss_not_nan_identical_views PASSED  (loss={loss.item():.4f})")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== batch_norm_collate_fn NaN regression tests ===\n")

    test_collate_no_nan_normal_batch()
    test_collate_no_nan_identical_views()
    test_collate_no_nan_small_batch()
    test_loss_not_nan_after_collate_normal()
    test_loss_not_nan_after_collate_identical_views()
    test_cross_view_loss_not_nan_identical_views()

    print("\nAll tests passed.")
