#!/usr/bin/python3
"""
Lightweight regression tests for training.lars.LARS, added for the Eddie
variant of EXP-26-IY036's SupCon pretraining (large-batch training, matching
Khosla et al.'s optimizer choice).
"""

import torch
import torch.nn as nn

from training.lars import LARS


def test_loss_decreases_on_toy_problem():
    """LARS should behave like a working optimizer on an ordinary problem:
    loss trends down over a handful of steps on a tiny MLP."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = LARS(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(l == l for l in losses), "loss must never be NaN"
    assert sum(losses[-3:]) < sum(losses[:3]), \
        "loss should trend down over training, not up"


def test_1d_params_get_plain_sgd_no_trust_ratio():
    """Biases (1-D params) must NOT get LARS's per-layer trust-ratio scaling --
    only plain momentum-SGD with local_lr=1.0, per standard LARS practice."""
    torch.manual_seed(1)
    bias = nn.Parameter(torch.ones(4) * 10.0)  # large weight norm
    opt = LARS([bias], lr=0.5, momentum=0.0, weight_decay=0.0)

    bias.grad = torch.ones(4) * 0.01  # tiny gradient norm -> would get a huge trust ratio if scaled
    opt.step()

    # With local_lr=1.0 and momentum=0, the update is exactly lr * grad.
    expected = 10.0 - 0.5 * 0.01
    assert torch.allclose(bias, torch.full((4,), expected), atol=1e-6)


def test_local_lr_matches_manual_trust_ratio_for_2d_param():
    """For a >=2D param, the applied step must match the paper's trust-ratio
    formula computed by hand, for a single momentum=0 step."""
    torch.manual_seed(2)
    w = nn.Parameter(torch.eye(4) * 2.0)  # known norm
    opt = LARS([w], lr=1.0, momentum=0.0, weight_decay=0.0, trust_coefficient=0.01, eps=1e-8)

    grad = torch.ones(4, 4) * 0.1
    w.grad = grad.clone()

    w_norm = torch.norm(torch.eye(4) * 2.0)
    g_norm = torch.norm(grad)
    expected_local_lr = (0.01 * w_norm / (g_norm + 1e-8)).item()
    expected_update = expected_local_lr * grad  # lr=1.0, momentum=0 -> buf = local_lr * grad

    w_before = torch.eye(4) * 2.0
    opt.step()
    assert torch.allclose(w_before - w, expected_update, atol=1e-5)


def test_missing_grad_is_skipped():
    """Parameters with grad=None (e.g. frozen layers) must be left untouched."""
    p1 = nn.Parameter(torch.randn(3, 3))
    p2 = nn.Parameter(torch.randn(3, 3))
    opt = LARS([p1, p2], lr=0.1)
    p1.grad = torch.randn(3, 3)
    # p2.grad stays None
    p2_before = p2.clone()
    opt.step()
    assert torch.equal(p2, p2_before), "params without a gradient must not change"
