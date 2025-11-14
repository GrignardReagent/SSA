#!/usr/bin/python3

import torch
import torch.nn as nn
from training.eval import _compute_loss_and_accuracy

'''
Modular test: _compute_loss_and_accuracy function used for evaluate_model()
'''

# ----------------------------------------------------------------------
# 1) BCEWITHLOGITSLOSS — SINGLE LOGIT CASE
# ----------------------------------------------------------------------
def test_bcewithlogits_single_logit():
    outputs = torch.tensor([[2.0], [-1.5], [0.3]])   # logits
    targets = torch.tensor([1, 0, 1])                # ints
    loss_fn = nn.BCEWithLogitsLoss()

    loss, preds, tgt = _compute_loss_and_accuracy(outputs, targets, loss_fn)

    # expected behaviour
    expected_preds = torch.tensor([1, 0, 1])         # logits > 0 means class=1
    assert torch.equal(preds, expected_preds)
    assert torch.equal(tgt, targets)
    assert loss.item() > 0


# ----------------------------------------------------------------------
# 2) BCEWITHLOGITSLOSS — TWO LOGITS (N,2)
# ----------------------------------------------------------------------
def test_bcewithlogits_two_logits():
    outputs = torch.tensor([[0.1, 1.5], [1.2, -0.7]])   # class0 logit, class1 logit
    targets = torch.tensor([1, 0])
    loss_fn = nn.BCEWithLogitsLoss()

    loss, preds, tgt = _compute_loss_and_accuracy(outputs, targets, loss_fn)

    # positive-class logits: [1.5, -0.7] → preds = [1, 0]
    expected_preds = torch.tensor([1, 0])
    assert torch.equal(preds, expected_preds)
    assert torch.equal(tgt, targets)
    assert loss.item() > 0


# ----------------------------------------------------------------------
# 3) BCELoss — SINGLE PROBABILITY CASE
# ----------------------------------------------------------------------
def test_bce_single_probability():
    probs = torch.tensor([[0.8], [0.2], [0.6]])         # already probabilities
    targets = torch.tensor([1, 0, 1])
    loss_fn = nn.BCELoss()

    loss, preds, tgt = _compute_loss_and_accuracy(probs, targets, loss_fn)

    expected_preds = torch.tensor([1, 0, 1])             # threshold at 0.5
    assert torch.equal(preds, expected_preds)
    assert torch.equal(tgt, targets)
    assert loss.item() > 0


# ----------------------------------------------------------------------
# 4) BCELoss — TWO PROBABILITIES (N,2) (softmax-like)
# ----------------------------------------------------------------------
def test_bce_two_probabilities():
    outputs = torch.tensor([[0.3, 0.7], [0.9, 0.1]])      # pseudo-softmax probs
    targets = torch.tensor([1, 0])
    loss_fn = nn.BCELoss()

    loss, preds, tgt = _compute_loss_and_accuracy(outputs, targets, loss_fn)

    # Positive class prob is column 1 → probs = [0.7, 0.1]
    expected_preds = torch.tensor([1, 0])
    assert torch.equal(preds, expected_preds)
    assert torch.equal(tgt, targets)
    assert loss.item() > 0


# ----------------------------------------------------------------------
# 5) CrossEntropyLoss (multiclass)
# ----------------------------------------------------------------------
def test_crossentropy():
    outputs = torch.tensor([[0.5, 1.2, -0.3],  # logits for 3 classes
                            [2.0, 0.1, -0.5]])
    targets = torch.tensor([1, 0])
    loss_fn = nn.CrossEntropyLoss()

    loss, preds, tgt = _compute_loss_and_accuracy(outputs, targets, loss_fn)

    expected_preds = outputs.argmax(dim=1)
    assert torch.equal(preds, expected_preds)
    assert torch.equal(tgt, targets)
    assert loss.item() > 0


# ----------------------------------------------------------------------
# 6) BCEWithLogitsLoss — Output shape consistency
# ----------------------------------------------------------------------
def test_bcewithlogits_shapes():
    outputs = torch.randn(4, 1)    # valid (N,1)
    targets = torch.randint(0, 2, (4,))
    loss_fn = nn.BCEWithLogitsLoss()

    loss, preds, tgt = _compute_loss_and_accuracy(outputs, targets, loss_fn)
    assert preds.shape == (4,)
    assert tgt.shape == (4,)


# ----------------------------------------------------------------------
# 7) BCELoss — Error on invalid shape
# ----------------------------------------------------------------------
def test_bce_invalid_shape():
    outputs = torch.randn(4, 3)    # invalid for BCELoss
    targets = torch.randint(0, 2, (4,))
    loss_fn = nn.BCELoss()

    try:
        _compute_loss_and_accuracy(outputs, targets, loss_fn)
        assert False, "Should raise ValueError"
    except ValueError:
        pass