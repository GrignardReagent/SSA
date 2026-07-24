#!/usr/bin/python3
"""
Lightweight regression tests for training.train.train_supcon_model, the
shared SupCon training loop added for EXP-26-IY036
(IY036_supcon_allclass_training.py, IY036_msn2_cross_session.py).

CPU-only, tiny model/dataset, few epochs -- these check loop plumbing
(history shape, val-loss gating, checkpoint save/load, eval_fn cadence,
wandb-off path, required-arg guards), not training quality.
"""

import tempfile
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning.losses import SupConLoss

from models.ssl_transformer import SSL_Transformer
from training.train import train_supcon_model

D_MODEL, N_CLASSES, N_CELLS, SEQ_LEN = 8, 4, 32, 20


class _CellDataset(Dataset):
    """Synthetic (trace, label) cells -- same shape convention as IY036's CellDataset."""

    def __init__(self, n=N_CELLS, n_classes=N_CLASSES, seq_len=SEQ_LEN, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, seq_len, generator=g)
        self.y = torch.randint(0, n_classes, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i].unsqueeze(-1), int(self.y[i])


def _augment(x):
    return x + torch.randn_like(x) * 0.05


def _make_model():
    torch.manual_seed(0)
    return SSL_Transformer(input_size=1, d_model=D_MODEL, nhead=2, num_layers=1, dropout=0.0)


def _make_loader(seed=0, batch_size=8):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(_CellDataset(seed=seed), batch_size=batch_size, shuffle=True, generator=gen)


def test_history_dict_shape_and_finite_loss():
    model = _make_model()
    history = train_supcon_model(
        model, _make_loader(), epochs=8, loss_fn=SupConLoss(temperature=0.07),
        augment_fn=_augment, grad_clip=None, verbose=False,
    )
    for key in ("epoch", "train_loss", "val_loss", "lr"):
        assert key in history, f"missing history key: {key}"
    assert len(history["train_loss"]) == 8
    assert all(v == v and abs(v) != float("inf") for v in history["train_loss"]), \
        "train_loss must be finite (no NaN/inf)"
    # loose decreasing check: mean of the last 2 epochs should not be worse than the first 2
    assert sum(history["train_loss"][-2:]) <= sum(history["train_loss"][:2]) + 1.0


def test_val_loss_present_iff_val_loader_given():
    model_no_val = _make_model()
    history_no_val = train_supcon_model(
        model_no_val, _make_loader(), epochs=3, loss_fn=SupConLoss(temperature=0.07),
        augment_fn=_augment, grad_clip=None, verbose=False,
    )
    assert all(v is None for v in history_no_val["val_loss"])

    model_val = _make_model()
    history_val = train_supcon_model(
        model_val, _make_loader(seed=0), val_loader=_make_loader(seed=1), epochs=3,
        loss_fn=SupConLoss(temperature=0.07), augment_fn=_augment, grad_clip=None, verbose=False,
    )
    assert all(v is not None and v == v for v in history_val["val_loss"])


def test_checkpoint_saved_and_loadable():
    model = _make_model()
    calls = {"n": 0}

    def eval_fn(m):
        calls["n"] += 1
        return {"acc": float(calls["n"])}  # monotonically improving -> saves every eval

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "ckpt.pth"
        train_supcon_model(
            model, _make_loader(), epochs=6, loss_fn=SupConLoss(temperature=0.07),
            augment_fn=_augment, grad_clip=None, save_path=str(save_path),
            eval_fn=eval_fn, eval_every=2, eval_metric_key="acc", verbose=False,
        )
        assert save_path.exists(), "checkpoint file was not saved"
        fresh = _make_model()
        fresh.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))


def test_eval_fn_called_at_expected_epochs():
    model = _make_model()
    history = train_supcon_model(
        model, _make_loader(), epochs=6, loss_fn=SupConLoss(temperature=0.07),
        augment_fn=_augment, grad_clip=None,
        eval_fn=lambda m: {"acc": 0.5}, eval_every=2, eval_metric_key="acc", verbose=False,
    )
    assert history["eval_epoch"] == [2, 4, 6]
    assert history["eval/acc"] == [0.5, 0.5, 0.5]


def test_wandb_off_path_runs_without_wandb():
    model = _make_model()
    history = train_supcon_model(
        model, _make_loader(), epochs=2, loss_fn=SupConLoss(temperature=0.07),
        augment_fn=_augment, grad_clip=None, wandb_logging=False, verbose=False,
    )
    assert len(history["train_loss"]) == 2


def test_missing_augment_fn_or_loss_fn_raises():
    model = _make_model()
    try:
        train_supcon_model(model, _make_loader(), epochs=1, augment_fn=_augment, verbose=False)
        assert False, "expected ValueError for missing loss_fn"
    except ValueError:
        pass

    try:
        train_supcon_model(model, _make_loader(), epochs=1, loss_fn=SupConLoss(temperature=0.07), verbose=False)
        assert False, "expected ValueError for missing augment_fn"
    except ValueError:
        pass
