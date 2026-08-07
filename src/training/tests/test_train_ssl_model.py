#!/usr/bin/python3
"""
Lightweight regression tests for training.train.train_ssl_model, specifically
the eval_fn/eval_every/eval_metric_key/return_run additions made for
EXP-26-IY039 (downstream-KNN-gated early stopping during SimCLR pretraining).

CPU-only, tiny model/dataset, few epochs -- these check loop plumbing
(eval_fn cadence, checkpoint save/load, early-stopping switch-over,
wandb-off path, return_run), not training quality.
"""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from models.ssl_transformer import SSL_Transformer
from training.train import train_ssl_model

D_MODEL, N_CELLS, SEQ_LEN = 8, 32, 20


class _PairDataset(Dataset):
    """Synthetic (X1, X2, y) pairs -- same shape convention as SimCLR_Dataset."""

    def __init__(self, n=N_CELLS, seq_len=SEQ_LEN, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X1 = torch.randn(n, seq_len, 1, generator=g)
        self.X2 = torch.randn(n, seq_len, 1, generator=g)
        self.y = torch.zeros(n)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.y[i]


def _make_model():
    torch.manual_seed(0)
    return SSL_Transformer(input_size=1, d_model=D_MODEL, nhead=2, num_layers=1, dropout=0.0)


def _make_loader(seed=0, batch_size=8):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(_PairDataset(seed=seed), batch_size=batch_size, shuffle=True, generator=gen)


def test_default_path_unchanged_without_eval_fn():
    """No eval_fn given -> falls back to today's contrastive-val_acc gating."""
    model = _make_model()
    history = train_ssl_model(
        model, _make_loader(), val_loader=_make_loader(seed=1), epochs=4,
        grad_clip=None, verbose=False,
    )
    assert "eval_epoch" not in history
    assert len(history["val_acc"]) == 4


def test_checkpoint_saved_and_loadable_via_eval_fn():
    model = _make_model()
    calls = {"n": 0}

    def eval_fn(m):
        calls["n"] += 1
        return {"acc": float(calls["n"])}  # monotonically improving -> saves every eval

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "ckpt.pth"
        train_ssl_model(
            model, _make_loader(), epochs=6, grad_clip=None, save_path=str(save_path),
            eval_fn=eval_fn, eval_every=2, eval_metric_key="acc", verbose=False,
        )
        assert save_path.exists(), "checkpoint file was not saved"
        fresh = _make_model()
        fresh.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))


def test_eval_fn_called_at_expected_epochs():
    model = _make_model()
    history = train_ssl_model(
        model, _make_loader(), epochs=6, grad_clip=None,
        eval_fn=lambda m: {"acc": 0.5}, eval_every=2, eval_metric_key="acc", verbose=False,
    )
    assert history["eval_epoch"] == [2, 4, 6]
    assert history["eval/acc"] == [0.5, 0.5, 0.5]


def test_early_stopping_gated_on_eval_metric_not_contrastive_val_acc():
    """A stalled eval metric must trigger early stopping even if contrastive
    val_acc keeps looking fine -- the whole point of the IY039 change."""
    model = _make_model()
    history = train_ssl_model(
        model, _make_loader(), val_loader=_make_loader(seed=1), epochs=20, patience=2,
        grad_clip=None, eval_fn=lambda m: {"acc": 0.5}, eval_every=1,
        eval_metric_key="acc", verbose=False,
    )
    # acc never improves past the first eval -> stop after patience=2 non-improving evals
    assert len(history["train_loss"]) == 3


def test_missing_eval_metric_key_with_save_path_raises():
    model = _make_model()
    try:
        train_ssl_model(
            model, _make_loader(), epochs=1, grad_clip=None,
            save_path="unused.pth", eval_fn=lambda m: {"acc": 0.5}, verbose=False,
        )
        assert False, "expected ValueError for missing eval_metric_key"
    except ValueError:
        pass


def test_wandb_off_path_runs_without_wandb():
    model = _make_model()
    history = train_ssl_model(
        model, _make_loader(), epochs=2, grad_clip=None, wandb_logging=False, verbose=False,
    )
    assert len(history["train_loss"]) == 2


# ── return_run: keeping the wandb run open for one-shot final metrics ─────────

class _FakeRun:
    """Minimal stand-in for a wandb run, so these tests never touch the network."""

    def __init__(self):
        self.summary = {}
        self.logged = []
        self.finished = False

    def log(self, metrics):
        self.logged.append(metrics)

    def finish(self):
        self.finished = True


def _patch_wandb(monkeypatch):
    from training import train as train_mod

    run = _FakeRun()
    monkeypatch.setattr(train_mod, "init_wandb_run", lambda cfg: run)
    monkeypatch.setattr(train_mod, "wandb_log", lambda r, m: r.log(m))
    return run


_CFG = {"entity": "e", "project": "p", "name": "n"}


def test_return_run_false_keeps_the_old_single_return_value():
    model = _make_model()
    history = train_ssl_model(
        model, _make_loader(), epochs=2, grad_clip=None, wandb_logging=False, verbose=False,
    )
    assert isinstance(history, dict)


def test_return_run_true_yields_none_when_wandb_is_off():
    model = _make_model()
    history, run = train_ssl_model(
        model, _make_loader(), epochs=2, grad_clip=None, wandb_logging=False, verbose=False,
        return_run=True,
    )
    assert isinstance(history, dict) and run is None


def test_return_run_true_leaves_the_run_open_for_the_caller(monkeypatch):
    run = _patch_wandb(monkeypatch)
    model = _make_model()
    _, returned = train_ssl_model(
        model, _make_loader(), epochs=2, grad_clip=None, wandb_logging=True, wandb_config=_CFG,
        verbose=False, return_run=True,
    )
    assert returned is run
    assert not run.finished, "run must stay open so the caller can add summaries"
    assert "training_time_sec" in run.summary
    run.summary.update({"final/knn_test_acc": 0.8})
    run.finish()
    assert run.finished


def test_run_is_still_closed_automatically_when_return_run_is_false(monkeypatch):
    run = _patch_wandb(monkeypatch)
    model = _make_model()
    train_ssl_model(
        model, _make_loader(), epochs=2, grad_clip=None, wandb_logging=True, wandb_config=_CFG,
        verbose=False,
    )
    assert run.finished, "default path must still close the run"
