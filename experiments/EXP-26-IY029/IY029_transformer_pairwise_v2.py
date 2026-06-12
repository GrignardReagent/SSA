"""
IY029 v2: Supervised TransformerClassifier — Pairwise Same/Different Task (IY029 datasets, n=14 000)

Code is unchanged from IY029_transformer_pairwise.py; only data paths point to the
regenerated IY029 static .pt files and output filenames carry a _v2 suffix.

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Submit with: qsub IY029_transformer_pairwise_v2.sh  (from the EXP-26-IY029 dir)

The full concatenated pair (T, 1) is fed directly — no siamese splitting.
CrossEntropyLoss with num_classes=2 (0=different, 1=same). Chance = 50%.

BATCH_SIZE=32, early stopping patience=15 (stops once val acc stalls).
Partial results are written after each (dataset, fold) so a timeout doesn't lose work.

Saves:
  IY029_transformer_pairwise_v2_results.json
  IY029_transformer_pairwise_v2_loss.png
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataloaders import load_loader_from_disk
from models.transformer import TransformerClassifier

IY029_DATA = SCRIPT_DIR / 'data'   # EXP-26-IY029/data/{2_fold,10_fold}/<cond>/
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Config ─────────────────────────────────────────────────────────────────────
D_MODEL    = 16
NHEAD      = 4
NUM_LAYERS = 2
DROPOUT    = 0.1
# BATCH_SIZE=8: safe on V100 (16 GB); max attention ≈ 8.7 GB for T=5821
BATCH_SIZE = 32
N_EPOCHS   = 100
PATIENCE   = 15   # early-stop if val acc doesn't improve for this many epochs
LR         = 1e-3

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.titlesize': 14,
})

# ── Dataset config ─────────────────────────────────────────────────────────────
# Now loading from the regenerated IY029 static .pt files.
DATASET_CONFIGS = [
    {
        'name':        'Baseline',
        'iy011_train': IY029_DATA / '2_fold'  / 'baseline' / 'IY029_static_train.pt',
        'iy011_val':   IY029_DATA / '2_fold'  / 'baseline' / 'IY029_static_val.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 'baseline' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_train.pt',
        'iy014_val':   IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_val.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_test.pt',
    },
    {
        'name':        'Mu',
        'iy011_train': IY029_DATA / '2_fold'  / 'mu' / 'IY029_static_train.pt',
        'iy011_val':   IY029_DATA / '2_fold'  / 'mu' / 'IY029_static_val.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 'mu' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 'mu' / 'IY029_static_train.pt',
        'iy014_val':   IY029_DATA / '10_fold' / 'mu' / 'IY029_static_val.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 'mu' / 'IY029_static_test.pt',
    },
    {
        'name':        'CV',
        'iy011_train': IY029_DATA / '2_fold'  / 'cv' / 'IY029_static_train.pt',
        'iy011_val':   IY029_DATA / '2_fold'  / 'cv' / 'IY029_static_val.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 'cv' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 'cv' / 'IY029_static_train.pt',
        'iy014_val':   IY029_DATA / '10_fold' / 'cv' / 'IY029_static_val.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 'cv' / 'IY029_static_test.pt',
    },
    {
        'name':        'T_ac',
        'iy011_train': IY029_DATA / '2_fold'  / 't_ac' / 'IY029_static_train.pt',
        'iy011_val':   IY029_DATA / '2_fold'  / 't_ac' / 'IY029_static_val.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 't_ac' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_train.pt',
        'iy014_val':   IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_val.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_test.pt',
    },
]
DS_NAMES = [cfg['name'] for cfg in DATASET_CONFIGS]


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_split(pt_path: Path):
    """Load a .pt static loader → (X_np, y_np) numpy arrays."""
    loader = load_loader_from_disk(pt_path, batch_size=2048)
    Xs, ys = [], []
    for X, y in loader:
        # The saved static loaders already contain full concatenated pairs:
        # X has shape (batch, T_total, 1), y is 0=different or 1=same.
        Xs.append(X.numpy())
        ys.append(y.numpy().ravel())
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64)


def make_tensor_loader(X_np, y_np, batch_size, shuffle=True):
    """Wrap numpy arrays without altering the pair order or sequence length."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Model helpers ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    """Return classification accuracy on a DataLoader."""
    model.eval()
    correct = total = 0
    for X, y in loader:
        preds   = model(X.to(DEVICE)).argmax(dim=1)
        correct += (preds == y.to(DEVICE)).sum().item()
        total   += len(y)
    return correct / total


def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
                   ds_name: str, fold: str) -> tuple:
    """
    Train a fresh TransformerClassifier on the full pair sequence.
    Returns (test_accuracy, loss_curve, best_state).
    """
    model     = TransformerClassifier(
        input_size=1, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, num_classes=2, dropout=DROPOUT,
    ).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS)

    tr_loader = make_tensor_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_tensor_loader(X_va, y_va, BATCH_SIZE, shuffle=False)
    te_loader = make_tensor_loader(X_te, y_te, BATCH_SIZE, shuffle=False)

    best_va, best_state = 0.0, None
    loss_curve = []
    no_improve = 0
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X, y in tr_loader:
            # Feed the complete pair sequence directly; the classifier learns the
            # same/different decision from the concatenated trajectory.
            optimiser.zero_grad()
            loss = criterion(model(X.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        loss_curve.append(epoch_loss / len(tr_loader))

        va_acc = evaluate(model, va_loader)
        if va_acc > best_va:
            # Store CPU copies so the best validation checkpoint is independent
            # of later optimizer updates and can be restored after training.
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
            print(f'  [{ds_name}/{fold}] epoch {epoch+1:3d}  '
                  f'loss={loss_curve[-1]:.4f}  val={va_acc:.3f}  '
                  f'ETA {eta/60:.1f}min', flush=True)

        if no_improve >= PATIENCE:
            print(f'  [{ds_name}/{fold}] early stop at epoch {epoch+1} '
                  f'(best val={best_va:.3f})', flush=True)
            break

    model.load_state_dict(best_state)
    te_acc = evaluate(model, te_loader)
    return te_acc, loss_curve, best_state


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_loss_curves(loss_curves: dict, fname: Path):
    palette   = sns.color_palette('colorblind')
    # Keep dataset colors stable across the 2-fold and 10-fold panels.
    ds_colors = {n: palette[i] for i, n in enumerate(DS_NAMES)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, fold, fold_title in [
        (axes[0], 'iy011', 'IY029 (2-fold)'),
        (axes[1], 'iy014', 'IY029 (10-fold)'),
    ]:
        for name in DS_NAMES:
            ax.plot(loss_curves[name][fold], color=ds_colors[name],
                    label=name, lw=1.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('CrossEntropy loss', fontsize=12)
        ax.set_title(fold_title, fontsize=13)
        ax.legend(fontsize=10)

    fig.suptitle('Training loss curves — TransformerClassifier (v2, full pair input)', fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'REPO_ROOT  : {REPO_ROOT}')
    print(f'IY029_DATA : {IY029_DATA}  (exists: {IY029_DATA.exists()})')
    if not IY029_DATA.exists():
        raise FileNotFoundError('Data directory not found — check path above.')

    print(f'Device: {DEVICE}')
    print(f'BATCH_SIZE={BATCH_SIZE}  N_EPOCHS={N_EPOCHS}  D_MODEL={D_MODEL}')

    results     = {}
    loss_curves = {}
    t_total     = time.time()
    ckpt_dir    = SCRIPT_DIR / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    for cfg in DATASET_CONFIGS:
        name = cfg['name']
        results[name]     = {}
        loss_curves[name] = {}
        print(f'\n=== {name} ===', flush=True)

        for fold in ('iy011', 'iy014'):
            print(f'  {fold}: loading data...', flush=True)
            # iy011 keys map to the 2-fold IY029 datasets; iy014 maps to 10-fold.
            X_tr, y_tr = load_split(cfg[f'{fold}_train'])
            X_va, y_va = load_split(cfg[f'{fold}_val'])
            X_te, y_te = load_split(cfg[f'{fold}_test'])
            print(f'    train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}', flush=True)

            acc, lc, state = train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, name, fold)
            results[name][fold]     = acc
            loss_curves[name][fold] = lc
            print(f'  [{name}/{fold}] test acc = {acc:.4f}', flush=True)

            # Save best-validation checkpoint for downstream embedding visualisation.
            ckpt_path = ckpt_dir / f'IY029_transformer_v2_{name.lower()}_{fold}.pth'
            torch.save({'state_dict': state, 'd_model': D_MODEL, 'nhead': NHEAD,
                        'num_layers': NUM_LAYERS, 'dropout': DROPOUT}, ckpt_path)
            print(f'  Saved checkpoint: {ckpt_path.name}', flush=True)

            # Write partial results after each run so a timeout doesn't lose them.
            partial_path = SCRIPT_DIR / 'IY029_transformer_pairwise_v2_results.json'
            with open(partial_path, 'w') as _f:
                json.dump(
                    {ds: {f: float(results[ds][f]) for f in results[ds]}
                     for ds in results},
                    _f, indent=2,
                )
            print(f'  Partial results saved.', flush=True)

    print(f'\nTotal training time: {(time.time()-t_total)/60:.1f} min')

    # ── Save JSON ──────────────────────────────────────────────────────────────
    save_path = SCRIPT_DIR / 'IY029_transformer_pairwise_v2_results.json'
    with open(save_path, 'w') as f:
        # Convert numpy/PyTorch scalar values to plain floats for portable JSON.
        json.dump(
            {ds: {fold: float(results[ds][fold]) for fold in ('iy011', 'iy014')}
             for ds in DS_NAMES},
            f, indent=2,
        )
    print(f'Saved {save_path}')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n=== Summary ===')
    header = f"{'Dataset':<12}" + ''.join(f'  {fold:>10}' for fold in ('iy011 2-fold', 'iy014 10-fold'))
    print(header)
    for n in DS_NAMES:
        print(f'{n:<12}  {results[n]["iy011"]:>10.4f}  {results[n]["iy014"]:>12.4f}')

    # ── Loss plot ──────────────────────────────────────────────────────────────
    plot_loss_curves(loss_curves, SCRIPT_DIR / 'IY029_transformer_pairwise_v2_loss.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
