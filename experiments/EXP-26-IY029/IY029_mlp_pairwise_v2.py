"""
IY029 v2: MLP Pairwise Baseline — Same/Different Task (IY029 datasets, n=14 000)

Converted from IY029_mlp_pairwise.ipynb.
Code is unchanged; only data paths point to the regenerated IY029 static
.pt files and output filenames carry a _v2 suffix.

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Submit with: qsub IY029_mlp_pairwise_v2.sh  (from the EXP-26-IY029 dir)

Architecture:
  - Input: full flattened pair [x1_full | x2_full] — each half is the full trajectory
  - Network: Linear → BN → ReLU → Dropout, three hidden layers (512, 256, 64)
  - Head: Linear → 1 logit, BCE loss

Saves:
  IY029_mlp_pairwise_v2_results.json
  IY029_mlp_pairwise_v2_loss.png
  IY029_mlp_pairwise_v2_acc.png
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataloaders import load_loader_from_disk

IY029_DATA = SCRIPT_DIR / 'data'   # EXP-26-IY029/data/{2_fold,10_fold}/<cond>/
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Config ─────────────────────────────────────────────────────────────────────
# Use the first 500 points from each trajectory half to keep the dense baseline tractable.
SAMPLE_LEN = 500
BATCH_SIZE = 256
N_EPOCHS   = 100
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


# ── Model ─────────────────────────────────────────────────────────────────────

class PairwiseMLP(nn.Module):
    """
    Simple MLP for pairwise same/different classification.
    Input: full flattened [x1 | x2] pair, shape (T,) where T = full concatenated length.
    """
    def __init__(self, input_dim, hidden_dims=(512, 256, 64), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        # Repeat Linear-BatchNorm-ReLU-Dropout blocks before the binary logit.
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)   # (B,)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_split(pt_path):
    """Load a .pt loader → (X_np, y_np) numpy arrays."""
    loader = load_loader_from_disk(pt_path, batch_size=2048)
    Xs, ys = [], []
    for X, y in loader:
        # Load paired trajectories exactly as saved; flattening happens separately.
        Xs.append(X.numpy()); ys.append(y.numpy().ravel())
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32)


def flatten_pairs(X_np):
    """Flatten the full concatenated pair → (N, T), matching IY025 raw SVM (no crop)."""
    # Recover the two halves of the concatenated pair before flattening.
    half = X_np.shape[1] // 2
    x1 = X_np[:, :half,  0]   # (N, half)   — full first trajectory
    x2 = X_np[:, half:,  0]   # (N, T-half) — full second trajectory
    return np.concatenate([x1, x2], axis=1)   # (N, T)


def make_loader(X_flat, y, shuffle=True):
    # BCEWithLogitsLoss uses float targets rather than class-index labels.
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_flat, dtype=torch.float32),
        torch.tensor(y,      dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        # Threshold sigmoid probabilities at 0.5 for the same/different decision.
        preds = (torch.sigmoid(model(X)) > 0.5).float()
        correct += (preds == y).sum().item()
        total   += len(y)
    return correct / total


def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, verbose=False):
    """Train a fresh PairwiseMLP and return (test_accuracy, loss_curve)."""
    input_dim = X_tr.shape[1]
    # A fresh model is trained for each dataset/fold combination.
    model  = PairwiseMLP(input_dim).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=LR)
    crit   = nn.BCEWithLogitsLoss()
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS)

    tr_loader = make_loader(X_tr, y_tr, shuffle=True)
    va_loader = make_loader(X_va, y_va, shuffle=False)
    te_loader = make_loader(X_te, y_te, shuffle=False)

    best_va, best_state = 0.0, None
    loss_curve = []

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        sched.step()
        loss_curve.append(epoch_loss / len(tr_loader))

        va_acc = evaluate(model, va_loader)
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            print(f'  epoch {epoch+1:3d}  loss={loss_curve[-1]:.4f}  val={va_acc:.3f}')

    model.load_state_dict(best_state)
    return evaluate(model, te_loader), loss_curve


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'IY029_DATA : {IY029_DATA}  (exists: {IY029_DATA.exists()})')
    if not IY029_DATA.exists():
        raise FileNotFoundError('Data directory not found — check path above.')

    print(f'Device: {DEVICE}')
    print(f'SAMPLE_LEN={SAMPLE_LEN}  BATCH_SIZE={BATCH_SIZE}  N_EPOCHS={N_EPOCHS}')

    results     = {}   # {ds_name: {'iy011': acc, 'iy014': acc}}
    loss_curves = {}

    for cfg in DATASET_CONFIGS:
        name = cfg['name']
        results[name]     = {}
        loss_curves[name] = {}
        print(f'\n=== {name} ===')

        # Compare the same baseline across 2-fold and 10-fold variation settings.
        for fold in ('iy011', 'iy014'):
            print(f'  {fold}:', end=' ', flush=True)
            X_tr_raw, y_tr = load_split(cfg[f'{fold}_train'])
            X_va_raw, y_va = load_split(cfg[f'{fold}_val'])
            X_te_raw, y_te = load_split(cfg[f'{fold}_test'])

            # Flatten: [x1_crop | x2_crop] → (N, 2*SAMPLE_LEN)
            X_tr = flatten_pairs(X_tr_raw)
            X_va = flatten_pairs(X_va_raw)
            X_te = flatten_pairs(X_te_raw)

            acc, lc = train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, verbose=True)
            results[name][fold]     = acc
            loss_curves[name][fold] = lc
            print(f'test acc = {acc:.4f}')

    print('\nDone.')

    # ── Save JSON ──────────────────────────────────────────────────────────────
    save_path = SCRIPT_DIR / 'IY029_mlp_pairwise_v2_results.json'
    with open(save_path, 'w') as f:
        json.dump(
            {ds: {fold: float(results[ds][fold]) for fold in ('iy011', 'iy014')}
             for ds in DS_NAMES},
            f, indent=2,
        )
    print(f'Saved {save_path}')

    # ── Summary ────────────────────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for n in DS_NAMES:
        rows.append({
            'Dataset':         n,
            '2-fold (IY029)':  results[n]['iy011'],
            '10-fold (IY029)': results[n]['iy014'],
            'Mean':            np.mean([results[n]['iy011'], results[n]['iy014']]),
        })
    df = pd.DataFrame(rows).set_index('Dataset')
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(df.to_string())

    # ── Loss curves ───────────────────────────────────────────────────────────
    palette   = sns.color_palette('colorblind')
    ds_colors = {n: palette[i] for i, n in enumerate(DS_NAMES)}

    # Shared y-axis keeps loss magnitudes comparable between fold panels.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, fold, fold_title in [
        (axes[0], 'iy011', 'IY029 (2-fold)'),
        (axes[1], 'iy014', 'IY029 (10-fold)'),
    ]:
        for name in DS_NAMES:
            ax.plot(loss_curves[name][fold], color=ds_colors[name], label=name, lw=1.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('BCE loss', fontsize=12)
        ax.set_title(fold_title, fontsize=13)
        ax.legend(fontsize=10)
    fig.suptitle('Training loss curves — Pairwise MLP (v2)', fontsize=14)
    plt.tight_layout()
    loss_png = SCRIPT_DIR / 'IY029_mlp_pairwise_v2_loss.png'
    plt.savefig(loss_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {loss_png}')

    # ── Accuracy bars ─────────────────────────────────────────────────────────
    colors = [palette[i] for i in range(len(DS_NAMES))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    x     = np.arange(len(DS_NAMES))
    width = 0.5

    for ax, fold_key, fold_title in [
        (axes[0], 'iy011', 'IY029 (2-fold)'),
        (axes[1], 'iy014', 'IY029 (10-fold)'),
    ]:
        accs = [results[n][fold_key] for n in DS_NAMES]
        bars = ax.bar(x, accs, width, color=colors, edgecolor='black', linewidth=0.6)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
        ax.axhline(0.5, color='dimgrey', linestyle='--', linewidth=1.2, label='Chance (50%)')
        ax.set_xticks(x)
        ax.set_xticklabels(DS_NAMES, fontsize=11)
        ax.set_xlabel('Varied statistic', fontsize=12)
        ax.set_ylim(0, 1.25)
        ax.set_title(fold_title, fontsize=13)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    axes[0].set_ylabel('Test accuracy', fontsize=12)
    fig.suptitle('MLP — pairwise same/different accuracy (v2, n=14 000)\n'
                 'input: full flattened [x1 | x2] (no crop)',
                 fontsize=13, weight='bold')
    plt.tight_layout()
    acc_png = SCRIPT_DIR / 'IY029_mlp_pairwise_v2_acc.png'
    plt.savefig(acc_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {acc_png}')


if __name__ == '__main__':
    main()
