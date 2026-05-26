"""
IY029 v2: Bidirectional LSTM — Pairwise Same/Different Task (IY029 datasets, n=14 000)

Code is unchanged from IY029_lstm_pairwise.py; only data paths point to the
regenerated IY029 static .pt files and output filenames carry a _v2 suffix.

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Submit with: qsub IY029_lstm_pairwise_v2.sh  (from the EXP-26-IY029 dir)

Architecture: input_size=1, hidden_size=64, num_layers=2, bidirectional=True,
              use_conv1d=False, use_attention=False — plain LSTM baseline.
              CrossEntropyLoss (label_smoothing=0.1), Adam, ReduceLROnPlateau,
              early stopping patience=15.

BATCH_SIZE=32: BPTT stores all T hidden states per sample in GPU memory
              (O(T × batch)); must be small for full-length pair sequences.

Saves:
  IY029_lstm_pairwise_v2_results.json
  IY029_lstm_pairwise_v2_loss.png
"""
import sys
import json
import time
import numpy as np
import torch
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
from models.lstm import LSTMClassifier

IY029_DATA = SCRIPT_DIR / 'data'   # EXP-26-IY029/data/{2_fold,10_fold}/<cond>/

# ── Config ─────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 32    # BPTT stores O(T × batch) hidden states; keep small for full-length pairs
N_EPOCHS    = 100
PATIENCE    = 100
LR          = 1e-3

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
    """Load a .pt static loader → (X_np float32, y_np int64) numpy arrays."""
    loader = load_loader_from_disk(pt_path, batch_size=2048)
    Xs, ys = [], []
    for X, y in loader:
        Xs.append(X.numpy())
        ys.append(y.numpy().ravel())
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64)


def make_loader(X_np, y_np, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Training ───────────────────────────────────────────────────────────────────

def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
                   ds_name: str, fold: str) -> tuple:
    """
    Train a fresh LSTMClassifier and evaluate on the test set.
    Returns (test_accuracy, train_loss_curve, model) — model is returned so
    the caller can save its state_dict for embedding visualisation.
    """
    model = LSTMClassifier(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=2,
        dropout_rate=DROPOUT,
        learning_rate=LR,
        use_conv1d=False,
        use_attention=False,
        bidirectional=True,
        verbose=False,
    )
    tr_loader = make_loader(X_tr, y_tr, shuffle=True)
    va_loader = make_loader(X_va, y_va, shuffle=False)
    te_loader = make_loader(X_te, y_te, shuffle=False)

    t0      = time.time()
    history = model.train_model(tr_loader, va_loader,
                                epochs=N_EPOCHS, patience=PATIENCE)
    elapsed = time.time() - t0
    te_acc  = model.evaluate(te_loader)

    n_epochs = len(history['train_loss'])
    print(f'  [{ds_name}/{fold}] test acc = {te_acc:.4f}  '
          f'({n_epochs} epochs, {elapsed/60:.1f} min)', flush=True)
    return te_acc, history['train_loss'], model


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_loss_curves(loss_curves: dict, fname: Path):
    palette   = sns.color_palette('colorblind')
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

    fig.suptitle('Training loss curves — Bidirectional LSTM (v2, full pair input)', fontsize=14)
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

    print(f'HIDDEN_SIZE={HIDDEN_SIZE}  NUM_LAYERS={NUM_LAYERS}  '
          f'BATCH_SIZE={BATCH_SIZE}  N_EPOCHS={N_EPOCHS}  PATIENCE={PATIENCE}')

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
            X_tr, y_tr = load_split(cfg[f'{fold}_train'])
            X_va, y_va = load_split(cfg[f'{fold}_val'])
            X_te, y_te = load_split(cfg[f'{fold}_test'])
            print(f'    train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}', flush=True)

            acc, lc, model = train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, name, fold)
            results[name][fold]     = acc
            loss_curves[name][fold] = lc

            # Save model state_dict for downstream embedding visualisation.
            ckpt_path = ckpt_dir / f'IY029_lstm_v2_{name.lower()}_{fold}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Saved checkpoint: {ckpt_path.name}', flush=True)

    print(f'\nTotal training time: {(time.time()-t_total)/60:.1f} min')

    # ── Save JSON ──────────────────────────────────────────────────────────────
    save_path = SCRIPT_DIR / 'IY029_lstm_pairwise_v2_results.json'
    with open(save_path, 'w') as f:
        json.dump(
            {ds: {fold: float(results[ds][fold]) for fold in ('iy011', 'iy014')}
             for ds in DS_NAMES},
            f, indent=2,
        )
    print(f'Saved {save_path}')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n=== Summary ===')
    header = f"{'Dataset':<12}" + ''.join(f'  {fold:>12}' for fold in ('iy011 2-fold', 'iy014 10-fold'))
    print(header)
    for n in DS_NAMES:
        print(f'{n:<12}  {results[n]["iy011"]:>12.4f}  {results[n]["iy014"]:>14.4f}')

    # ── Loss plot ──────────────────────────────────────────────────────────────
    plot_loss_curves(loss_curves, SCRIPT_DIR / 'IY029_lstm_pairwise_v2_loss.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
