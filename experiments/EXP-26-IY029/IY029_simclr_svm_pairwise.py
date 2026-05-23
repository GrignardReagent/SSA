"""
IY029: SimCLR Pairwise SVM — All SimCLR Models (IY017–IY024)

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Expected repo root on Eddie: /home/s1732775/eddie_swain/SSA/
Submit with: qsub IY029_simclr_svm_pairwise.sh  (from the EXP-26-IY029 dir)

Evaluates all 34 SimCLR checkpoints on a pairwise same/different task using
IY011 (2-fold) and IY014 (10-fold) variation datasets across four conditions
(Baseline, Mu, CV, T_ac). Two feature sets are evaluated:

  1. Embeddings only:  [z1 | z2]           shape (N, 2*d_model)
  2. Combined:         [x1_raw | x2_raw | z1 | z2]  shape (N, T + 2*d_model)

Chance = 50%. Full trajectories used for both raw and encoded parts.
ENCODE_BATCH_SIZE=64 is calibrated for Eddie V100 (16 GB VRAM):
  max attention cost = 64 × 4 × 2910² × 4 B ≈ 8.6 GB.

Saves:
  IY029_simclr_svm_pairwise_results.json             (after embeddings section)
  IY029_simclr_svm_pairwise_results_combined.json    (after combined section)
  IY029_simclr_svm_pairwise_2fold.png
  IY029_simclr_svm_pairwise_10fold.png
  IY029_simclr_svm_pairwise_best_model.png
  IY029_simclr_svm_pairwise_combined_2fold.png
  IY029_simclr_svm_pairwise_combined_10fold.png
  IY029_simclr_svm_pairwise_combined_best_model.png
"""
import os
# Reduce CUDA allocator fragmentation — must be set before importing torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import re
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Config ─────────────────────────────────────────────────────────────────────
# batch_size=64: attention cost = 64 × 4 × T_half² × 4 B
# For max T_half ≈ 2910 → 8.64 GB, safe on Eddie V100 (16 GB VRAM).
ENCODE_BATCH_SIZE = 64

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataloaders import load_loader_from_disk
from models.ssl_transformer import SSL_Transformer

IY011_ROOT = REPO_ROOT / 'experiments' / 'EXP-25-IY011'
IY014_ROOT = REPO_ROOT / 'experiments' / 'EXP-26-IY014'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize':  14,
})

# ── Dataset config (mirrors IY025 exactly) ────────────────────────────────────
DATASET_CONFIGS = [
    {
        'name':        'Baseline',
        'iy011_train': IY011_ROOT / 'data'                / 'IY011_static_train.pt',
        'iy011_test':  IY011_ROOT / 'data'                / 'IY011_static_test.pt',
        'iy014_train': IY014_ROOT / 'data'                / 'IY014_static_train.pt',
        'iy014_test':  IY014_ROOT / 'data'                / 'IY014_static_test.pt',
    },
    {
        'name':        'Mu',
        'iy011_train': IY011_ROOT / 'data_mu_variation'   / 'IY011_static_train.pt',
        'iy011_test':  IY011_ROOT / 'data_mu_variation'   / 'IY011_static_test.pt',
        'iy014_train': IY011_ROOT / 'data_mu_variation'   / 'IY014_static_train.pt',
        'iy014_test':  IY011_ROOT / 'data_mu_variation'   / 'IY014_static_test.pt',
    },
    {
        'name':        'CV',
        'iy011_train': IY011_ROOT / 'data_cv_variation'   / 'IY011_static_train.pt',
        'iy011_test':  IY011_ROOT / 'data_cv_variation'   / 'IY011_static_test.pt',
        'iy014_train': IY014_ROOT / 'data_cv_variation'   / 'IY014_static_train.pt',
        'iy014_test':  IY014_ROOT / 'data_cv_variation'   / 'IY014_static_test.pt',
    },
    {
        'name':        'T_ac',
        'iy011_train': IY011_ROOT / 'data_t_ac_variation' / 'IY011_static_train.pt',
        'iy011_test':  IY011_ROOT / 'data_t_ac_variation' / 'IY011_static_test.pt',
        'iy014_train': IY014_ROOT / 'data_t_ac_variation' / 'IY014_static_train.pt',
        'iy014_test':  IY014_ROOT / 'data_t_ac_variation' / 'IY014_static_test.pt',
    },
]

# ── Model registry ─────────────────────────────────────────────────────────────
# Norm type overrides for checkpoints that lack an explicit tag in their filename
NORM_OVERRIDES = {
    'IY022_simCLR_b64_lr0.01_L2_H4_D16_20260303_170229_model': 'global',
    'IY022_simCLR_b64_lr0.01_L2_H4_D16_20260308_125632_model': 'global',
    'IY022_simCLR_b64_lr0.01_L2_H4_D16_20260308_132219_model': 'joint',
    'IY023_simCLR_b64_lr0.01_L2_H4_D16_20260308_125550_model': 'global',
    'IY023_simCLR_b64_lr0.01_L2_H4_D16_20260308_165118_model': 'joint',
}


def parse_arch_from_name(name: str) -> dict:
    """Extract transformer architecture hyperparameters from checkpoint filename."""
    d_model    = int(re.search(r'_D(\d+)', name).group(1))
    nhead      = int(re.search(r'_H(\d+)', name).group(1))
    num_layers = int(re.search(r'_L(\d+)', name).group(1))
    return dict(input_size=1, d_model=d_model, nhead=nhead,
                num_layers=num_layers, dropout=0.01, use_conv1d=False)


def make_short_label(path: Path) -> str:
    """Human-readable model label from checkpoint path."""
    name   = path.stem
    exp    = re.search(r'(IY\d+)', name).group(1)
    arch_m = re.search(r'(b\d+_lr[\d.]+(?:_d[\d.]+)?_L\d+_H\d+_D\d+)', name)
    arch   = arch_m.group(1) if arch_m else re.search(r'(L\d+_H\d+_D\d+)', name).group(1)
    ts     = re.search(r'(\d{8}_\d{6})', name)
    ts     = ts.group(1)[-6:] if ts else ''
    norm   = (NORM_OVERRIDES.get(name)
              or (re.search(r'_(instance|global|joint)_', name) or [None, ''])[1]
              or 'instance')
    return f'{exp}-{arch}-{norm}-{ts}'


CKPT_DIRS = ['EXP-26-IY017', 'EXP-26-IY022', 'EXP-26-IY023', 'EXP-26-IY024']
MODEL_REGISTRY = [
    (p, make_short_label(p))
    for d in CKPT_DIRS
    for p in sorted((REPO_ROOT / 'experiments' / d).glob('*.pth'))
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pair_data(train_pt: Path, test_pt: Path):
    """
    Load IY011/IY014 static pairwise loader.
    Each sample shape (T, 1): two concatenated full trajectories.
    Returns: X_train (N,T,1), X_test (M,T,1), y_train (N,), y_test (M,)
    """
    def _extract(pt_path):
        loader = load_loader_from_disk(pt_path, batch_size=2048)
        Xs, ys = [], []
        for X, y in loader:
            Xs.append(X.numpy())
            ys.append(y.numpy().ravel())
        return np.concatenate(Xs, axis=0), np.concatenate(ys).astype(int)

    X_train, y_train = _extract(train_pt)
    X_test,  y_test  = _extract(test_pt)
    return X_train, X_test, y_train, y_test


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device) -> SSL_Transformer:
    """Instantiate SSL_Transformer from filename and load checkpoint weights."""
    arch  = parse_arch_from_name(ckpt_path.stem)
    model = SSL_Transformer(**arch).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def encode_pairs(model: SSL_Transformer, X_np: np.ndarray,
                 device, batch_size: int = ENCODE_BATCH_SIZE) -> np.ndarray:
    """
    Encode concatenated pair array as [z1 | z2].

    X_np: (N, T, 1) — two full trajectories concatenated along T.
    Splits at T//2, encodes each full half with no cropping.
    Returns: (N, 2 * d_model)

    Memory note: O(T²) attention per batch.
    batch_size=64 → ~8.6 GB attention for max T_half≈2910 (safe on V100 16 GB).
    """
    half = X_np.shape[1] // 2
    x1 = X_np[:, :half, :]    # (N, half, 1)
    x2 = X_np[:, half:, :]    # (N, T-half, 1)

    z1_parts, z2_parts = [], []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            b1 = torch.tensor(x1[i:i+batch_size], dtype=torch.float32).to(device)
            b2 = torch.tensor(x2[i:i+batch_size], dtype=torch.float32).to(device)
            z1_parts.append(model.backbone.encode(b1).cpu().numpy())
            z2_parts.append(model.backbone.encode(b2).cpu().numpy())

    z1 = np.concatenate(z1_parts, axis=0)
    z2 = np.concatenate(z2_parts, axis=0)
    return np.concatenate([z1, z2], axis=1)   # (N, 2*d_model)


def encode_pairs_combined(model: SSL_Transformer, X_np: np.ndarray,
                          device, batch_size: int = ENCODE_BATCH_SIZE) -> np.ndarray:
    """
    Encode concatenated pair array as [x1_raw | x2_raw | z1 | z2].

    Augments SimCLR embeddings with raw full-trajectory features so the SVM
    can exploit both the learned representation and the raw temporal signal.
    Returns: (N, T + 2*d_model)  where T = total concatenated length.
    """
    half = X_np.shape[1] // 2
    x1 = X_np[:, :half, :]    # (N, half, 1)
    x2 = X_np[:, half:, :]    # (N, T-half, 1)

    z1_parts, z2_parts = [], []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            b1 = torch.tensor(x1[i:i+batch_size], dtype=torch.float32).to(device)
            b2 = torch.tensor(x2[i:i+batch_size], dtype=torch.float32).to(device)
            z1_parts.append(model.backbone.encode(b1).cpu().numpy())
            z2_parts.append(model.backbone.encode(b2).cpu().numpy())

    z1      = np.concatenate(z1_parts, axis=0)   # (N, d_model)
    z2      = np.concatenate(z2_parts, axis=0)   # (N, d_model)
    x1_flat = x1.reshape(len(X_np), -1)          # (N, half)
    x2_flat = x2.reshape(len(X_np), -1)          # (N, T-half)

    return np.concatenate([x1_flat, x2_flat, z1, z2], axis=1)


def run_svm(model: SSL_Transformer,
            X_train: np.ndarray, X_test: np.ndarray,
            y_train: np.ndarray, y_test: np.ndarray,
            device, encode_fn=encode_pairs) -> float:
    """Encode pairs, scale, fit RBF SVM, return test accuracy."""
    feats_tr = encode_fn(model, X_train, device)
    feats_te = encode_fn(model, X_test,  device)

    scaler   = StandardScaler()
    feats_tr = scaler.fit_transform(feats_tr)
    feats_te = scaler.transform(feats_te)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(feats_tr, y_train)
    return clf.score(feats_te, y_test)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_fold(results: dict, model_labels: list, ds_names: list,
              ds_colors: dict, fold_key: str, fold_title: str,
              fname: Path, title_prefix: str = 'Pairwise Same/Different SVM'):
    """Bar chart of all models × datasets, sorted high→low by mean accuracy."""
    n_ds    = len(ds_names)
    width   = 0.18
    offsets = np.linspace(-(n_ds - 1) / 2 * width, (n_ds - 1) / 2 * width, n_ds)

    sorted_labels = sorted(
        model_labels,
        key=lambda lbl: np.mean([results[lbl][ds][fold_key] for ds in ds_names]),
        reverse=True,
    )
    x = np.arange(len(sorted_labels))

    fig, ax = plt.subplots(figsize=(max(16, len(sorted_labels) * 0.55), 5),
                           constrained_layout=True)

    for j, ds_name in enumerate(ds_names):
        accs = [results[lbl][ds_name][fold_key] for lbl in sorted_labels]
        ax.bar(x + offsets[j], accs, width=width,
               color=ds_colors[ds_name], label=ds_name,
               edgecolor='white', linewidth=0.4)

    ax.axhline(0.5, color='dimgrey', linestyle='--', linewidth=1.2,
               label='Chance (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('SimCLR model', fontsize=12)
    ax.set_ylabel('SVM accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{title_prefix} — {fold_title}\n'
                 f'All SimCLR Models (IY017–IY024), sorted high→low, chance = 50%',
                 fontsize=14)

    legend_handles = [
        mpatches.Patch(color=ds_colors[n], label=n) for n in ds_names
    ] + [plt.Line2D([0], [0], color='dimgrey', linestyle='--',
                    linewidth=1.2, label='Chance (50%)')]
    ax.legend(handles=legend_handles, fontsize=10,
              bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')


def plot_best_model(results: dict, model_labels: list, ds_names: list,
                    ds_colors: dict, fname: Path,
                    suptitle_prefix: str = 'Best model') -> tuple:
    """Side-by-side 2-fold / 10-fold bars for the best overall model."""
    mean_accs = {
        lbl: np.mean([results[lbl][ds][fold]
                      for ds in ds_names for fold in ('iy011', 'iy014')])
        for lbl in model_labels
    }
    best_label = max(mean_accs, key=mean_accs.get)
    print(f'{suptitle_prefix}: {best_label}  (mean acc = {mean_accs[best_label]:.3f})')

    accs_2fold  = [results[best_label][ds]['iy011'] for ds in ds_names]
    accs_10fold = [results[best_label][ds]['iy014'] for ds in ds_names]
    colors      = [ds_colors[ds] for ds in ds_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x, width  = np.arange(len(ds_names)), 0.5

    for ax, accs, fold_title in [
        (axes[0], accs_2fold,  'IY011 (2-fold variation)'),
        (axes[1], accs_10fold, 'IY014 (10-fold variation)'),
    ]:
        bars = ax.bar(x, accs, width, color=colors, edgecolor='black', linewidth=0.6)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
        ax.axhline(0.5, color='dimgrey', linestyle='--', linewidth=1.2,
                   label='Chance (50%)')
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, fontsize=11)
        ax.set_xlabel('Varied statistic', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title(fold_title, fontsize=13)
        ax.legend(fontsize=10, loc='lower right')

    axes[0].set_ylabel('SimCLR + SVM accuracy', fontsize=12)
    fig.suptitle(f'{suptitle_prefix}: {best_label}', fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')

    return best_label, mean_accs


def print_summary_table(results: dict, model_labels: list, ds_names: list,
                        mean_accs: dict, title: str = 'Summary'):
    """Print a sorted accuracy table to stdout."""
    rows = []
    for lbl in model_labels:
        row = {'model': lbl}
        for ds in ds_names:
            row[f'{ds} 2-fold']  = results[lbl][ds]['iy011']
            row[f'{ds} 10-fold'] = results[lbl][ds]['iy014']
        row['mean'] = mean_accs[lbl]
        rows.append(row)

    df = (pd.DataFrame(rows)
          .set_index('model')
          .sort_values('mean', ascending=False))
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(f'\n=== {title} ===')
    print(df.to_string())


# ── Evaluation loop helper ─────────────────────────────────────────────────────

def run_evaluation_loop(prepared_data: dict, ds_names: list,
                        encode_fn, label: str) -> dict:
    """
    Run all 34 models × 4 datasets × 2 folds and return nested results dict.
    Prints one progress line per model.
    """
    results = {}
    t0      = time.time()

    for i, (ckpt_path, mdl_label) in enumerate(MODEL_REGISTRY):
        results[mdl_label] = {}
        model = load_model(ckpt_path, DEVICE)

        for ds_name in ds_names:
            results[mdl_label][ds_name] = {}
            for fold in ('iy011', 'iy014'):
                X_tr, X_te, y_tr, y_te = prepared_data[ds_name][fold]
                acc = run_svm(model, X_tr, X_te, y_tr, y_te, DEVICE, encode_fn)
                results[mdl_label][ds_name][fold] = acc

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (len(MODEL_REGISTRY) - i - 1)
        accs    = [
            f"{results[mdl_label][ds]['iy011']:.3f}/{results[mdl_label][ds]['iy014']:.3f}"
            for ds in ds_names
        ]
        print(f'[{label}] [{i+1:2d}/{len(MODEL_REGISTRY)}] {mdl_label}  '
              + '  '.join(f'{d}={a}' for d, a in zip(ds_names, accs))
              + f'  ETA {eta/60:.1f}min', flush=True)

    print(f'[{label}] Total time: {(time.time()-t0)/60:.1f} min')
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Print resolved paths so the .out file confirms the correct Eddie filesystem
    print(f'REPO_ROOT  : {REPO_ROOT}')
    print(f'IY011_ROOT : {IY011_ROOT}  (exists: {IY011_ROOT.exists()})')
    print(f'IY014_ROOT : {IY014_ROOT}  (exists: {IY014_ROOT.exists()})')
    if not IY011_ROOT.exists() or not IY014_ROOT.exists():
        raise FileNotFoundError('Data roots not found — check paths above.')

    print(f'Device: {DEVICE}')
    print(f'ENCODE_BATCH_SIZE: {ENCODE_BATCH_SIZE}')
    print(f'Found {len(MODEL_REGISTRY)} model checkpoints')

    # ── Load all datasets ──────────────────────────────────────────────────────
    print('\n=== Loading pairwise data ===')
    prepared_data = {}
    for cfg in DATASET_CONFIGS:
        name = cfg['name']
        print(f'Loading {name}...', flush=True)
        prepared_data[name] = {
            'iy011': load_pair_data(cfg['iy011_train'], cfg['iy011_test']),
            'iy014': load_pair_data(cfg['iy014_train'], cfg['iy014_test']),
        }
        for fold, (Xtr, Xte, ytr, yte) in prepared_data[name].items():
            print(f'  {fold}: train={Xtr.shape}, test={Xte.shape}, '
                  f'same={yte.sum()}, diff={(yte==0).sum()}')

    ds_names  = [cfg['name'] for cfg in DATASET_CONFIGS]
    palette   = sns.color_palette('colorblind')
    ds_colors = {name: palette[i] for i, name in enumerate(ds_names)}
    model_labels = [label for _, label in MODEL_REGISTRY]

    # ══════════════════════════════════════════════════════════════════════════
    # Section 1: Embeddings only — [z1 | z2]
    # ══════════════════════════════════════════════════════════════════════════
    print('\n=== Section 1: [z1|z2] SimCLR SVM ===')
    results = run_evaluation_loop(prepared_data, ds_names, encode_pairs, 'z1z2')

    mean_accs  = {
        lbl: np.mean([results[lbl][ds][fold]
                      for ds in ds_names for fold in ('iy011', 'iy014')])
        for lbl in model_labels
    }
    best_label = max(mean_accs, key=mean_accs.get)

    # Save JSON immediately so results are not lost if combined section fails
    save_path = SCRIPT_DIR / 'IY029_simclr_svm_pairwise_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'best_label': best_label,
            'results': {
                lbl: {
                    ds: {fold: float(results[lbl][ds][fold]) for fold in ('iy011', 'iy014')}
                    for ds in ds_names
                }
                for lbl in results
            },
        }, f, indent=2)
    print(f'Saved {save_path}  (best: {best_label})')

    print_summary_table(results, model_labels, ds_names, mean_accs,
                        title='[z1|z2] Embeddings SVM')

    # Plots
    print('\n=== Plots: Section 1 ===')
    plot_fold(results, model_labels, ds_names, ds_colors,
              'iy011', 'IY011 (2-fold variation)',
              SCRIPT_DIR / 'IY029_simclr_svm_pairwise_2fold.png')
    plot_fold(results, model_labels, ds_names, ds_colors,
              'iy014', 'IY014 (10-fold variation)',
              SCRIPT_DIR / 'IY029_simclr_svm_pairwise_10fold.png')
    plot_best_model(results, model_labels, ds_names, ds_colors,
                    SCRIPT_DIR / 'IY029_simclr_svm_pairwise_best_model.png',
                    suptitle_prefix='Best model [z1|z2]')

    # ══════════════════════════════════════════════════════════════════════════
    # Section 2: Combined — [x1_raw | x2_raw | z1 | z2]
    # Note: SVM on raw trajectories (~3000–5800 features) is slower than
    # the embedding SVM (~16–256 features), expect ~3–5× longer wall time.
    # ══════════════════════════════════════════════════════════════════════════
    print('\n=== Section 2: [raw|z1|z2] Combined SVM ===')
    results_combined = run_evaluation_loop(prepared_data, ds_names,
                                           encode_pairs_combined, 'combined')

    mean_accs_combined = {
        lbl: np.mean([results_combined[lbl][ds][fold]
                      for ds in ds_names for fold in ('iy011', 'iy014')])
        for lbl in model_labels
    }
    best_combined = max(mean_accs_combined, key=mean_accs_combined.get)

    save_path_combined = SCRIPT_DIR / 'IY029_simclr_svm_pairwise_results_combined.json'
    with open(save_path_combined, 'w') as f:
        json.dump({
            'best_label': best_combined,
            'results': {
                lbl: {
                    ds: {fold: float(results_combined[lbl][ds][fold])
                         for fold in ('iy011', 'iy014')}
                    for ds in ds_names
                }
                for lbl in results_combined
            },
        }, f, indent=2)
    print(f'Saved {save_path_combined}  (best: {best_combined})')

    print_summary_table(results_combined, model_labels, ds_names,
                        mean_accs_combined, title='[raw|z1|z2] Combined SVM')

    # Plots
    print('\n=== Plots: Section 2 ===')
    plot_fold(results_combined, model_labels, ds_names, ds_colors,
              'iy011', 'IY011 (2-fold variation)',
              SCRIPT_DIR / 'IY029_simclr_svm_pairwise_combined_2fold.png',
              title_prefix='Pairwise SVM [raw + encoded]')
    plot_fold(results_combined, model_labels, ds_names, ds_colors,
              'iy014', 'IY014 (10-fold variation)',
              SCRIPT_DIR / 'IY029_simclr_svm_pairwise_combined_10fold.png',
              title_prefix='Pairwise SVM [raw + encoded]')
    plot_best_model(results_combined, model_labels, ds_names, ds_colors,
                    SCRIPT_DIR / 'IY029_simclr_svm_pairwise_combined_best_model.png',
                    suptitle_prefix='Best model [raw|z1|z2]')

    print('\nAll done.')


if __name__ == '__main__':
    main()
