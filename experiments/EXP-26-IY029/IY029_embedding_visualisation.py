"""
IY029: Pairwise Embedding Space Visualisation
==============================================
Computes PCA and t-SNE projections for the latent embeddings of all 6 pairwise
same/different methods, across all 8 (condition × fold) test sets.

Separation quality is quantified with three metrics:
  • D-score  — Fisher multivariate discriminability: ||μ₁−μ₀||² / (tr(Σ₁)+tr(Σ₀))
  • ARI      — Adjusted Rand Index (k-means k=2 vs true labels)
  • NMI      — Normalised Mutual Information (k-means k=2 vs true labels)

Memory strategy (OOM fix)
--------------------------
All data and embeddings are processed one (cond, fold) pair at a time.
Each embedding is written to `embeddings_cache/` as a .npy file immediately
after extraction, and the raw data arrays are freed before moving to the next
pair.  Plotting loads from cache files — one method at a time — so peak RAM
equals one train set + one test set + one embedding batch rather than all 8.

Outputs
-------
  embeddings_cache/{method}_{cond}_{fold}.npy   — (N_test, D) float32
  embeddings_cache/labels_{cond}_{fold}.npy     — (N_test,) int
  IY029_embedding_metrics.json                  — D-score / ARI / NMI for all combos
  IY029_embedding_{method}_pca.png              — one 4×2 figure per method
  IY029_embedding_{method}_tsne.png             — one 4×2 figure per method
  IY029_embedding_dscore_heatmap.png            — D-score summary heatmap
  IY029_embedding_ari_heatmap.png               — ARI summary heatmap
  IY029_embedding_nmi_heatmap.png               — NMI summary heatmap

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Submit with: qsub IY029_embedding_visualisation.sh  (from the EXP-26-IY029 dir)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import re
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataloaders import load_loader_from_disk
from features.catch22 import catch22_features
from models.ssl_transformer import SSL_Transformer
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier

IY029_DATA = SCRIPT_DIR / 'data'
CKPT_DIR   = SCRIPT_DIR / 'checkpoints'
CACHE_DIR  = SCRIPT_DIR / 'embeddings_cache'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset & display config ──────────────────────────────────────────────────
CONDITIONS   = ['baseline', 'mu', 'cv', 't_ac']
FOLDS        = ['2_fold', '10_fold']
COND_NAMES   = ['Baseline', 'Mu', 'CV', 'T_ac']
FOLD_DISPLAY = ['2-fold', '10-fold']
N_JOBS_C22   = -1   # joblib workers for catch22

METHOD_DISPLAY = [
    ('svm',         'Raw SVM'),
    ('catch22',     'catch22 + SVM'),
    ('simclr',      'SimCLR + SVM'),
    ('mlp',         'MLP'),
    ('transformer', 'Transformer'),
    ('lstm',        'LSTM'),
]

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'axes.titlesize':  11,
})
palette  = sns.color_palette('colorblind')
COL_SAME = palette[0]   # blue      — same (y=1)
COL_DIFF = palette[3]   # vermilion — different (y=0)

# ── SimCLR model registry ─────────────────────────────────────────────────────
NORM_OVERRIDES = {
    'IY022_simCLR_b64_lr0.01_L2_H4_D16_20260303_170229_model': 'global',
    'IY022_simCLR_b64_lr0.01_L2_H4_D16_20260308_125632_model': 'global',
    'IY022_simCLR_b64_lr0.01_L2_H4_D16_20260308_132219_model': 'joint',
    'IY023_simCLR_b64_lr0.01_L2_H4_D16_20260308_125550_model': 'global',
    'IY023_simCLR_b64_lr0.01_L2_H4_D16_20260308_165118_model': 'joint',
}
SIMCLR_CKPT_DIRS = ['EXP-26-IY017', 'EXP-26-IY022', 'EXP-26-IY023', 'EXP-26-IY024']


# ── Architecture helpers ───────────────────────────────────────────────────────

def parse_arch(name: str) -> dict:
    return dict(
        input_size=1,
        d_model    = int(re.search(r'_D(\d+)', name).group(1)),
        nhead      = int(re.search(r'_H(\d+)', name).group(1)),
        num_layers = int(re.search(r'_L(\d+)', name).group(1)),
        dropout=0.01, use_conv1d=False,
    )


def make_label(path: Path) -> str:
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


# ── MLP architecture (must match IY029_mlp_pairwise_v2.py) ────────────────────

class PairwiseMLP(nn.Module):
    """MLP for pairwise same/different; exposes encode() for penultimate layer."""
    def __init__(self, input_dim: int, hidden_dims: tuple = (512, 256, 64),
                 dropout: float = 0.3):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

    def encode(self, x):
        """64-dim penultimate activations (Dropout is identity in eval mode)."""
        return self.net[:-1](x)


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_pt(pt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a .pt static loader → (X_np float32, y_np int)."""
    loader = load_loader_from_disk(pt_path, batch_size=2048)
    Xs, ys = [], []
    for X, y in loader:
        Xs.append(X.numpy()); ys.append(y.numpy().ravel())
    return np.concatenate(Xs), np.concatenate(ys).astype(int)


def load_split(cond: str, fold: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one (cond, fold, split) triple from the IY029 v2 .pt files."""
    pt = IY029_DATA / fold / cond / f'IY029_static_{split}.pt'
    if not pt.exists():
        raise FileNotFoundError(
            f'File not found: {pt}\n'
            'Run IY029_regen_pt_datasets.sh on HPC first.'
        )
    return _load_pt(pt)


# ── catch22 helpers ────────────────────────────────────────────────────────────


# ── Per-method embedding extractors ───────────────────────────────────────────

def embed_svm(X_train_np: np.ndarray, X_test_np: np.ndarray) -> np.ndarray:
    """Flatten pairs and z-score via a StandardScaler fit on the training set."""
    X_tr = X_train_np.reshape(len(X_train_np), -1)
    X_te = X_test_np.reshape(len(X_test_np),   -1)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_te).astype(np.float32)          # (N, T)


def embed_catch22(X_train_np: np.ndarray, X_test_np: np.ndarray) -> np.ndarray:
    """[catch22(x1)|catch22(x2)] + StandardScaler. Returns (N, 44)."""
    print('    catch22 train...', end='', flush=True); t0 = time.time()
    f_tr = catch22_features(X_train_np)
    print(f' {time.time()-t0:.0f}s | test...', end='', flush=True); t0 = time.time()
    f_te = catch22_features(X_test_np)
    print(f' {time.time()-t0:.0f}s', flush=True)
    scaler = StandardScaler().fit(f_tr)
    return scaler.transform(f_te).astype(np.float32)           # (N, 44)


def embed_simclr(X_test_np: np.ndarray, model: SSL_Transformer,
                 batch_size: int = 128) -> np.ndarray:
    """Encode pairs as [z1|z2] via best SimCLR backbone. Returns (N, 2*d_model)."""
    model.eval()
    half = X_test_np.shape[1] // 2
    x1, x2 = X_test_np[:, :half, :], X_test_np[:, half:, :]
    z1_list, z2_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_test_np), batch_size):
            b1 = torch.tensor(x1[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            b2 = torch.tensor(x2[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            z1_list.append(model.backbone.encode(b1).cpu().numpy())
            z2_list.append(model.backbone.encode(b2).cpu().numpy())
    z1 = np.concatenate(z1_list); z2 = np.concatenate(z2_list)
    return np.concatenate([z1, z2], axis=1).astype(np.float32)  # (N, 2*d_model)


def embed_mlp(X_test_np: np.ndarray, model: PairwiseMLP,
              batch_size: int = 512) -> np.ndarray:
    """64-dim penultimate activations via model.encode()."""
    model.eval()
    X_flat = X_test_np.reshape(len(X_test_np), -1)
    X_t    = torch.tensor(X_flat, dtype=torch.float32)
    parts  = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            parts.append(model.encode(X_t[i:i+batch_size].to(DEVICE)).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)            # (N, 64)


def embed_transformer(X_test_np: np.ndarray, model: TransformerClassifier,
                      batch_size: int = 32) -> np.ndarray:
    """d_model-dim mean-pooled Transformer encoding via model.encode()."""
    model.eval()
    X_t   = torch.tensor(X_test_np, dtype=torch.float32)
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            parts.append(
                model.encode(X_t[i:i+batch_size].to(DEVICE)).cpu().numpy()
            )
    return np.concatenate(parts).astype(np.float32)            # (N, d_model)


def embed_lstm(X_test_np: np.ndarray, model: LSTMClassifier,
               batch_size: int = 64) -> np.ndarray:
    """
    128-dim LSTM context vector extracted via a forward pre-hook on fc_layers.
    Captures the input to the classification head (bidirectional, hidden=64).
    """
    model.eval()
    X_t      = torch.tensor(X_test_np, dtype=torch.float32)
    contexts = []

    def _hook(module, inputs):
        contexts.append(inputs[0].detach().cpu().numpy())

    handle = model.fc_layers.register_forward_pre_hook(_hook)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            model(X_t[i:i+batch_size].to(DEVICE))
    handle.remove()
    return np.concatenate(contexts).astype(np.float32)         # (N, 128)


# ── SimCLR model loader ────────────────────────────────────────────────────────

def load_simclr_best() -> SSL_Transformer:
    """Load the best SimCLR backbone from the v2 (or original) results JSON."""
    registry = [
        (p, make_label(p))
        for d in SIMCLR_CKPT_DIRS
        for p in sorted((REPO_ROOT / 'experiments' / d).glob('*.pth'))
    ]
    print(f'SimCLR checkpoints found: {len(registry)}')

    for json_path in [
        SCRIPT_DIR / 'IY029_simclr_svm_pairwise_v2_results.json',
        SCRIPT_DIR / 'IY029_simclr_svm_pairwise_results.json',
    ]:
        if json_path.exists():
            best_label = json.load(open(json_path))['best_label']
            print(f'SimCLR best label ({json_path.name}): {best_label}')
            matches = [(p, lbl) for p, lbl in registry if lbl == best_label]
            if matches:
                ckpt_path, _ = matches[0]
                arch  = parse_arch(ckpt_path.stem)
                model = SSL_Transformer(**arch).to(DEVICE)
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state)
                model.eval()
                print(f'  Loaded: {ckpt_path.name}')
                return model

    print('No SimCLR results JSON found; using first checkpoint in registry.')
    ckpt_path, lbl = registry[0]
    arch  = parse_arch(ckpt_path.stem)
    model = SSL_Transformer(**arch).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


_FOLD_TO_CKPT = {'2_fold': 'iy011', '10_fold': 'iy014'}


def load_supervised_model(method: str, cond: str, fold: str):
    """
    Load one MLP/Transformer/LSTM checkpoint. Returns None if missing.
    Models are loaded to CPU to avoid GPU memory fragmentation across 8 checkpoints.
    Checkpoint filenames use iy011/iy014 (from training scripts), while FOLDS uses
    2_fold/10_fold (matching the data directory layout) — _FOLD_TO_CKPT bridges them.
    """
    key_sfx = f'{cond}_{_FOLD_TO_CKPT[fold]}'
    if method == 'mlp':
        ckpt = CKPT_DIR / f'IY029_mlp_v2_{key_sfx}.pth'
        if not ckpt.exists():
            return None
        data  = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = PairwiseMLP(data['input_dim']).eval().to(DEVICE)
        model.load_state_dict(data['state_dict'])
        return model

    elif method == 'transformer':
        ckpt = CKPT_DIR / f'IY029_transformer_v2_{key_sfx}.pth'
        if not ckpt.exists():
            return None
        data  = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = TransformerClassifier(
                    input_size=1, d_model=data['d_model'], nhead=data['nhead'],
                    num_layers=data['num_layers'], num_classes=2,
                    dropout=data['dropout'],
                ).eval().to(DEVICE)
        model.load_state_dict(data['state_dict'])
        return model

    elif method == 'lstm':
        ckpt = CKPT_DIR / f'IY029_lstm_v2_{key_sfx}.pth'
        if not ckpt.exists():
            return None
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = LSTMClassifier(
                    input_size=1, hidden_size=64, num_layers=2, output_size=2,
                    dropout_rate=0.3, learning_rate=1e-3,
                    use_conv1d=False, use_attention=False,
                    bidirectional=True, verbose=False,
                ).eval().to(DEVICE)
        model.load_state_dict(state)
        return model

    return None


# ── Dimensionality reduction ───────────────────────────────────────────────────

def reduce_2d(emb: np.ndarray, method: str = 'PCA',
              pca_pre: int = 50, tsne_perp: int = 30,
              random_state: int = 42) -> np.ndarray:
    """
    Reduce (N, D) → (N, 2) via PCA or t-SNE.
    t-SNE: PCA pre-reduction to pca_pre dims when D > pca_pre, to cap memory.
    """
    D = emb.shape[1]
    if method == 'PCA':
        n = min(2, D)
        return PCA(n_components=n, random_state=random_state).fit_transform(emb)
    else:  # t-SNE
        if D > pca_pre:
            n_pre = min(pca_pre, D)
            emb = PCA(n_components=n_pre, random_state=random_state).fit_transform(emb)
        return TSNE(
            n_components=2, perplexity=tsne_perp,
            random_state=random_state, n_jobs=-1,
        ).fit_transform(emb)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_embedding_grid(method_key: str, method_title: str,
                        reduction: str, metrics: dict,
                        fname: Path | None = None):
    """
    4 × 2 figure (conditions × folds) for one method and one reduction type.
    D-score / ARI / NMI read from precomputed metrics dict; displayed in each panel title.
    Cache files (embeddings_cache/{method_key}_{cond}_{fold}.npy) loaded one
    at a time to minimise peak memory during plotting.
    """
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    legend_handles = [
        mpatches.Patch(color=COL_SAME, label='Same (y=1)'),
        mpatches.Patch(color=COL_DIFF, label='Different (y=0)'),
    ]
    any_plotted = False

    for row, cond in enumerate(CONDITIONS):
        for col, fold in enumerate(FOLDS):
            ax  = axes[row, col]
            key = f'{cond}_{fold}'

            emb_path    = CACHE_DIR / f'{method_key}_{key}.npy'
            labels_path = CACHE_DIR / f'labels_{key}.npy'

            if not emb_path.exists() or not labels_path.exists():
                ax.text(0.5, 0.5, 'No checkpoint', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='grey')
                ax.set_title(f'{COND_NAMES[row]}  {FOLD_DISPLAY[col]}', fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            emb = np.load(emb_path)
            y   = np.load(labels_path)

            # 2-D projection
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                emb2d = reduce_2d(emb, method=reduction)

            # Retrieve precomputed metrics for this panel
            mkey    = f'{method_key}_{key}'
            d_val   = metrics['d_score'].get(mkey)
            ari_val = metrics['ari'].get(mkey)
            nmi_val = metrics['nmi'].get(mkey)
            metric_str = (f'D={d_val:.3f}  ARI={ari_val:.3f}  NMI={nmi_val:.3f}'
                          if d_val is not None else '')

            # Scatter: different first so same is rendered on top
            mask_diff = y == 0; mask_same = y == 1
            ax.scatter(emb2d[mask_diff, 0], emb2d[mask_diff, 1],
                       c=[COL_DIFF], s=5, alpha=0.35, linewidths=0, rasterized=True)
            ax.scatter(emb2d[mask_same, 0], emb2d[mask_same, 1],
                       c=[COL_SAME], s=5, alpha=0.35, linewidths=0, rasterized=True)

            ax.set_title(
                f'{COND_NAMES[row]}  {FOLD_DISPLAY[col]}'
                + (f'\n{metric_str}' if metric_str else ''),
                fontsize=10,
            )
            ax.set_xlabel(f'{reduction}1', fontsize=9)
            ax.set_ylabel(f'{reduction}2', fontsize=9)
            ax.tick_params(labelsize=8)

            # Free the loaded arrays immediately after plotting this panel
            del emb, emb2d, y
            any_plotted = True

    if not any_plotted:
        print(f'⚠️  No embeddings cached for {method_key} — skipping plot.')
        plt.close(fig)
        return

    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(0.98, 0.995), fontsize=10, framealpha=0.9)
    fig.suptitle(
        f'{method_title} — {reduction} of test-set pairwise embeddings',
        fontsize=14, weight='bold', y=1.002,
    )
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Saved {fname.name}')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'REPO_ROOT  : {REPO_ROOT}')
    print(f'IY029_DATA : {IY029_DATA}  (exists: {IY029_DATA.exists()})')
    print(f'CKPT_DIR   : {CKPT_DIR}   (exists: {CKPT_DIR.exists()})')
    print(f'Device     : {DEVICE}')
    if not IY029_DATA.exists():
        raise FileNotFoundError('Data directory not found. Check IY029_DATA path.')

    CACHE_DIR.mkdir(exist_ok=True)
    print(f'Cache dir  : {CACHE_DIR}')

    # ── Load SimCLR backbone (single shared model — small enough to keep loaded) ──
    simclr_model = load_simclr_best()

    # ── Phase 1: compute & cache embeddings, one (cond, fold) at a time ──────
    # Loading one pair at a time keeps peak RAM ≈ 1 train set + 1 test set.
    metrics  = {'d_score': {}, 'ari': {}, 'nmi': {}}   # {metric: {key: float}}
    t_phase1 = time.time()

    for cond in CONDITIONS:
        for fold in FOLDS:
            key_sfx = f'{cond}_{fold}'
            print(f'\n══ {cond} / {fold} ══', flush=True)

            # Load data for this pair only
            t0 = time.time()
            X_tr, _    = load_split(cond, fold, 'train')
            X_te, y_te = load_split(cond, fold, 'test')
            print(f'  Data loaded in {time.time()-t0:.1f}s  '
                  f'train={X_tr.shape}  test={X_te.shape}', flush=True)

            # Save labels once
            labels_path = CACHE_DIR / f'labels_{key_sfx}.npy'
            if not labels_path.exists():
                np.save(labels_path, y_te)

            # --- SVM ---
            emb_path = CACHE_DIR / f'svm_{key_sfx}.npy'
            if emb_path.exists():
                print('  SVM: cache hit', flush=True)
                emb = np.load(emb_path)
            else:
                print('  SVM...', end=' ', flush=True); t0 = time.time()
                emb = embed_svm(X_tr, X_te)
                np.save(emb_path, emb)
                print(f'{time.time()-t0:.1f}s  shape={emb.shape}')
            _compute_metrics(metrics, 'svm', key_sfx, emb, y_te)
            del emb

            # --- catch22 ---
            emb_path = CACHE_DIR / f'catch22_{key_sfx}.npy'
            if emb_path.exists():
                print('  catch22: cache hit', flush=True)
                emb = np.load(emb_path)
            else:
                print('  catch22...', flush=True)
                emb = embed_catch22(X_tr, X_te)
                np.save(emb_path, emb)
                print(f'    shape={emb.shape}')
            _compute_metrics(metrics, 'catch22', key_sfx, emb, y_te)
            del emb

            # Free training data — not needed beyond SVM / catch22
            del X_tr; gc.collect()

            # --- SimCLR ---
            emb_path = CACHE_DIR / f'simclr_{key_sfx}.npy'
            if emb_path.exists():
                print('  SimCLR: cache hit', flush=True)
                emb = np.load(emb_path)
            else:
                print('  SimCLR...', end=' ', flush=True); t0 = time.time()
                emb = embed_simclr(X_te, simclr_model)
                np.save(emb_path, emb)
                print(f'{time.time()-t0:.1f}s  shape={emb.shape}')
            _compute_metrics(metrics, 'simclr', key_sfx, emb, y_te)
            del emb

            # --- MLP ---
            emb_path = CACHE_DIR / f'mlp_{key_sfx}.npy'
            if emb_path.exists():
                print('  MLP: cache hit', flush=True)
                emb = np.load(emb_path)
                _compute_metrics(metrics, 'mlp', key_sfx, emb, y_te)
                del emb
            else:
                mlp_model = load_supervised_model('mlp', cond, fold)
                if mlp_model is not None:
                    print('  MLP...', end=' ', flush=True); t0 = time.time()
                    emb = embed_mlp(X_te, mlp_model)
                    np.save(emb_path, emb)
                    print(f'{time.time()-t0:.1f}s  shape={emb.shape}')
                    _compute_metrics(metrics, 'mlp', key_sfx, emb, y_te)
                    del emb, mlp_model
                else:
                    print('  MLP: ⚠️  no checkpoint', flush=True)

            # --- Transformer ---
            emb_path = CACHE_DIR / f'transformer_{key_sfx}.npy'
            if emb_path.exists():
                print('  Transformer: cache hit', flush=True)
                emb = np.load(emb_path)
                _compute_metrics(metrics, 'transformer', key_sfx, emb, y_te)
                del emb
            else:
                tfm_model = load_supervised_model('transformer', cond, fold)
                if tfm_model is not None:
                    print('  Transformer...', end=' ', flush=True); t0 = time.time()
                    emb = embed_transformer(X_te, tfm_model)
                    np.save(emb_path, emb)
                    print(f'{time.time()-t0:.1f}s  shape={emb.shape}')
                    _compute_metrics(metrics, 'transformer', key_sfx, emb, y_te)
                    del emb, tfm_model
                else:
                    print('  Transformer: ⚠️  no checkpoint', flush=True)

            # --- LSTM ---
            emb_path = CACHE_DIR / f'lstm_{key_sfx}.npy'
            if emb_path.exists():
                print('  LSTM: cache hit', flush=True)
                emb = np.load(emb_path)
                _compute_metrics(metrics, 'lstm', key_sfx, emb, y_te)
                del emb
            else:
                lstm_model = load_supervised_model('lstm', cond, fold)
                if lstm_model is not None:
                    print('  LSTM...', end=' ', flush=True); t0 = time.time()
                    emb = embed_lstm(X_te, lstm_model)
                    np.save(emb_path, emb)
                    print(f'{time.time()-t0:.1f}s  shape={emb.shape}')
                    _compute_metrics(metrics, 'lstm', key_sfx, emb, y_te)
                    del emb, lstm_model
                else:
                    print('  LSTM: ⚠️  no checkpoint', flush=True)

            # Free test data and GPU cache before the next pair
            del X_te, y_te
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    print(f'\nPhase 1 done in {(time.time()-t_phase1)/60:.1f} min', flush=True)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = SCRIPT_DIR / 'IY029_embedding_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved {metrics_path.name}')

    # ── Phase 2: PCA plots ─────────────────────────────────────────────────────
    print('\n── Phase 2: PCA plots ──', flush=True)
    for method_key, method_title in METHOD_DISPLAY:
        print(f'\n▶ PCA — {method_title}', flush=True)
        plot_embedding_grid(
            method_key, method_title, 'PCA', metrics,
            fname=SCRIPT_DIR / f'IY029_embedding_{method_key}_pca.png',
        )
        gc.collect()

    # ── Phase 3: t-SNE plots ───────────────────────────────────────────────────
    print('\n── Phase 3: t-SNE plots ──', flush=True)
    for method_key, method_title in METHOD_DISPLAY:
        print(f'\n▶ t-SNE — {method_title}', flush=True)
        plot_embedding_grid(
            method_key, method_title, 'tSNE', metrics,
            fname=SCRIPT_DIR / f'IY029_embedding_{method_key}_tsne.png',
        )
        gc.collect()

    # ── Phase 4: metric heatmaps ───────────────────────────────────────────────
    print('\n── Phase 4: metric heatmaps ──', flush=True)
    _plot_all_heatmaps(metrics)

    print('\nDone.')


def _d_score(emb: np.ndarray, y: np.ndarray) -> float:
    """Fisher multivariate D-score: ||μ₁−μ₀||² / (tr(Σ₁)+tr(Σ₀))."""
    mu1    = emb[y == 1].mean(0)
    mu0    = emb[y == 0].mean(0)
    between = float(np.linalg.norm(mu1 - mu0) ** 2)
    within  = float(emb[y == 1].var(0).sum() + emb[y == 0].var(0).sum())
    return between / (within + 1e-12)


def _cluster_metrics(emb: np.ndarray, y: np.ndarray,
                     sample_size: int = 2000) -> tuple[float, float]:
    """ARI and NMI from k-means (k=2) clustering on a random subsample."""
    if len(emb) > sample_size:
        rng     = np.random.default_rng(42)
        idx     = rng.choice(len(emb), sample_size, replace=False)
        emb, y  = emb[idx], y[idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pred = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(emb)
    return (float(adjusted_rand_score(y, pred)),
            float(normalized_mutual_info_score(y, pred)))


def _compute_metrics(metrics: dict, method: str, key_sfx: str,
                     emb: np.ndarray, y: np.ndarray):
    """Compute D-score, ARI and NMI; store under metrics[metric][key]."""
    if emb.shape[1] >= 2 and len(np.unique(y)) == 2 and emb.shape[0] > 2:
        key = f'{method}_{key_sfx}'
        metrics['d_score'][key]  = _d_score(emb, y)
        ari, nmi                 = _cluster_metrics(emb, y)
        metrics['ari'][key]      = ari
        metrics['nmi'][key]      = nmi


# Metric display config: (key, display_title, cmap, vmin, vmax or None, output filename)
_METRIC_CFG = [
    ('d_score', 'D-score  (Fisher discriminability)',
     'YlOrRd', 0.0, None,  'IY029_embedding_dscore_heatmap.png'),
    ('ari',     'ARI  (k-means k=2 vs true labels)',
     'RdYlGn', -0.1, 1.0,  'IY029_embedding_ari_heatmap.png'),
    ('nmi',     'NMI  (k-means k=2 vs true labels)',
     'YlGn',    0.0, 1.0,  'IY029_embedding_nmi_heatmap.png'),
]


def _plot_metric_heatmap(scores: dict, metric_key: str, title: str,
                         cmap: str, vmin: float, vmax,
                         fname: Path):
    """6×8 heatmap (methods × condition-fold) for one embedding separation metric."""
    col_labels = [f'{COND_NAMES[i]}\n{FOLD_DISPLAY[j]}'
                  for i in range(len(CONDITIONS))
                  for j in range(len(FOLDS))]
    row_labels = [t for _, t in METHOD_DISPLAY]

    matrix = np.full((len(METHOD_DISPLAY), len(col_labels)), np.nan)
    for r, (method_key, _) in enumerate(METHOD_DISPLAY):
        c_idx = 0
        for cond in CONDITIONS:
            for fold in FOLDS:
                k = f'{method_key}_{cond}_{fold}'
                if k in scores:
                    matrix[r, c_idx] = scores[k]
                c_idx += 1

    # For D-score (unbounded), cap at 99th percentile for colour legibility
    eff_vmax = (vmax if vmax is not None
                else float(np.nanpercentile(matrix, 99))
                if not np.all(np.isnan(matrix)) else 1.0)

    fig, ax = plt.subplots(figsize=(14, 5))
    df      = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    mask    = np.isnan(matrix)

    sns.heatmap(
        df, ax=ax, mask=mask,
        cmap=cmap, vmin=vmin, vmax=eff_vmax,
        annot=True, fmt='.3f', annot_kws={'fontsize': 9},
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': metric_key, 'shrink': 0.8},
    )

    # Grey N/A panels for missing checkpoints
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if mask[r, c]:
                ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                           fill=True, color='lightgrey',
                                           lw=0, zorder=2))
                ax.text(c + 0.5, r + 0.5, 'N/A', ha='center', va='center',
                        fontsize=8, color='grey', zorder=3)

    ax.set_title(f'Same / Different embedding separation — {title}',
                 fontsize=14, weight='bold', pad=12)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=10)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  Saved {fname.name}')
    plt.close(fig)


def _plot_all_heatmaps(metrics: dict):
    """Produce one heatmap figure per metric (D-score, ARI, NMI)."""
    for metric_key, title, cmap, vmin, vmax, fname_str in _METRIC_CFG:
        _plot_metric_heatmap(
            metrics[metric_key], metric_key, title, cmap, vmin, vmax,
            SCRIPT_DIR / fname_str,
        )


if __name__ == '__main__':
    main()
