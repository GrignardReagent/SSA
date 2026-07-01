"""
IY029 v2: catch22 + SVM — Pairwise Same/Different Task (IY029 datasets, n=14 000)

Converted from IY029_catch22_svm_pairwise.ipynb.
Code is unchanged; only data paths point to the regenerated IY029 static
.pt files and output filenames carry a _v2 suffix.

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Submit with: qsub IY029_catch22_svm_pairwise_v2.sh  (from the EXP-26-IY029 dir)

Feature extraction:
  - Split each concatenated pair at T//2 → full x1 and x2 (no crop)
  - pycatch22.catch22_all on each half → 22 features per half
  - Feature vector per pair: [catch22(x1) | catch22(x2)], shape (N, 44)
  - Parallelised across samples with joblib to reduce wall time

Saves:
  IY029_catch22_svm_pairwise_v2_results.json
  IY029_catch22_svm_pairwise_v2_acc.png
"""
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataloaders import load_loader_from_disk
from features.catch22 import extract_catch22_pair

IY029_DATA = SCRIPT_DIR / 'data'   # EXP-26-IY029/data/{2_fold,10_fold}/<cond>/

# ── Config ─────────────────────────────────────────────────────────────────────
# Number of parallel workers for catch22 extraction (-1 = all CPU cores)
N_JOBS = -1

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.titlesize': 14,
})

# ── Dataset config ─────────────────────────────────────────────────────────────
# Each config points to one statistic-variation dataset for both fold magnitudes,
# now loading from the regenerated IY029 static .pt files.
DATASET_CONFIGS = [
    {
        'name':        'Baseline',
        'iy011_train': IY029_DATA / '2_fold'  / 'baseline' / 'IY029_static_train.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 'baseline' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_train.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_test.pt',
    },
    {
        'name':        'Mu',
        'iy011_train': IY029_DATA / '2_fold'  / 'mu' / 'IY029_static_train.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 'mu' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 'mu' / 'IY029_static_train.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 'mu' / 'IY029_static_test.pt',
    },
    {
        'name':        'CV',
        'iy011_train': IY029_DATA / '2_fold'  / 'cv' / 'IY029_static_train.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 'cv' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 'cv' / 'IY029_static_train.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 'cv' / 'IY029_static_test.pt',
    },
    {
        'name':        'T_ac',
        'iy011_train': IY029_DATA / '2_fold'  / 't_ac' / 'IY029_static_train.pt',
        'iy011_test':  IY029_DATA / '2_fold'  / 't_ac' / 'IY029_static_test.pt',
        'iy014_train': IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_train.pt',
        'iy014_test':  IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_test.pt',
    },
]
DS_NAMES = [cfg['name'] for cfg in DATASET_CONFIGS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pair_data(train_pt: Path, test_pt: Path):
    """Load IY029 static pairwise loader → (X_train, X_test, y_train, y_test) numpy arrays."""
    def _extract(pt_path):
        loader = load_loader_from_disk(pt_path, batch_size=2048)
        Xs, ys = [], []
        for X, y in loader:
            # Saved loaders already contain concatenated trajectory pairs and 0/1 labels.
            Xs.append(X.numpy())
            ys.append(y.numpy().ravel())
        return np.concatenate(Xs), np.concatenate(ys).astype(int)

    X_train, y_train = _extract(train_pt)
    X_test,  y_test  = _extract(test_pt)
    return X_train, X_test, y_train, y_test


def run_svm(feats_tr: np.ndarray, feats_te: np.ndarray,
            y_tr: np.ndarray, y_te: np.ndarray) -> float:
    """Scale features, fit RBF SVM, return test accuracy."""
    # Fit scaling on the training fold only to avoid test-set leakage.
    scaler   = StandardScaler()
    feats_tr = scaler.fit_transform(feats_tr)
    feats_te = scaler.transform(feats_te)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(feats_tr, y_tr)
    return clf.score(feats_te, y_te)


# ── Evaluation ────────────────────────────────────────────────────────────────

def main():
    print(f'IY029_DATA : {IY029_DATA}  (exists: {IY029_DATA.exists()})')
    if not IY029_DATA.exists():
        raise FileNotFoundError(f'Data directory not found — check path above.')

    results = {}   # {ds_name: {'iy011': acc, 'iy014': acc}}

    for cfg in DATASET_CONFIGS:
        name = cfg['name']
        results[name] = {}
        print(f'\n=== {name} ===')

        # Evaluate the same dataset condition on the 2-fold and 10-fold tasks.
        for fold in ('iy011', 'iy014'):
            print(f'  {fold}: loading...', end=' ', flush=True)
            X_tr, X_te, y_tr, y_te = load_pair_data(
                cfg[f'{fold}_train'], cfg[f'{fold}_test']
            )
            print(f'train={X_tr.shape}, test={X_te.shape}  |  computing catch22...', flush=True)

            t0       = time.time()
            feats_tr = extract_catch22_pair(X_tr)
            feats_te = extract_catch22_pair(X_te)
            t1       = time.time()
            print(f'    features: {feats_tr.shape}  ({t1-t0:.1f}s)  |  fitting SVM...', flush=True)

            acc = run_svm(feats_tr, feats_te, y_tr, y_te)
            results[name][fold] = acc
            print(f'    test acc = {acc:.4f}')

    print('\nDone.')

    # ── Save JSON ──────────────────────────────────────────────────────────────
    save_path = SCRIPT_DIR / 'IY029_catch22_svm_pairwise_v2_results.json'
    with open(save_path, 'w') as f:
        json.dump(
            {ds: {fold: float(results[ds][fold]) for fold in ('iy011', 'iy014')}
             for ds in DS_NAMES},
            f, indent=2,
        )
    print(f'Saved {save_path}')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n=== Summary ===')
    rows = []
    for n in DS_NAMES:
        rows.append({
            'Dataset':         n,
            '2-fold (IY011)':  results[n]['iy011'],
            '10-fold (IY014)': results[n]['iy014'],
            'Mean':            np.mean([results[n]['iy011'], results[n]['iy014']]),
        })
    df = pd.DataFrame(rows).set_index('Dataset')
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(df.to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    palette = sns.color_palette('colorblind')
    colors  = [palette[i] for i in range(len(DS_NAMES))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    x     = np.arange(len(DS_NAMES))
    width = 0.5

    for ax, fold_key, fold_title in [
        (axes[0], 'iy011', 'IY011 (2-fold)'),
        (axes[1], 'iy014', 'IY014 (10-fold)'),
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
    fig.suptitle('catch22 + SVM — pairwise same/different accuracy (v2, n=14 000)\n'
                 'features: [catch22(x1) | catch22(x2)], 44-dim',
                 fontsize=13, weight='bold')
    plt.tight_layout()
    out_png = SCRIPT_DIR / 'IY029_catch22_svm_pairwise_v2_acc.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
