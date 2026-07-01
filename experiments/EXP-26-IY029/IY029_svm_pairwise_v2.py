"""
IY029 v2: Pure SVM — Pairwise Same/Different Task (IY029 datasets, n=14 000)

Converted from EXP-26-IY025/IY025_svm_pairwise_variation.ipynb.
Code is unchanged; only data paths point to the regenerated IY029 static
.pt files and output filenames carry a _v2 suffix.

*** Designed to run on the University of Edinburgh Eddie HPC cluster. ***
Submit with: qsub IY029_svm_pairwise_v2.sh  (from the EXP-26-IY029 dir)

Saves:
  IY029_svm_pairwise_v2_results.json
  IY029_svm_pairwise_v2_2fold.png
  IY029_svm_pairwise_v2_10fold.png
  IY029_svm_pairwise_v2_comparison.png
  IY029_svm_pairwise_v2_permutation_test.png
  IY029_svm_pairwise_v2_permutation_results.json
"""
import sys
import json
import numpy as np
import torch
import matplotlib
from utils.svm import extract_data_for_svm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataloaders import load_loader_from_disk
from classifiers.svm_classifier import svm_classifier

IY029_DATA = SCRIPT_DIR / 'data'   # EXP-26-IY029/data/{2_fold,10_fold}/<cond>/

# ── Plotting conventions ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize":  14,
})

def _colours(n):
    return sns.color_palette("colorblind", n_colors=n)


# ── Dataset config ─────────────────────────────────────────────────────────────
# 2-fold (IY011 equivalent) and 10-fold (IY014 equivalent) experiments, now
# loading from the regenerated IY029 datasets.
IY011_EXPERIMENTS = [
    {
        'name': 'baseline',
        'train': IY029_DATA / '2_fold' / 'baseline' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '2_fold' / 'baseline' / 'IY029_static_test.pt',
        'color': 'black',
    },
    {
        'name': 'mu',
        'train': IY029_DATA / '2_fold' / 'mu' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '2_fold' / 'mu' / 'IY029_static_test.pt',
        'color': 'blue',
    },
    {
        'name': 'cv',
        'train': IY029_DATA / '2_fold' / 'cv' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '2_fold' / 'cv' / 'IY029_static_test.pt',
        'color': 'green',
    },
    {
        'name': 't_ac',
        'train': IY029_DATA / '2_fold' / 't_ac' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '2_fold' / 't_ac' / 'IY029_static_test.pt',
        'color': 'red',
    },
]

IY014_EXPERIMENTS = [
    {
        'name': 'baseline',
        'train': IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '10_fold' / 'baseline' / 'IY029_static_test.pt',
        'color': 'black',
    },
    {
        'name': 'mu',
        'train': IY029_DATA / '10_fold' / 'mu' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '10_fold' / 'mu' / 'IY029_static_test.pt',
        'color': 'blue',
    },
    {
        'name': 'cv',
        'train': IY029_DATA / '10_fold' / 'cv' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '10_fold' / 'cv' / 'IY029_static_test.pt',
        'color': 'green',
    },
    {
        'name': 't_ac',
        'train': IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_train.pt',
        'test':  IY029_DATA / '10_fold' / 't_ac' / 'IY029_static_test.pt',
        'color': 'red',
    },
]


# ── Data helpers ───────────────────────────────────────────────────────────────


def run_svm_for_experiment(exp):
    """Load data, scale, and evaluate SVM. Returns (accuracy, n_test)."""
    train_loader = load_loader_from_disk(exp['train'], batch_size=256)
    test_loader  = load_loader_from_disk(exp['test'],  batch_size=256)

    X_train, y_train = extract_data_for_svm(train_loader, verbose=False)
    X_test,  y_test  = extract_data_for_svm(test_loader, verbose=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    acc = svm_classifier(X_train, X_test, y_train, y_test)
    n_test = len(y_test)
    print(f"  [{exp['name']}] SVM accuracy: {acc:.4f}  (n_test={n_test})")
    return acc, n_test


# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_svm_results(experiments, accuracies, ses, title, ax):
    """Bar chart of SVM accuracy with 95% CI error bars (normal approximation)."""
    labels  = [e['name'] for e in experiments]
    colours = _colours(len(labels))
    x = np.arange(len(labels))

    bars = ax.bar(x, accuracies, color=colours, width=0.5,
                  edgecolor='black', linewidth=0.8)
    ax.errorbar(x, accuracies, yerr=1.96 * np.array(ses),
                fmt='none', color='black', capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt='%.2g', padding=16)
    ax.axhline(0.5, color='dimgrey', linestyle='--', linewidth=1,
               label='Chance (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Varied statistic')
    ax.set_ylabel('Test accuracy (fraction correct)')
    ax.set_ylim(0, 1.2)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)


# ── Permutation test helpers ──────────────────────────────────────────────────
N_PERM = 1_000


def run_svm_returning_clf(exp):
    """Load data, scale, train RBF SVM; return (clf, X_test_scaled, y_test, accuracy)."""
    train_loader = load_loader_from_disk(exp['train'], batch_size=256)
    test_loader  = load_loader_from_disk(exp['test'],  batch_size=256)
    X_train, y_train = extract_data_for_svm(train_loader, verbose=False)
    X_test,  y_test  = extract_data_for_svm(test_loader, verbose=False)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    obs = clf.score(X_test, y_test)
    return clf, X_test, y_test, obs


def label_permutation_test(clf, X_test, y_test, n_permutations=N_PERM, seed=42):
    """Fix clf; shuffle test labels n_permutations times; return (observed, null_dist, p_value)."""
    rng      = np.random.default_rng(seed)
    observed = clf.score(X_test, y_test)
    null     = np.array([
        clf.score(X_test, rng.permutation(y_test))
        for _ in range(n_permutations)
    ])
    # Floor p-value at 1/n_perm so it is never exactly zero
    p_val = max(float((null >= observed).mean()), 1.0 / n_permutations)
    return observed, null, p_val


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f'REPO_ROOT  : {REPO_ROOT}')
    print(f'IY029_DATA : {IY029_DATA}  (exists: {IY029_DATA.exists()})')

    # ── IY011 (2-fold) ────────────────────────────────────────────────────────
    print('=== IY011: 2-fold variation ===')
    iy011_accs, iy011_ses = [], []
    for exp in IY011_EXPERIMENTS:
        print(f'\nExperiment: {exp["name"]}')
        acc, n_test = run_svm_for_experiment(exp)
        iy011_accs.append(acc)
        iy011_ses.append(np.sqrt(acc * (1 - acc) / n_test))

    print('\nSummary (IY011):')
    for exp, acc, se in zip(IY011_EXPERIMENTS, iy011_accs, iy011_ses):
        print(f'  {exp["name"]:12s}: {acc:.4f} ± {1.96*se:.4f} (95% CI)')

    # ── IY014 (10-fold) ───────────────────────────────────────────────────────
    print('=== IY014: 10-fold variation ===')
    iy014_accs, iy014_ses = [], []
    for exp in IY014_EXPERIMENTS:
        print(f'\nExperiment: {exp["name"]}')
        acc, n_test = run_svm_for_experiment(exp)
        iy014_accs.append(acc)
        iy014_ses.append(np.sqrt(acc * (1 - acc) / n_test))

    print('\nSummary (IY014):')
    for exp, acc, se in zip(IY014_EXPERIMENTS, iy014_accs, iy014_ses):
        print(f'  {exp["name"]:12s}: {acc:.4f} ± {1.96*se:.4f} (95% CI)')

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_svm_results(
        IY011_EXPERIMENTS, iy011_accs, iy011_ses,
        'SVM test accuracy — IY011 (2-fold variation)', ax
    )
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'IY029_svm_pairwise_v2_2fold.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_svm_results(
        IY014_EXPERIMENTS, iy014_accs, iy014_ses,
        'SVM test accuracy — IY014 (10-fold variation)', ax
    )
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'IY029_svm_pairwise_v2_10fold.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    plot_svm_results(IY011_EXPERIMENTS, iy011_accs, iy011_ses, 'SVM — 2-fold (IY011)', axes[0])
    plot_svm_results(IY014_EXPERIMENTS, iy014_accs, iy014_ses, 'SVM — 10-fold (IY014)', axes[1])
    axes[1].set_ylabel('')
    fig.suptitle(
        'SVM accuracy: effect of variation magnitude on single-stat experiments\n'
        'Error bars: 95% CI (normal approximation)',
        fontsize=14, weight='bold'
    )
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'IY029_svm_pairwise_v2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    _results = {
        'iy011': {exp['name']: float(acc) for exp, acc in zip(IY011_EXPERIMENTS, iy011_accs)},
        'iy014': {exp['name']: float(acc) for exp, acc in zip(IY014_EXPERIMENTS, iy014_accs)},
    }
    with open(SCRIPT_DIR / 'IY029_svm_pairwise_v2_results.json', 'w') as _f:
        json.dump(_results, _f, indent=2)
    print('Saved IY029_svm_pairwise_v2_results.json')
    print(_results)

    # ── Permutation test ──────────────────────────────────────────────────────
    print(f'Running permutation tests (N_PERM={N_PERM}) …')
    perm_results = {}
    perm_nulls   = {}

    for fold_label, experiments in [('iy011', IY011_EXPERIMENTS), ('iy014', IY014_EXPERIMENTS)]:
        perm_results[fold_label] = {}
        perm_nulls[fold_label]   = {}
        for exp in experiments:
            print(f'  [{fold_label}/{exp["name"]}] fitting SVM …', flush=True)
            clf, X_te, y_te, obs = run_svm_returning_clf(exp)
            print(f'  [{fold_label}/{exp["name"]}] running {N_PERM} permutations …', flush=True)
            observed, null, p_val = label_permutation_test(clf, X_te, y_te)
            perm_results[fold_label][exp['name']] = {
                'observed':  float(observed),
                'null_mean': float(null.mean()),
                'null_std':  float(null.std()),
                'p_value':   float(p_val),
            }
            perm_nulls[fold_label][exp['name']] = null
            print(f'    obs={observed:.4f}  null={null.mean():.4f}±{null.std():.4f}  p={p_val:.4f}')

    print('\nDone.')

    ds_names_plot  = [e['name'] for e in IY011_EXPERIMENTS]
    fold_labels    = [('iy011', 'IY011 (2-fold)'), ('iy014', 'IY014 (10-fold)')]
    palette        = sns.color_palette('colorblind', n_colors=len(ds_names_plot))

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for row_idx, (fold_key, fold_title) in enumerate(fold_labels):
        for col_idx, ds_name in enumerate(ds_names_plot):
            ax   = axes[row_idx, col_idx]
            res  = perm_results[fold_key][ds_name]
            null = perm_nulls[fold_key][ds_name]
            obs  = res['observed']
            pv   = res['p_value']

            ax.hist(null, bins=30, color=palette[col_idx], alpha=0.7, edgecolor='white')
            ax.axvline(obs, color='black', linewidth=2, label=f'observed: {obs:.2f}')
            ax.axvline(0.5, color='dimgrey', linewidth=1, linestyle='--', label='chance')

            p_str = f'p = {pv:.3f}' if pv >= 0.001 else 'p < 0.001'
            ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_xlabel('Permuted accuracy', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f'{fold_title}\nCount', fontsize=11)
            ax.set_title(ds_name, fontsize=13)
            ax.legend(fontsize=9, loc='upper left')

    fig.suptitle(
        'Label permutation test — SVM pairwise accuracy\n'
        f'Null: test labels shuffled {N_PERM:,}×; observed: vertical black line',
        fontsize=14, weight='bold',
    )
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'IY029_svm_pairwise_v2_permutation_test.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved IY029_svm_pairwise_v2_permutation_test.png')

    ds_names_ordered = [e['name'] for e in IY011_EXPERIMENTS]
    print(f'{"Fold":<8}  {"Dataset":<12}  {"Observed":>10}  {"Null mean":>10}  {"Null SD":>8}  {"p-value":>10}')
    print('-' * 68)
    for fold_key, fold_title in [('iy011', 'IY011'), ('iy014', 'IY014')]:
        for ds_name in ds_names_ordered:
            r   = perm_results[fold_key][ds_name]
            sig = ' *' if r['p_value'] < 0.05 else '  '
            print(f'{fold_title:<8}  {ds_name:<12}  {r["observed"]:>10.4f}  '
                  f'{r["null_mean"]:>10.4f}  {r["null_std"]:>8.4f}  {r["p_value"]:>10.4f}{sig}')

    with open(SCRIPT_DIR / 'IY029_svm_pairwise_v2_permutation_results.json', 'w') as _f:
        json.dump(perm_results, _f, indent=2)
    print('\nSaved IY029_svm_pairwise_v2_permutation_results.json')


if __name__ == '__main__':
    main()
