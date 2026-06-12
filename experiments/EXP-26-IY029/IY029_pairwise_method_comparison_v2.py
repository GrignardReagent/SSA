"""
IY029 v2: Pairwise Same/Different — Method Comparison (v2 datasets, n=14 000)

Aggregates test accuracy from all 6 pairwise classification methods evaluated
on the IY029 v2 static datasets (2-fold and 10-fold variation, n=14 000 pairs)
across four conditions.

Mirrors IY029_pairwise_method_comparison.ipynb but uses v2 results throughout.

Saves:
  IY029_pairwise_method_comparison_v2.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.family':     'sans-serif',
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize':  14,
})

SCRIPT_DIR = Path(__file__).resolve().parent

# Canonical dataset order and name mapping (SVM v2 uses lowercase keys)
DS_NAMES = ['Baseline', 'Mu', 'CV', 'T_ac']
KEY_MAP  = {'baseline': 'Baseline', 'mu': 'Mu', 'cv': 'CV', 't_ac': 'T_ac'}
FOLDS    = [('iy011', 'IY029 (2-fold)'), ('iy014', 'IY029 (10-fold)')]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_flat(path):
    """Load a {DS: {iy011, iy014}} JSON (catch22, mlp, lstm, transformer)."""
    return json.load(open(path))


def load_svm_v2(path):
    """SVM v2 stores {iy011: {ds: acc}, iy014: {ds: acc}} — invert to {DS: {fold: acc}}."""
    raw = json.load(open(path))
    return {
        KEY_MAP[ds]: {'iy011': raw['iy011'][ds], 'iy014': raw['iy014'][ds]}
        for ds in raw['iy011']
    }


def load_simclr_best(path):
    """SimCLR JSON: {best_label, results: {label: {DS: {fold: acc}}}}; return best."""
    d = json.load(open(path))
    best = d['best_label']
    print(f'SimCLR best model: {best}')
    return d['results'][best], best


# ── Load all results ───────────────────────────────────────────────────────────

raw_svm_data                   = load_svm_v2(SCRIPT_DIR / 'IY029_svm_pairwise_v2_results.json')
catch22_data                   = load_flat(SCRIPT_DIR   / 'IY029_catch22_svm_pairwise_v2_results.json')
mlp_data                       = load_flat(SCRIPT_DIR   / 'IY029_mlp_pairwise_v2_results.json')
transformer_data               = load_flat(SCRIPT_DIR   / 'IY029_transformer_pairwise_v2_results.json')
lstm_data                      = load_flat(SCRIPT_DIR   / 'IY029_lstm_pairwise_v2_results.json')
simclr_data, simclr_best_label = load_simclr_best(SCRIPT_DIR / 'IY029_simclr_svm_pairwise_v2_results.json')

METHODS = [
    ('Raw SVM',        raw_svm_data),
    ('Catch22 + SVM',  catch22_data),
    ('SimCLR + SVM',   simclr_data),
    ('MLP',            mlp_data),
    ('LSTM',           lstm_data),
    ('Transformer',    transformer_data),
]

# Sanity-check: print all accuracies
print(f'\n{"Method":<20}  {"Fold":<8}  ' + '  '.join(f'{ds:<10}' for ds in DS_NAMES))
print('-' * 74)
for label, data in METHODS:
    for fold_key, fold_title in FOLDS:
        accs = [data[ds][fold_key] for ds in DS_NAMES]
        print(f'{label:<20}  {fold_key:<8}  ' + '  '.join(f'{a:<10.3f}' for a in accs))


# ── Plot ───────────────────────────────────────────────────────────────────────

palette   = sns.color_palette('colorblind', n_colors=len(METHODS))
n_methods = len(METHODS)
width     = 0.12
offsets   = np.linspace(-(n_methods - 1) / 2 * width,
                         (n_methods - 1) / 2 * width, n_methods)
x         = np.arange(len(DS_NAMES))

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

for ax, (fold_key, fold_title) in zip(axes, FOLDS):
    for j, (label, data) in enumerate(METHODS):
        accs = [data[ds][fold_key] for ds in DS_NAMES]
        bars = ax.bar(x + offsets[j], accs, width=width,
                      color=palette[j], label=label,
                      edgecolor='white', linewidth=0.4)
        ax.bar_label(bars, fmt='%.2f', padding=2,
                     fontsize=7, rotation=90, label_type='edge')

    ax.axhline(0.5, color='dimgrey', linestyle='--', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(DS_NAMES, fontsize=11)
    ax.set_xlabel('Varied statistic', fontsize=12)
    ax.set_ylim(0, 1.38)
    ax.set_title(fold_title, fontsize=13)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

axes[0].set_ylabel('Test accuracy', fontsize=12)

legend_handles = [
    mpatches.Patch(color=palette[j], label=label)
    for j, (label, _) in enumerate(METHODS)
] + [plt.Line2D([0], [0], color='dimgrey', linestyle='--',
                linewidth=1.2, label='Chance (50%)')]
axes[1].legend(handles=legend_handles, fontsize=10,
               bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

fig.suptitle(
    f'Pairwise same/different — method comparison (v2, n=14 000)\n'
    f'SimCLR best model: {simclr_best_label}',
    fontsize=13, weight='bold',
)
plt.tight_layout()
out_path = SCRIPT_DIR / 'IY029_pairwise_method_comparison_v2.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {out_path}')


# ── Summary table ──────────────────────────────────────────────────────────────

rows = []
for label, data in METHODS:
    for fold_key, fold_title in FOLDS:
        row = {'Method': label, 'Fold': fold_title}
        for ds in DS_NAMES:
            row[ds] = data[ds][fold_key]
        row['Mean'] = np.mean([data[ds][fold_key] for ds in DS_NAMES])
        rows.append(row)

df = pd.DataFrame(rows).set_index(['Method', 'Fold'])
pd.set_option('display.float_format', '{:.3f}'.format)
print('\n' + df.to_string())

print('\n── Best method per condition ──')
for fold_key, fold_title in FOLDS:
    print(f'\n{fold_title}')
    for ds in DS_NAMES:
        best = max(METHODS, key=lambda m: m[1][ds][fold_key])
        print(f'  {ds:<12}: {best[0]:<20} ({best[1][ds][fold_key]:.3f})')
