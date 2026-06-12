"""
IY029 v2: TransformerClassifier — pairwise accuracy bar chart.
Loads IY029_transformer_pairwise_v2_results.json and saves IY029_transformer_pairwise_v2_acc.png.
Style is identical to the plot-acc cell in IY029_transformer_pairwise.ipynb.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif', 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.titlesize': 14,
})

SCRIPT_DIR = Path(__file__).resolve().parent
DS_NAMES   = ['Baseline', 'Mu', 'CV', 'T_ac']

results = json.load(open(SCRIPT_DIR / 'IY029_transformer_pairwise_v2_results.json'))

palette = sns.color_palette('colorblind')
colors  = [palette[i] for i in range(len(DS_NAMES))]

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
fig.suptitle('TransformerClassifier — pairwise same/different accuracy\n'
             'full pair sequence, trained end-to-end (v2, n=14 000)',
             fontsize=13, weight='bold')
plt.tight_layout()

out = SCRIPT_DIR / 'IY029_transformer_pairwise_v2_acc.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
