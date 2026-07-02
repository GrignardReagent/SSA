"""
IY025: SVM pairwise accuracy restricted to high t_ac test conditions.

Design:
  - Train an RBF SVM (PCA-100 preprocessing for speed) on the pre-built
    IY014_static_train.pt (3000 pairs, original 2× t_ac distance threshold).
  - Sanity-check against IY014_static_test.pt to confirm we reproduce the
    original ~81% accuracy before modifying the test distribution.
  - Pre-load all .npz files once, then build test pairs where BOTH conditions
    have t_ac ≥ cutoff AND |log(t_ac_a/t_ac_b)| > 0.7 (same 2× threshold).
  - Cutoffs: [0, 10, 20, 30, 40, 50] min.
  - Expected: accuracy decreases as cutoff increases, because high t_ac
    conditions are all unimodal/bursty and the bimodal shape signal vanishes.

Run (from experiments/EXP-26-IY025/):
    micromamba run -n stochastic_sim python IY025_tac_svm_high_tac.py
"""
import sys
sys.path.insert(0, '../../src')

import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from dataloaders import load_loader_from_disk
import torch

plt.rcParams.update({
    'font.family':     'sans-serif',
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize':  14,
})
PALETTE = sns.color_palette('colorblind')

IY014_TAC_DIR   = Path('../EXP-26-IY014/data_t_ac_variation')
TAC_CUTOFFS     = [0, 10, 20, 30, 40, 50]
LOG_DIST_THRESH = 0.7   # |log(t_ac_a/t_ac_b)| > 0.7  ≡ ≥2× ratio
N_TEST          = 600   # pairs per cutoff (300 same + 300 different)
N_PCA           = 100   # PCA components; retains majority of variance
TRAJS_PER_FILE  = 50
SEP_VAL         = -100.0
SEED            = 42

t_wall = time.time()

def _load_static(path, batch_size=9999):
    """Load a static .pt dataset → (X array (N, T), y array (N,))."""
    loader = load_loader_from_disk(path, batch_size=batch_size)
    X_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            X_list.append(xb.numpy().reshape(xb.shape[0], -1))
            y_list.append(yb.numpy().ravel())
    return np.vstack(X_list), np.concatenate(y_list)

# ── Step 1: Fit PCA + RBF SVM on pre-built training data ─────────────────────
print('Loading pre-built training data...', flush=True)
X_train, y_train = _load_static(IY014_TAC_DIR / 'IY014_static_train.pt')
print(f'  {X_train.shape[0]} pairs × {X_train.shape[1]} features; '
      f'{(y_train==1).sum()} same / {(y_train==0).sum()} different', flush=True)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)

print(f'Fitting PCA ({N_PCA} components)...', flush=True)
pca = PCA(n_components=N_PCA, random_state=SEED)
X_tr_pca = pca.fit_transform(X_tr_sc)
expl_var = pca.explained_variance_ratio_.sum()
print(f'  Explained variance: {expl_var*100:.1f}%', flush=True)

print('Fitting RBF SVM on PCA-reduced data...', flush=True)
t_svm = time.time()
clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=SEED)
clf.fit(X_tr_pca, y_train)
train_acc = (clf.predict(X_tr_pca) == y_train).mean()
print(f'  Fitted in {time.time()-t_svm:.0f}s; train acc={train_acc:.4f}; '
      f'{clf.n_support_.sum()} support vectors', flush=True)

# ── Step 2: Sanity-check on static test set ───────────────────────────────────
print('\nSanity-check on IY014_static_test.pt...', flush=True)
X_test_st, y_test_st = _load_static(IY014_TAC_DIR / 'IY014_static_test.pt')
X_test_st_pca = pca.transform(scaler.transform(X_test_st))
static_acc = (clf.predict(X_test_st_pca) == y_test_st).mean()
print(f'  Static test accuracy: {static_acc:.4f}  '
      f'(original RBF SVM = 0.8133; PCA compression expected to lower slightly)', flush=True)

# ── Step 3: Pre-load ALL .npz trajectory files once ──────────────────────────
t1 = time.time()
print('\nPre-loading all t_ac trajectory files...', flush=True)
df_params = pd.read_csv(IY014_TAC_DIR / 'IY014_simulation_t_ac_parameters_sobol.csv')
ok = (df_params['success'].astype(bool) & df_params['error_message'].isna()
      & (df_params['mean_rel_error_pct'] < 20) & (df_params['cv_rel_error_pct'] < 20)
      & (df_params['t_ac_rel_error_pct'] < 20))
df_params = df_params[ok].reset_index(drop=True)

traj_cache = {}  # float(t_ac_observed) → np.ndarray (N_trajs, T)
for _, row in df_params.iterrows():
    npz = str(IY014_TAC_DIR / row['trajectory_filename'].replace('.csv', '.npz'))
    data = np.load(npz, allow_pickle=True)
    traj_cache[float(row['t_ac_observed'])] = data['trajectories'][:TRAJS_PER_FILE].astype(np.float32)

T = next(iter(traj_cache.values())).shape[1]
all_tac = np.array(sorted(traj_cache.keys()))
print(f'  {len(traj_cache)} conditions in {time.time()-t1:.0f}s; '
      f'T={T}; t_ac ∈ [{all_tac.min():.1f}, {all_tac.max():.1f}] min', flush=True)

# ── Step 4: Build test pairs for a given t_ac subset ─────────────────────────
def build_test_pairs(tac_subset, cache, n_pairs, seed, log_thresh):
    """
    Build test pairs from conditions in tac_subset with |log(a/b)| > log_thresh for
    'different' pairs. Returns (X (N, T_total), y (N,)).
    """
    rng     = np.random.default_rng(seed)
    tac_arr = np.array(tac_subset)
    log_tac = np.log(tac_arr)
    sep     = np.full((1, 1), SEP_VAL, dtype=np.float32)

    # Enumerate all valid (i<j) 'different' condition-index pairs
    valid_diff = [(i, j)
                  for i in range(len(tac_arr))
                  for j in range(i + 1, len(tac_arr))
                  if abs(log_tac[i] - log_tac[j]) > log_thresh]

    n_same = n_pairs // 2
    n_diff = n_pairs - n_same
    if len(valid_diff) == 0:
        print('    [warn] no valid "different" pairs — skipped', flush=True)
        return np.empty((0, 2*T+1)), np.empty(0)
    if len(valid_diff) < n_diff:
        print(f'    [warn] only {len(valid_diff)} valid "diff" pairs; using all', flush=True)
        n_diff = len(valid_diff)
        n_same = n_diff

    def _pair_feat(t1, t2):
        pair = np.concatenate([t1.reshape(-1, 1), sep, t2.reshape(-1, 1)], axis=0)
        m, s = pair.mean(), pair.std() + 1e-8
        return ((pair - m) / s).squeeze()

    X_list, y_list = [], []
    # Same pairs (two trajectories from the same condition)
    for ci in rng.integers(0, len(tac_arr), size=n_same):
        trajs = cache[tac_arr[ci]]
        i, j  = rng.choice(len(trajs), size=2, replace=False)
        X_list.append(_pair_feat(trajs[i], trajs[j])); y_list.append(1.0)

    # Different pairs (sampled from valid condition pairs)
    chosen = rng.choice(len(valid_diff), size=n_diff, replace=(len(valid_diff) < n_diff))
    for idx in chosen:
        ci_a, ci_b = valid_diff[int(idx)]
        ta = cache[tac_arr[ci_a]]; tb = cache[tac_arr[ci_b]]
        X_list.append(_pair_feat(ta[rng.integers(0, len(ta))],
                                 tb[rng.integers(0, len(tb))]))
        y_list.append(0.0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# ── Step 5: Evaluate at each t_ac cutoff ─────────────────────────────────────
print('\nEvaluating at each t_ac cutoff (2× distance threshold)...', flush=True)
results = []
for cutoff in TAC_CUTOFFS:
    filtered = all_tac[all_tac >= cutoff].tolist()
    X_test, y_test = build_test_pairs(filtered, traj_cache, N_TEST, seed=SEED + 2,
                                      log_thresh=LOG_DIST_THRESH)
    if len(X_test) == 0:
        continue
    # count valid 'different' condition pairs
    tac_sub  = np.array(filtered)
    log_sub  = np.log(tac_sub)
    n_cpairs = sum(1 for i in range(len(tac_sub))
                   for j in range(i+1, len(tac_sub))
                   if abs(log_sub[i]-log_sub[j]) > LOG_DIST_THRESH)
    X_pca    = pca.transform(scaler.transform(X_test))
    acc      = (clf.predict(X_pca) == y_test).mean()
    se       = np.sqrt(acc * (1 - acc) / len(y_test))
    results.append({'cutoff': cutoff, 'n_cond': len(filtered),
                    'n_valid_cond_pairs': n_cpairs, 'n_test_pairs': len(y_test),
                    'accuracy': acc, 'se': se})
    print(f'  Cutoff {cutoff:>3} min  ({len(filtered):>4} conds, '
          f'{n_cpairs:>7} cond-pairs):  acc={acc:.4f} ± {1.96*se:.4f}', flush=True)

results_df = pd.DataFrame(results)
print('\nSummary:')
print(results_df[['cutoff', 'n_cond', 'n_valid_cond_pairs', 'accuracy', 'se']].to_string(index=False))

# ── Step 6: Permutation test ──────────────────────────────────────────────────
print('\nRunning permutation tests (N_PERM=300)...', flush=True)
N_PERM = 300
perm_results = []
for row in results_df.itertuples():
    filtered = all_tac[all_tac >= row.cutoff].tolist()
    X_test, y_test = build_test_pairs(filtered, traj_cache, row.n_test_pairs, seed=SEED + 2,
                                      log_thresh=LOG_DIST_THRESH)
    X_pca   = pca.transform(scaler.transform(X_test))
    obs     = (clf.predict(X_pca) == y_test).mean()
    rng_p   = np.random.default_rng(SEED)
    null    = np.array([(clf.predict(X_pca) == rng_p.permutation(y_test)).mean()
                        for _ in range(N_PERM)])
    p_val   = max(float((null >= obs).mean()), 1.0 / N_PERM)
    perm_results.append({'cutoff': row.cutoff, 'obs': obs,
                         'null_mean': float(null.mean()), 'null_std': float(null.std()),
                         'p_val': p_val})
    sig = ' *' if p_val < 0.05 else '  '
    print(f'  cutoff {row.cutoff:>3} min: obs={obs:.4f}  '
          f'null={null.mean():.4f}±{null.std():.4f}  p={p_val:.4f}{sig}', flush=True)
perm_df = pd.DataFrame(perm_results)

# ── Step 7: Plot ──────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
cutoffs = results_df['cutoff'].values
accs    = results_df['accuracy'].values
ses     = results_df['se'].values
n_conds = results_df['n_cond'].values

ax1.plot(cutoffs, accs, 'o-', color=PALETTE[0], linewidth=2, markersize=8,
         label='RBF SVM accuracy (PCA-100)')
ax1.fill_between(cutoffs, accs - 1.96 * ses, accs + 1.96 * ses,
                 color=PALETTE[0], alpha=0.2, label='95% CI')
ax1.axhline(0.5,        color='dimgrey',  linestyle='--', linewidth=1.2, label='Chance (0.50)')
ax1.axhline(static_acc, color=PALETTE[3], linestyle=':',  linewidth=1.5,
            label=f'Static test baseline ({static_acc:.2f})')
for c, a in zip(cutoffs, accs):
    ax1.annotate(f'{a:.2f}', (c, a), textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=10)
ax1.set_xlabel('t_ac cutoff / min\n'
               '(test: BOTH conditions ≥ cutoff AND |log(t_ac_a/t_ac_b)| > 0.7)', fontsize=12)
ax1.set_ylabel('SVM test accuracy', fontsize=12)
ax1.set_ylim(0.4, 1.05)
ax1.set_xticks(cutoffs)
ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
ax2 = ax1.twinx()
ax2.bar(cutoffs, n_conds, width=4, color=PALETTE[1], alpha=0.2, label='N conditions')
ax2.set_ylabel('Number of conditions', fontsize=11, color=PALETTE[1])
ax2.tick_params(axis='y', labelcolor=PALETTE[1])
ax2.legend(bbox_to_anchor=(1.01, 0.65), loc='upper left', fontsize=10)
ax1.set_title('SVM pairwise accuracy — restricting test to high t_ac conditions\n'
              'IY014 t_ac variation (μ=1000, CV=0.5); '
              'PCA-100 + RBF SVM trained on full range',
              fontsize=11, weight='bold')
plt.tight_layout()
plt.savefig('figures/IY025_svm_accuracy_vs_tac_cutoff.png', dpi=150, bbox_inches='tight')
print('\nSaved figures/IY025_svm_accuracy_vs_tac_cutoff.png', flush=True)

from scipy.stats import norm as scipy_norm
fig2, axes = plt.subplots(1, len(results_df), figsize=(5 * len(results_df), 4),
                          constrained_layout=True)
if len(results_df) == 1:
    axes = [axes]
for ax, (row, p_row) in zip(axes, zip(results_df.itertuples(), perm_df.itertuples())):
    x = np.linspace(p_row.null_mean - 4 * p_row.null_std,
                    p_row.null_mean + 4 * p_row.null_std, 200)
    ax.plot(x, scipy_norm.pdf(x, p_row.null_mean, p_row.null_std),
            color=PALETTE[0], linewidth=1.5, label='Null (normal approx.)')
    ax.axvline(p_row.obs, color='black', linewidth=2, label=f'obs={p_row.obs:.2f}')
    ax.axvline(0.5, color='dimgrey', linestyle='--', linewidth=1)
    p_str = f'p={p_row.p_val:.3f}' if p_row.p_val >= 0.001 else 'p<0.001'
    ax.text(0.97, 0.97, p_str, transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_title(f't_ac ≥ {row.cutoff} min\n({row.n_cond} conds)', fontsize=12)
    ax.set_xlabel('Test accuracy', fontsize=12)
    if ax is axes[0]:
        ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
fig2.suptitle(f'Permutation test: RBF SVM (PCA-100) accuracy on restricted test subsets\n'
              f'(trained on full range; N_PERM={N_PERM}; test uses 2× distance threshold)',
              fontsize=12, weight='bold')
plt.savefig('figures/IY025_svm_tac_cutoff_permutation.png', dpi=150, bbox_inches='tight')
print('Saved figures/IY025_svm_tac_cutoff_permutation.png', flush=True)

out = {
    'design': ('PCA-100 + RBF SVM trained on pre-built full-range data; '
               'test pairs: BOTH conditions t_ac>=cutoff AND |log(a/b)|>0.7'),
    'n_pca': N_PCA, 'explained_variance': float(expl_var),
    'static_test_acc': float(static_acc),
    'svm': results_df.to_dict(orient='records'),
    'perm': perm_df.to_dict(orient='records'),
}
with open('IY025_tac_svm_high_tac_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved IY025_tac_svm_high_tac_results.json', flush=True)
print(f'\nAll done in {time.time()-t_wall:.0f}s total.', flush=True)
