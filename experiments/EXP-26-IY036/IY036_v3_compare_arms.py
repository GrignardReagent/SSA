"""
IY036 v3: compare the two augmentation arms once both jobs have finished.

Reads the per-seed CSVs written by IY036_supcon_allclass_training_eddie_v3.py
(one file per arm) and reports the composed-vs-jitter_only difference.

WHY PAIRED. Both arms run the SAME seeds against the SAME fixed split and the
same 220-cell test set -- only the augmentation differs. So seed 42's composed
run and seed 42's jitter_only run are a matched pair, and the right statistic is
the mean of the per-seed differences, not the difference of the two means. The
paired form removes the seed-to-seed component that both arms share.

HONESTY. n = 3 matched pairs. This is indicative, not conclusive: a paired t-test
on 3 pairs has almost no power, so the per-seed differences are printed in full
and the reader is expected to look at them rather than at a p-value. What it CAN
rule out is a large effect being missed, and it makes the direction visible.

Usage
-----
    python IY036_v3_compare_arms.py                     # newest CSV per arm
    python IY036_v3_compare_arms.py --composed A.csv --jitter B.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IY036_DIR = Path(__file__).resolve().parent
METRICS = [
    ("svm_accuracy", "SupCon + SVM  (balanced acc)"),
    ("knn_accuracy", "SupCon + KNN  (balanced acc)"),
    ("vs_raw_svm_diff", "  margin over Raw SVM"),
    ("epochs_run", "epochs actually run"),
]


def _newest(arm: str) -> Path:
    """Most recently modified per-seed results CSV for one arm."""
    hits = sorted(IY036_DIR.glob(f"IY036_v3_{arm}_final_results_eddie_*.csv"),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(
            f"No results CSV found for arm '{arm}'. Expected "
            f"IY036_v3_{arm}_final_results_eddie_*.csv in {IY036_DIR}. "
            f"Has that job finished?"
        )
    return hits[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--composed", type=Path, default=None)
    ap.add_argument("--jitter", type=Path, default=None)
    args = ap.parse_args()

    p_comp = args.composed or _newest("composed")
    p_jit = args.jitter or _newest("jitter_only")
    comp, jit = pd.read_csv(p_comp), pd.read_csv(p_jit)
    print(f"composed   : {p_comp.name}  ({len(comp)} seeds)")
    print(f"jitter_only: {p_jit.name}  ({len(jit)} seeds)\n")

    # Pair on seed. An arm that lost a seed to a crash must not be silently
    # compared against a different set of seeds.
    merged = comp.merge(jit, on="seed", suffixes=("_composed", "_jitter"))
    if merged.empty:
        raise SystemExit("No seeds in common between the two arms -- cannot pair.")
    dropped = (set(comp.seed) | set(jit.seed)) - set(merged.seed)
    if dropped:
        print(f"WARNING: seeds present in only one arm, excluded from pairing: "
              f"{sorted(dropped)}\n")

    # The test set must be identical, or the comparison is meaningless.
    for col in ("n_test", "raw_svm_accuracy", "catch22_svm_accuracy"):
        a, b = merged[f"{col}_composed"], merged[f"{col}_jitter"]
        if not np.allclose(a, b):
            raise SystemExit(
                f"'{col}' differs between arms ({a.iloc[0]} vs {b.iloc[0]}). The "
                f"arms are not scored on the same split -- the comparison is invalid."
            )

    n_pairs = len(merged)
    print(f"=== IY036 v3: composed vs jitter_only, paired over {n_pairs} seeds ===\n")
    print(f"{'metric':32s} {'composed':>18s} {'jitter_only':>18s} {'paired diff':>18s}")
    print("-" * 90)
    for col, label in METRICS:
        c, j = merged[f"{col}_composed"], merged[f"{col}_jitter"]
        diffs = c - j
        print(f"{label:32s} {c.mean():9.4f} ± {c.std():.4f} "
              f"{j.mean():9.4f} ± {j.std():.4f} "
              f"{diffs.mean():+9.4f} ± {diffs.std():.4f}")
        print(f"{'':32s} per-seed diffs: "
              f"{', '.join(f'seed {s}: {d:+.4f}' for s, d in zip(merged.seed, diffs))}")

    # Constant reference points -- identical in both arms by construction.
    r = merged.iloc[0]
    print("\nConstant across every run (same split, no training randomness):")
    print(f"  {'Raw SVM':24s} {r['raw_svm_accuracy_composed']:.4f}")
    print(f"  {'Raw KNN':24s} {r['raw_knn_accuracy_composed']:.4f}")
    print(f"  {'Catch22 + SVM':24s} {r['catch22_svm_accuracy_composed']:.4f}")
    print(f"  {'Chance':24s} {r['chance_composed']:.4f}")
    print(f"  {'Test set size':24s} {int(r['n_test_composed'])} cells")

    svm_diffs = merged["svm_accuracy_composed"] - merged["svm_accuracy_jitter"]
    print(f"\nREAD THIS BEFORE CONCLUDING ANYTHING: n = {n_pairs} matched pairs. The "
          f"mean paired\ndifference in SupCon+SVM balanced accuracy is "
          f"{svm_diffs.mean():+.4f} (sd {svm_diffs.std():.4f}); the per-seed values\n"
          f"above show whether that is a consistent direction or one seed carrying "
          f"the mean.\nA single-run test-set difference below ~0.08 is not resolvable "
          f"at n=220 regardless.")


if __name__ == "__main__":
    main()
