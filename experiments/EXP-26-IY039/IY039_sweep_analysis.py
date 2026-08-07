"""
IY039 post-sweep analysis: reload the shared Optuna study (written to by the
4 parallel Eddie workers launched via IY039_submit_sweep.sh), export the full
trial history, and rank hyperparameter/architecture importance.

The study's JournalStorage file is the single source of truth for trial
results -- there is no per-trial manual CSV to reconcile across the 4
concurrent workers (that would risk interleaved/corrupted writes); this
script queries the shared study directly, once, after all workers finish.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import optuna
import pandas as pd
import seaborn as sns

IY039_DIR = Path(__file__).resolve().parent


def main(sweep_timestamp: str) -> None:
    storage_path = IY039_DIR / f"IY039_optuna_study_{sweep_timestamp}.journal"
    study_name = f"IY039_sweep_{sweep_timestamp}"
    if not storage_path.exists():
        raise FileNotFoundError(f"No Optuna storage found at {storage_path}")

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(storage_path))
    )
    study = optuna.load_study(study_name=study_name, storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Study '{study_name}': {len(study.trials)} total trials, {len(completed)} completed.")
    if study.best_trial is not None:
        print(f"Best trial: #{study.best_trial.number}, value={study.best_trial.value:.4f}")
        print("Best params:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

    # ── Full trial history ───────────────────────────────────────────────────
    results_df = study.trials_dataframe()
    results_csv = IY039_DIR / f"IY039_sweep_results_{sweep_timestamp}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved trial history -> {results_csv}")

    # ── Parameter importance ─────────────────────────────────────────────────
    importances = optuna.importance.get_param_importances(study)
    importance_df = pd.DataFrame(
        {"parameter": list(importances.keys()), "importance": list(importances.values())}
    ).sort_values("importance", ascending=True)

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("colorblind", n_colors=len(importance_df))
    bars = ax.barh(importance_df["parameter"], importance_df["importance"], color=colors)
    max_importance = importance_df["importance"].max()
    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(bar.get_width() + 0.01 * max_importance, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2g}", va="center", fontsize=8)
    ax.set_xlabel("Hyperparameter importance (fraction of variance explained)", fontsize=12)
    ax.set_ylabel("Hyperparameter", fontsize=12)
    ax.set_title(f"IY039 parameter importance (n={len(completed)} completed trials)", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    fig.tight_layout()

    fig_path = IY039_DIR / "IY039_param_importance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved importance plot -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-timestamp", required=True,
                         help="The SWEEP_TIMESTAMP printed by IY039_submit_sweep.sh.")
    args = parser.parse_args()
    main(args.sweep_timestamp)
