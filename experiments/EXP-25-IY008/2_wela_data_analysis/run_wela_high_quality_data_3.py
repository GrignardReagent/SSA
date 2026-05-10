import os
import sys
import tempfile
import traceback
from pathlib import Path

# Prevent user site-packages (~/.local/lib) from shadowing the env's packages.
# Must re-exec because PYTHONNOUSERSITE is read at interpreter startup.
if not os.environ.get("PYTHONNOUSERSITE"):
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import h5py
import matplotlib.pylab as plt
import seaborn as sns
from agora.io.signal import Signal
from wela.dataloader import DataLoader
from wela.plotting import kymograph
from wela.sorting import sort_by_budding


h5dir = Path("/home/ianyang/alibylite/high_quality_data_analysis/")

# OMIDs that failed in run_wela_high_quality_data_2.py.
omids = [
    4052,
    4051,
    4054,
    4102,
    4103,
    4106,
    4110,
    3903,
    2853,
    2852,
    2849,
    2801,
]

key_index = "mother_median_GFP"


def find_experiment(omid):
    return next(
        (path for path in h5dir.iterdir() if path.is_dir() and str(omid) in path.name),
        None,
    )


def find_bad_h5files(expt_dir):
    """Find H5 files that are known to break DataLoader for these runs."""
    bad = {}
    for h5file in sorted(expt_dir.glob("*.h5")):
        reasons = []
        try:
            with h5py.File(h5file, "r") as f:
                if "postprocessing" not in f:
                    reasons.append("missing postprocessing")
        except Exception as e:
            reasons.append(f"cannot open h5: {type(e).__name__}: {e}")

        try:
            lineage = Signal(h5file).lineage()
            if getattr(lineage, "ndim", None) == 1 and getattr(lineage, "size", None) == 0:
                reasons.append("empty lineage")
        except Exception as e:
            reasons.append(f"lineage error: {type(e).__name__}: {e}")

        if reasons:
            bad[h5file.name] = reasons
    return bad


def make_filtered_experiment(expt_dir, filtered_root, bad_h5files):
    """Create a temporary symlinked experiment folder excluding bad H5 files."""
    filtered_expt = filtered_root / expt_dir.name
    filtered_expt.mkdir(parents=True, exist_ok=True)
    for h5file in sorted(expt_dir.glob("*.h5")):
        if h5file.name in bad_h5files:
            continue
        target = filtered_expt / h5file.name
        target.symlink_to(h5file)
    return filtered_expt


with tempfile.TemporaryDirectory(prefix="wela_filtered_h5_") as tmpdir:
    filtered_root = Path(tmpdir)

    for omid in omids:
        expt_dir = find_experiment(omid)
        if expt_dir is None:
            print(f"FAILURE: No experiment found for omid {omid}, skipping")
            continue

        bad_h5files = find_bad_h5files(expt_dir)
        print(f"\n---\nomid {omid}: {expt_dir.name}\n---")
        if bad_h5files:
            print("Skipping bad H5 files:")
            for name, reasons in bad_h5files.items():
                print(f"  {name}: {', '.join(reasons)}")
        else:
            print("No bad H5 files detected.")

        filtered_expt = make_filtered_experiment(expt_dir, filtered_root, bad_h5files)
        kept_h5files = list(filtered_expt.glob("*.h5"))
        if not kept_h5files:
            print(f"FAILURE: omid {omid} has no usable H5 files after filtering")
            continue

        dl = DataLoader(str(filtered_root), ".")
        try:
            dl.load(filtered_expt.name, key_index=key_index, cutoff=0.9)
        except Exception as e:
            print(
                f"FAILURE: omid {omid}, key_index {key_index!r} failed after "
                f"filtering: {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue

        dl.save()

        groups = dl.df.group.unique()
        for group in groups:
            _, buddings = dl.get_time_series("buddings", group=group)
            sort_order = sort_by_budding(buddings)
            fig_kymograph, ax_kymograph = kymograph(
                dl.df[dl.df.group == group],
                hue=key_index,
                title=group,
                sort_order=sort_order,
                returnfig=True,
            )
            fig_kymograph.savefig(f"./{omid}_{group}_kymograph.png")
            plt.close(fig_kymograph)

        mean_plot = sns.relplot(data=dl.df, x="time", y=key_index, kind="line", hue="group")
        mean_plot.savefig(f"./{omid}_means_plot.png")
        plt.close(mean_plot.fig)
