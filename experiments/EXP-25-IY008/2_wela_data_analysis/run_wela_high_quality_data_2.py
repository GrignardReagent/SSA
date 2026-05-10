import os
import sys
import traceback

# Prevent user site-packages (~/.local/lib) from shadowing the env's packages.
# Must re-exec because PYTHONNOUSERSITE is read at interpreter startup.
if not os.environ.get("PYTHONNOUSERSITE"):
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path
import h5py
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from agora.io.signal import Signal
from wela.dataloader import DataLoader
from wela.imageviewer import ImageViewer
from wela.utils import get_h5files
from wela.plotting import kymograph
from wela.sorting import sort_by_budding

h5dir = "/home/ianyang/alibylite/high_quality_data_analysis/"
omids = [4053, 4052, 4051, 4054, 4102, 4103, 4105, 4104, 4106, 4107, 4108, 4109, 4110, 3903, 3902, 4251, 2858, 2854, 2853, 2852, 2841, 2842, 2843, 2844, 2849, 2801]

# FILL IN
server_info = {
    "host": "staffa.bio.ed.ac.uk",
    "username": "upload",
    "password": "gothamc1ty",
}
view = False
# pick the experiment to analyse
# omid = "19330_2020_11_02_steadystate_glucose_1345m_2w2_00"


def find_h5files_missing_postprocessing(expt_dir):
    """Return H5 files that do not contain the postprocessing group."""
    missing = []
    for h5file in sorted(expt_dir.glob("*.h5")):
        if not h5py.is_hdf5(h5file):
            continue
        with h5py.File(h5file, "r") as f:
            if "postprocessing" not in f:
                missing.append(h5file.name)
    return missing


def find_h5files_with_bad_lineage(expt_dir):
    """Return H5 files with empty or unreadable lineage data."""
    bad = []
    for h5file in sorted(expt_dir.glob("*.h5")):
        try:
            lineage = Signal(h5file).lineage()
        except Exception as e:
            bad.append((h5file.name, f"{type(e).__name__}: {e}"))
            continue
        if (
            getattr(lineage, "ndim", None) == 1
            and getattr(lineage, "size", None) == 0
        ):
            bad.append((h5file.name, "empty lineage"))
    return bad


# 1. Run with view=True to check visually that aliby has worked correctly.
# 2. Set key_index, the signal you are most interested in.
# 3. Run with view=False to run dataloader and save a tsv file.
for omid in omids:
    if view:
        omero_name = next(
            (d.name for d in Path(h5dir).iterdir() if d.is_dir() and str(omid) in d.name),
            None,
        )
        if omero_name is None:
            print(f"FAILURE: No experiment directory found for omid {omid}, skipping")
            continue
        h5files = get_h5files(h5dir, omero_name, noh5_suffix=True)
        position = h5files[0]
        iv = ImageViewer.remote(position, f"{h5dir}{omero_name}", omid, server_info)
        tpt_end = 10
        no_cells = 6
        iv.view(
            traps=iv.sample_traps_with_cells(
                tp_end=tpt_end, no_cells=no_cells
            ),
            tp_end=tpt_end,
            channels_to_skip=["cy5"],
            no_rows=2,
        )
        sys.exit(0)
    else:
        # run dataloader
        dl = DataLoader(h5dir, ".")
        expt = next(
            (name for name in dl.experiments.values() if str(omid) in name),
            None,
        )
        if expt is None:
            print(f"FAILURE: No experiment found for omid {omid}, skipping")
            continue

        key_index = "mother_median_GFP"
        try:
            dl.load(expt, key_index=key_index, cutoff=0.9, skip_bad_h5=False)
        except KeyError as e:
            # catch the postprocessing missing error and print a specific message
            if "postprocessing is missing" in str(e):
                missing = find_h5files_missing_postprocessing(
                    Path(h5dir) / expt
                )
                print(
                    f"FAILURE: omid {omid} has H5 file(s) missing "
                    "postprocessing:"
                )
                for h5file in missing:
                    print(f"  {h5file}")
                if not missing:
                    print("  Could not identify the exact H5 file.")
                print(f"Skipping omid {omid}")
                continue
            print(
                f"FAILURE: omid {omid}, key_index {key_index!r} failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue
        except IndexError as e:
            if "too many indices for array" in str(e):
                bad_lineage = find_h5files_with_bad_lineage(
                    Path(h5dir) / expt
                )
                print(
                    f"FAILURE: omid {omid} has malformed or empty lineage "
                    "in H5 file(s):"
                )
                for h5file, reason in bad_lineage:
                    print(f"  {h5file}: {reason}")
                if not bad_lineage:
                    print("  Could not identify the exact H5 file.")
                print(f"Skipping omid {omid}")
                continue
            print(
                f"FAILURE: omid {omid}, key_index {key_index!r} failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue
        except Exception as e:
            print(
                f"FAILURE: omid {omid}, key_index {key_index!r} failed: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue

        dl.save()

        # plot kymographs
        groups = dl.df.group.unique()
        for group in groups:
            _, buddings = dl.get_time_series("buddings", group=group)
            sort_order = sort_by_budding(buddings)
            fig_kymograph, ax_kymograph = kymograph(
                dl.df[dl.df.group == group],
                hue=key_index,
                title=group,
                sort_order=sort_order,
                returnfig=True # this is to return the figure and axis
            )
            # save the kymograph(s)
            fig_kymograph.savefig(f"./{omid}_{group}_kymograph.png")

        # plot means
        sns.relplot(data=dl.df, x="time", y=key_index, kind="line", hue="group")
        plt.savefig(f"./{omid}_means_plot.png")
        # plt.show()
