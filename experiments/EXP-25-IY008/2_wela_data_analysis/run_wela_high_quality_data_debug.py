import sys
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from wela.dataloader import dataloader
from wela.imageviewer import ImageViewer, get_h5files
from wela.plotting import kymograph
from wela.sorting import sort_by_budding

h5dir = "/home/ianyang/alibylite/high_quality_data_analysis/"
# exp_list = pd.read_csv("exp_list.csv")
# omids = exp_list["exp_name"].tolist()
omids =['18572_2020_02_04_steady_2p0glc_2min_3z_mch5V_msn4msn2_00',
'19477_2020_11_27_steadystate_glucose_898_exposure_901_2w0p01_00',
'18367_2020_01_07_steady_0_1_gluc_2min_msn4msn2_00']

# FILL IN
server_info = {
    "host": "staffa.bio.ed.ac.uk",
    "username": "upload",
    "password": "gothamc1ty",
}
view = False
# pick the experiment to analyse
# omid = "19330_2020_11_02_steadystate_glucose_1345m_2w2_00"

# 1. Run with view=True to check visually that aliby has worked correctly.
# 2. Set key_index, the signal you are most interested in.
# 3. Run with view=False to run dataloader and save a tsv file.
for omid in omids:
    if view:
        omero_name = [om for om in omids if str(omid) in om][0]
        h5files = get_h5files(h5dir, omero_name)
        position = h5files[0]
        h5file = f"{h5dir}{omero_name}/{position}"
        iv = ImageViewer.remote(h5file, server_info, omid)
        tpt_end = 10
        no_cells = 6
        iv.view(
            trap_ids=iv.sample_traps_with_cells(
                tpt_end=tpt_end, no_cells=no_cells
            ),
            tpt_end=tpt_end,
            channels_to_skip=["cy5"],
            no_rows=2,
        )
        sys.exit(0)
    else:
        # run dataloader
        key_index = "median_GFP" # sometimes this is "median_GFP_Z" instead of "median_GFP"
        dl = dataloader(h5dir, ".")
        dl.define_g2a_dict(fl_channels=["GFP"]) # sometimes this is ["GFP"] instead of ["GFP_Z"], and we need to set fl_channels=["GFP_Z"]
        expt = [omid_full for omid_full in omids if str(omid) in omid_full][0]
        try:
            dl.load(expt, key_index=key_index, cutoff=0.9)

        # ! sometimes the key_index is not "median_GFP" but "median_GFP_Z", and so this is to avoid the key error
        except:
            print(f'WARNING: key_index {key_index} not found, trying "median_GFP_Z"')
            key_index = "median_GFP_Z"
            dl = dataloader(h5dir, ".")
            dl.define_g2a_dict(fl_channels=["GFP_Z"]) # sometimes this is ["GFP"] instead of ["GFP_Z"], and we need to set fl_channels=["GFP_Z"]
            expt = [omid_full for omid_full in omids if str(omid) in omid_full][0]
            try:
                dl.load(expt, key_index=key_index, cutoff=0.9)

            # third option if GFP_Z still doesn't work    
            except:
                print(f'WARNING: key_index {key_index} not found, trying "buddings"')
                key_index = "buddings"
                dl = dataloader(h5dir, ".")
                dl.define_g2a_dict(fl_channels=["buddings"]) # sometimes this is ["GFP"] instead of ["GFP_Z"], and we need to set fl_channels=["GFP_Z"]
                expt = [omid_full for omid_full in omids if str(omid) in omid_full][0]
                try:
                    dl.load(expt, key_index=key_index, cutoff=0.9)
                except:
                    print(f'FAILURE: key_index {key_index} not found, skipping {omid}')
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