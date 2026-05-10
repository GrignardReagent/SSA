from wela.calculate_stats import load_tsv
from wela.get_exp_conditions import get_exp_summary, get_switch_time_from_cy5
import pandas as pd
import glob
import os
import sys
sys.path.insert(0, "/home/ianyang/wela/src") # this script should be run in the wela environment, so we can import from src

### THIS SCRIPT SHOULD BE RUN IN THE WELA ENVIRONMENT ### 

# Prevent user site-packages (~/.local/lib) from shadowing the env's packages.
# Must re-exec because PYTHONNOUSERSITE is read at interpreter startup.
if not os.environ.get("PYTHONNOUSERSITE"):
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

DATA_DIR = "/home/ianyang/stochastic_simulations/experiments/EXP-25-IY008/2_wela_data_analysis"
omids = [4053, 4052, 4051, 4054, 4102, 4103, 4105, 4104, 4106, 4107, 4108, 4109, 4110, 3903, 3902, 4251, 2858, 2854, 2853, 2852, 2841, 2842, 2843, 2844, 2849, 2801]

for i in range(len(omids)):
    omid_dir = f"{DATA_DIR}/{omids[i]}"

    # load data
    tsv_files = glob.glob(f"{omid_dir}/{omids[i]}*.tsv")
    if not tsv_files:
        print(f"TSV file not found for OMID {omids[i]}. Skipping...")
        continue
    df = load_tsv(tsv_files[0])
    if df is None:
        print(f"Failed to load TSV file for OMID {omids[i]}. Skipping...")
        continue

    # Compute CV columns if missing (new datasets store raw mean/std instead)
    for ch in ['GFP', 'mCherry']:
        cv_col = f'CV_{ch}'
        if cv_col not in df.columns:
            mean_col = f'mother_mean_{ch}'
            std_col = f'mother_std_{ch}'
            if mean_col in df.columns and std_col in df.columns:
                df[cv_col] = df[std_col] / df[mean_col]

    print(f"\n=== ANALYZING EXPERIMENT {omids[i]} ===")
    # save the dataframe
    df.to_csv(f"{omid_dir}/{omids[i]}.tsv", sep='\t', index=False)
    print(f"Saved CV-computed data for OMID {omids[i]}")