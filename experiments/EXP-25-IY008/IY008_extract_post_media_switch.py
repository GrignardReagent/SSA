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

exp_list = pd.read_csv(f"{DATA_DIR}/exp_list.csv")
omids = exp_list["exp_name"].tolist()

for i in range(len(omids)):
    omid_num = omids[i].split('_')[0]
    omid_dir = f"{DATA_DIR}/{omid_num}"

    # the acq and log files to get the exp descriptions
    acq_files = glob.glob(f"/home/ianyang/alibylite/high_quality_data_analysis/{omids[i]}/*Acq.txt")
    if not acq_files:
        print(f"No acquisition file found for OMID {omids[i]}. Skipping...")
        continue
    acq_file_path = acq_files[0]

    log_files = glob.glob(f"/home/ianyang/alibylite/high_quality_data_analysis/{omids[i]}/*log.txt")
    if not log_files:
        print(f"No log file found for OMID {omids[i]}. Skipping...")
        continue
    log_file_path = log_files[0]

    # load data
    tsv_files = glob.glob(f"{omid_dir}/{omids[i]}*.tsv")
    # Skip files that already have "_post_media_switch" in the name
    tsv_files = [f for f in tsv_files if "_post_media_switch" not in f]
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
    # New-format experiments have no real Acq.txt (only staffa/omero upload records)
    is_new_format = not acq_file_path.endswith("Acq.txt")
    if not is_new_format:
        summary_df, tf_df, _ = get_exp_summary(acq_file_path, log_file_path)
    else:
        summary_df = None

    # Determine switch time in hours
    switch_time_hours = None

    if summary_df is not None:
        switch_times = summary_df['switch_times'].iloc[0]
        if switch_times not in ('N/A', None):
            if isinstance(switch_times, list):
                switch_times = switch_times[0]
            switch_time_hours = switch_times / 60  # minutes → hours

    # Fall back to cy5-based detection for new-format experiments
    if switch_time_hours is None:
        switch_time_hours = get_switch_time_from_cy5(df)
        if switch_time_hours is not None:
            print(f"Switch time detected from cy5 at {switch_time_hours:.3f} h")

    if switch_time_hours is None:
        print(f"Could not determine switch time for OMID {omids[i]}. Skipping...")
        continue

    # extract the time series after the media switch
    df = df[df["time"] >= switch_time_hours]

    # save the filtered dataframe
    df.to_csv(f"{omid_dir}/{omids[i]}_post_media_switch.tsv", sep='\t', index=False)
    print(f"Saved post-media-switch data for OMID {omids[i]} (switch at {switch_time_hours:.3f} h)")