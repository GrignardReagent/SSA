#! /usr/bin/env python

import os
from pathlib import Path
import numpy as np
import pandas as pd
from utils.data_processing import add_binary_labels
import subprocess
import tempfile

DATA_ROOT = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data")
RESULTS_PATH = DATA_ROOT / "IY011_simulation_parameters_sobol.csv" #  this csv file stores all the simulation parameters used
df_params = pd.read_csv(RESULTS_PATH) 
# TRAJ_PATH = [DATA_ROOT / f"mRNA_trajectories_mu{row['mu_target']:.3f}_cv{row['cv_target']:.3f}_tac{row['t_ac_target']:.3f}.csv" for idx, row in df_params.iterrows()] # the trajectories 
TRAJ_PATH = [DATA_ROOT / df_params['trajectory_filename'].values[i] for i in range(len(df_params))]

# extract meta data
parameter_sets = [{
    'sigma_b': row['sigma_b'],
    'sigma_u': row['sigma_u'],
    'rho': row['rho'],
    'd': row['d'],
    'label': 0
} for idx, row in df_params.iterrows()]
time_points = np.arange(0, 3000, 1.0)
size = 1000

# labelling
label_column = 'mu_target'  # column name to base the binary labels on, mu_target for simplicity
labelled_df_params = add_binary_labels(df_params,label_column)

labels = [] 
for i in range(len(df_params)):
    # find the filename for each trajectory file
    trajectory_filename = df_params['trajectory_filename'].values[i]
    # get the corresponding label from labelled_df_params
    label_value = labelled_df_params[labelled_df_params['trajectory_filename'] == trajectory_filename]['label'].iloc[0]
    labels.append(label_value)

# convert the trajectory files (csv) into npy format
for traj_file, params, label in zip(TRAJ_PATH, parameter_sets, labels):
    df_traj = pd.read_csv(traj_file)
    df_traj = df_traj.drop(columns=['label'], errors='ignore') # drop the OG label column if exists
    trajectories = df_traj.values
    trajectory_data = {
            'trajectories': trajectories.astype(np.float32),
            'time_points': time_points.astype(np.float32),
            'size': int(size),
            'parameters': params,
            'label': label,
    }
    try:
        np.savez_compressed(
        traj_file.with_suffix('.npz'),
        trajectories=trajectories.astype(np.float32),
        time_points=time_points.astype(np.float32),
        size=size,
        parameters=params,
        labels=labels
        )
    except PermissionError:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
            tmp_path = tmp_file.name
            np.savez_compressed(tmp_path, **trajectory_data)
        # Move temp file to final location with sudo
        subprocess.run(['sudo', 'mv', tmp_path, traj_file.with_suffix('.npz')], check=True)
        subprocess.run(['sudo', 'chown', f'{os.getenv("USER")}:{os.getenv("USER")}', traj_file.with_suffix('.npz')], check=True)
