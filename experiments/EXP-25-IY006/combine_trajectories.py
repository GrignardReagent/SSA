#!/usr/bin/env python3
import os
import pandas as pd
import glob

# Define the directory path
data_dir = '/home/ianyang/stochastic_simulations/experiments/EXP-25-IY006/data'

# Get all m_traj_cv_0.32_0.32_*_SS.csv files
traj_files = sorted(glob.glob(os.path.join(data_dir, 'm_traj_cv_0.32_0.32_*_SS.csv')))

print(f"Found {len(traj_files)} trajectory files to combine.")

# Read and combine all trajectory files
combined_df = pd.DataFrame()

for file_path in traj_files:
    df = pd.read_csv(file_path)
    print(f"Reading file: {os.path.basename(file_path)} with {len(df)} rows")
    
    # If this is the first file, initialize the combined_df with it
    if combined_df.empty:
        combined_df = df
    else:
        # Otherwise, append the rows from this file
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined data to the new file
output_file = os.path.join(data_dir, 'combined_traj_cv_0.32_0.32_SS.csv')
combined_df.to_csv(output_file, index=False)

print(f"Combined trajectory data saved to: {output_file}")
print(f"Total number of rows in combined file: {len(combined_df)}")
