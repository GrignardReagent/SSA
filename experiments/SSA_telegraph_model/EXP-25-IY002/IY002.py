#!/usr/bin/env python3

import pandas as pd
import glob
from utils.load_data import load_and_split_data
from classifiers.lstm_classifier import lstm_classifier

##########################################################################
# 1) Combine all steady state data from IY004
##########################################################################

# path to all *steady state* CSV files, for simplicity we only take the first set of steady state data ending with 0_SS.csv
file_paths = sorted(glob.glob('/home/ianyang/stochastic_simulations/experiments/SSA_telegraph_model/var_v_accuracy_plot/data_12_04_2025/mRNA_trajectories_variance_*/steady_state_trajectories/m_traj_*_0_SS.csv')) 
# len(file_paths)

# Read and combine
dfs = [pd.read_csv(f) for f in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: shuffle the rows
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
combined_df.to_csv('data/IY002_input_A.csv', index=False)
print(f"✅ Combined {len(file_paths)} files into {combined_df.shape[0]} rows.")

# Train LSTM model using SSA data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data('data/IY002_input_A.csv', split_val_size=0.2) # we must define split_val_size here to get a validation set

# finetune model 
lstm_accuracy = lstm_classifier(X_train, X_val, X_test, y_train, y_val, y_test, epochs=50,
                                        hidden_size=64, num_layers=2, dropout_rate=0.01,
                                        learning_rate=0.001, batch_size=32,
                                        use_conv1d=True, use_attention=True, num_attention_heads=4, save_path='IY002A.pth')