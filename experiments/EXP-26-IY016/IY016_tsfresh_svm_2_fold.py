import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataloaders import load_loader_from_disk

# ==========================================
# CONFIGURATION
# ==========================================
# 2-fold change experiments (from IY011)
EXPERIMENTS = [
    {
        "name": "Baseline",
        "data_dir": "../EXP-25-IY011/data",
        "csv_name": "IY011_simulation_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "black"
    },
    {
        "name": "Mu-Variation",
        "data_dir": "../EXP-25-IY011/data_mu_variation",
        "csv_name": "IY011_simulation_mu_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "blue"
    },
    {
        "name": "CV-Variation",
        "data_dir": "../EXP-25-IY011/data_cv_variation",
        "csv_name": "IY011_simulation_cv_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "green"
    },
    {
        "name": "T_ac-Variation",
        "data_dir": "../EXP-25-IY011/data_t_ac_variation",
        "csv_name": "IY011_simulation_t_ac_parameters_sobol.csv",
        "train_save_path":"IY011_static_train.pt",
        "val_save_path":"IY011_static_val.pt",
        "test_save_path":"IY011_static_test.pt",
        "color": "red"
    }
]

# ==========================================
BASELINE_DATA_ROOT = Path(EXPERIMENTS[0]["data_dir"])
MU_DATA_ROOT = Path(EXPERIMENTS[1]["data_dir"])
CV_DATA_ROOT = Path(EXPERIMENTS[2]["data_dir"]) 
TAC_DATA_ROOT = Path(EXPERIMENTS[3]["data_dir"])

baseline_train_save_path = BASELINE_DATA_ROOT / EXPERIMENTS[0]["train_save_path"]
baseline_val_save_path   = BASELINE_DATA_ROOT / EXPERIMENTS[0]["val_save_path"]
baseline_test_save_path  = BASELINE_DATA_ROOT / EXPERIMENTS[0]["test_save_path"]
baseline_train_loader = load_loader_from_disk(baseline_train_save_path)
baseline_val_loader = load_loader_from_disk(baseline_val_save_path)
baseline_test_loader = load_loader_from_disk(baseline_test_save_path)

cv_train_save_path = CV_DATA_ROOT / EXPERIMENTS[2]["train_save_path"]
cv_val_save_path   = CV_DATA_ROOT / EXPERIMENTS[2]["val_save_path"]
cv_test_save_path  = CV_DATA_ROOT / EXPERIMENTS[2]["test_save_path"]
cv_train_loader = load_loader_from_disk(cv_train_save_path)
cv_val_loader = load_loader_from_disk(cv_val_save_path)
cv_test_loader = load_loader_from_disk(cv_test_save_path)   

tac_train_save_path = TAC_DATA_ROOT / EXPERIMENTS[3]["train_save_path"]
tac_val_save_path   = TAC_DATA_ROOT / EXPERIMENTS[3]["val_save_path"]
tac_test_save_path  = TAC_DATA_ROOT / EXPERIMENTS[3]["test_save_path"]
tac_train_loader = load_loader_from_disk(tac_train_save_path)
tac_val_loader = load_loader_from_disk(tac_val_save_path)
tac_test_loader = load_loader_from_disk(tac_test_save_path)

mu_train_save_path = MU_DATA_ROOT / EXPERIMENTS[1]["train_save_path"]
mu_val_save_path   = MU_DATA_ROOT / EXPERIMENTS[1]["val_save_path"]
mu_test_save_path  = MU_DATA_ROOT / EXPERIMENTS[1]["test_save_path"]
mu_train_loader = load_loader_from_disk(mu_train_save_path)
mu_val_loader = load_loader_from_disk(mu_val_save_path)
mu_test_loader = load_loader_from_disk(mu_test_save_path)    

# verify the results seen from training script
def extract_data_for_svm(loader):
    """
    Extracts all batches from a DataLoader and flattens them for SVM input.
    Input X: (Batch, Time, Features) -> Output X: (Total_Samples, Time * Features)
    """
    X_list = []
    y_list = []
    
    print(f"Extracting data from loader for SVM...")
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            # Move to CPU and convert to numpy
            X_np = X_batch.cpu().numpy()
            y_np = y_batch.cpu().numpy()
            
            # Flatten the time series: 
            # (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
            # This turns the time series into a long feature vector
            X_flat = X_np.reshape(X_np.shape[0], -1)
            
            X_list.append(X_flat)
            y_list.append(y_np)
            
    # Concatenate all batches
    return np.vstack(X_list), np.concatenate(y_list)
import os
# Completely hide the GPU from Python
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["NUMBA_DISABLE_CUDA"] = "1"

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import pandas as pd
# --- 1. Helper: Convert Tensor to TSFresh DataFrame ---
def loader_to_tsfresh_df(loader, max_samples=500):
    """
    Converts PyTorch DataLoader -> Pandas DataFrame in 'Long' format.
    tsfresh expects columns: [id, time, value]
    
    Args:
        max_samples: Limit samples to speed up extraction (TSFresh is slow!)
    """
    X_list, y_list = [], []
    count = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.numpy().squeeze() # (B, T)
        y_batch = y_batch.numpy().flatten()
        
        # Iterate samples in batch
        for i in range(X_batch.shape[0]):
            if count >= max_samples: break
            
            # Create 'Long' format for this sample
            # ID needs to be unique for the whole dataset
            sample_id = count 
            time_steps = np.arange(X_batch.shape[1])
            values = X_batch[i, :]
            
            # Append to lists (much faster than appending to DF)
            X_list.append(pd.DataFrame({
                'id': sample_id,
                'time': time_steps,
                'value': values
            }))
            y_list.append(y_batch[i])
            
            count += 1
        if count >= max_samples: break
            
    # Combine
    if not X_list: return None, None
    
    df_ts = pd.concat(X_list, ignore_index=True)
    y = pd.Series(y_list, index=np.arange(len(y_list)))
    
    return df_ts, y

# --- 2. Main TSFresh + SVM Function ---
def run_tsfresh_svm(train_loader, test_loader, exp_name="Experiment"):
    print(f"\n=== Running TSFresh + SVM on {exp_name} ===")
    
    # A. Prepare Data (Limit training size for speed if needed)
    print("   Converting data to TSFresh format...")
    # Using 500 samples for Feature Selection is usually enough to find the "Good" features
    df_train, y_train = loader_to_tsfresh_df(train_loader, max_samples=500) 
    df_test, y_test   = loader_to_tsfresh_df(test_loader, max_samples=200)
    
    if df_train is None: return 0.0

    # B. Extract Features
    print("   Extracting features (this may take a minute)...")
    # EfficientFCParameters skips the super expensive features
    extraction_settings = EfficientFCParameters() 
    
    X_train_extracted = extract_features(
        df_train, column_id='id', column_sort='time', 
        default_fc_parameters=extraction_settings,
        impute_function=impute, show_warnings=False,
        disable_progressbar=False
    )
    
    X_test_extracted = extract_features(
        df_test, column_id='id', column_sort='time', 
        default_fc_parameters=extraction_settings,
        impute_function=impute, show_warnings=False,
        disable_progressbar=True
    )
    
    print(f"   Extracted {X_train_extracted.shape[1]} raw features.")

    # C. Feature Selection (Crucial Step)
    # TSFresh filters out features that are irrelevant to 'y'
    print("   Selecting relevant features...")
    X_train_selected = select_features(X_train_extracted, y_train)
    
    # Apply same selection to test set
    # (Only keep columns that exist in both, filling missing with 0)
    features_to_keep = X_train_selected.columns
    X_test_selected = X_test_extracted[features_to_keep]
    # Handle case where test set might produce NaNs or missing cols
    X_test_selected = X_test_selected.reindex(columns=features_to_keep).fillna(0)

    print(f"   ✅ Selected {X_train_selected.shape[1]} relevant features.")
    if X_train_selected.shape[1] == 0:
        print("   ❌ No relevant features found! (Data might be too noisy)")
        return 0.0
    
    # Print top 5 Feature Names to see WHAT physics it found
    # This helps you understand "Why" it works
    print(f"   Top Features: {list(X_train_selected.columns[:5])}")

    # D. SVM Training
    print("   Training SVM...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])
    
    pipe.fit(X_train_selected, y_train)
    
    # E. Evaluate
    y_pred = pipe.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🏆 {exp_name} TSFresh Accuracy: {acc:.2%}")
    print("-" * 30)
    
    return acc

# ==========================================
# EXECUTE
# ==========================================

# 1. Baseline
base_ts_acc = run_tsfresh_svm(baseline_train_loader, baseline_test_loader, "Baseline")

# 2. CV Variation (The problem child)
# Expectation: TSFresh should find "Autocorrelation" or "Energy" features
cv_ts_acc = run_tsfresh_svm(cv_train_loader, cv_test_loader, "CV Variation")

# 3. Mu Variation
mu_ts_acc = run_tsfresh_svm(mu_train_loader, mu_test_loader, "Mu Variation")

# 4. Tac Variation (Memory)
# Expectation: TSFresh should find "Mean Abs Change" or "Fourier" features
tac_ts_acc = run_tsfresh_svm(tac_train_loader, tac_test_loader, "Tac Variation")