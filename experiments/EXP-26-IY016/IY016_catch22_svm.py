from pathlib import Path
from utils.data_loader import baseline_data_prep, save_loader_to_disk
import pycatch22
import pandas as pd
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def make_loaders(DATA_ROOT: Path, results_csv: str, param_dist_threshold: float):
    # build the dataloaders, minimum 2-fold difference in group-making, like in IY011
    # Get the absolute path of the directory containing this script
    RESULTS_PATH = DATA_ROOT / results_csv
    df_params = pd.read_csv(RESULTS_PATH) 
    TRAJ_PATH = [DATA_ROOT / df_params['trajectory_filename'].values[i] for i in range(len(df_params))]
    TRAJ_NPZ_PATH = [traj_file.with_suffix('.npz') for traj_file in TRAJ_PATH]

    # === Dataloader hyperparams & data prep ===
    batch_size = 64
    num_groups_train=3000 
    num_groups_val=int(num_groups_train * 0.2)
    num_groups_test=int(num_groups_train * 0.2)
    num_traj=2
    train_loader, val_loader, test_loader, scaler = baseline_data_prep(
        TRAJ_NPZ_PATH,
        batch_size=batch_size,
        num_groups_train=num_groups_train,
        num_groups_val=num_groups_val,
        num_groups_test=num_groups_test,
        num_traj=num_traj,
        param_dist_threshold=param_dist_threshold,
        verbose=True,
    )
    # === Dataloader hyperparams & data prep ===

    # === Save data for debugging later === 
    # 1. Define paths
    train_save_path = DATA_ROOT / "IY011_static_train.pt"
    val_save_path   = DATA_ROOT / "IY011_static_val.pt"
    test_save_path  = DATA_ROOT / "IY011_static_test.pt"

    # 2. Check if static data already exists
    if not train_save_path.exists():
        print("Static data not found. Saving...")
        
        # Save them to disk
        save_loader_to_disk(train_loader, train_save_path)
        save_loader_to_disk(val_loader, val_save_path)
        save_loader_to_disk(test_loader, test_save_path)
    else:
        print("Found existing static data on disk, the simulated data will not be saved, to prevent overwriting existing data.")
    # === Save data for debugging later === 
    return train_loader, val_loader, test_loader


# --- 1. Helper: Extract Catch22 Features ---
def extract_catch22_from_loader(loader, exp_name):
    """
    Iterates through a DataLoader and extracts 22 canonical features per sample.
    Returns: DataFrame (N, 22), Labels (N,)
    """
    features_list = []
    y_list = []
    
    # We iterate manually to keep it simple and safe
    # Catch22 is fast enough that we don't need complex multiprocessing
    print(f"   Extracting Catch22 features for {exp_name}...")
    
    for X_batch, y_batch in tqdm(loader, leave=False):
        X_numpy = X_batch.numpy().squeeze() # (Batch, Time)
        y_numpy = y_batch.numpy().flatten()
        
        for i in range(X_numpy.shape[0]):
            time_series = X_numpy[i, :]
            
            # Catch22 returns a dictionary: {'names': [...], 'values': [...]}
            # We convert it to a simple dict for DataFrame creation
            c22_out = pycatch22.catch22_all(time_series)
            
            # Map names to values
            feat_dict = dict(zip(c22_out['names'], c22_out['values']))
            features_list.append(feat_dict)
            y_list.append(y_numpy[i])
            
    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)
    y_labels = np.array(y_list)
    
    return df_features, y_labels

# --- 2. Main Catch22 + SVM Function ---
def run_catch22_svm(train_loader, test_loader, exp_name="Experiment"):
    print(f"\n=== Running Catch22 + SVM on {exp_name} ===")
    
    # A. Extract
    X_train, y_train = extract_catch22_from_loader(train_loader, f"{exp_name} (Train)")
    X_test, y_test   = extract_catch22_from_loader(test_loader, f"{exp_name} (Test)")
    
    if len(X_train) == 0: return 0.0

    print(f"   Extracted {X_train.shape[1]} C22 features.")
    
    # B. Train SVM
    # Catch22 features have wildly different scales (some are counts, some are decimals)
    # StandardScaler is mandatory.
    print("   Training SVM...")
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])
    
    pipe.fit(X_train, y_train)
    
    # C. Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🏆 {exp_name} Catch22 Accuracy: {acc:.2%}")
    
    # D. Feature Importance (Permutation) - Optional Check
    # Since we only have 22 features, we can afford to print the most useful one.
    # Simple variance check:
    # If a feature has 0 variance (same for all files), Catch22 might return NaN or const.
    # We quickly handle NaNs (fill with 0) just in case.
    X_train = X_train.fillna(0)
    
    print("-" * 30)
    return acc

# ============================================
# make the loaders
baseline_train_loader, _, baseline_test_loader = make_loaders(
    DATA_ROOT=Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data"),
    results_csv="IY011_simulation_parameters_sobol.csv",
    param_dist_threshold=0.7 # 2-fold difference
)

cv_train_loader, _, cv_test_loader = make_loaders(
    DATA_ROOT=Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data_cv_variation"),
    results_csv="IY011_simulation_cv_parameters_sobol.csv",
    param_dist_threshold=0.7 # 2-fold difference
)

mu_train_loader, _, mu_test_loader = make_loaders(
    DATA_ROOT=Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data_mu_variation"),
    results_csv="IY011_simulation_mu_parameters_sobol.csv",
    param_dist_threshold=0.7 # 2-fold difference
)   

tac_train_loader, _, tac_test_loader = make_loaders(
    DATA_ROOT=Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY011/data_t_ac_variation"),
    results_csv="IY011_simulation_t_ac_parameters_sobol.csv",
    param_dist_threshold=0.7 # 2-fold difference
)


# catch22 + SVM experiments
# 1. Baseline
baseline_catch22_svm_acc = run_catch22_svm(baseline_train_loader, baseline_test_loader, "Baseline")

# 2. CV Variation
cv_catch22_svm_acc = run_catch22_svm(cv_train_loader, cv_test_loader, "CV Variation")

# 3. Mu Variation
mu_catch22_svm_acc = run_catch22_svm(mu_train_loader, mu_test_loader, "Mu Variation")

# 4. Tac Variation
tac_catch22_svm_acc = run_catch22_svm(tac_train_loader, tac_test_loader, "Tac Variation")