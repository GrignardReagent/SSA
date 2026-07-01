from pathlib import Path
from dataloaders import baseline_data_prep, save_loader_to_disk
import pandas as pd
from features.catch22 import run_catch22_svm

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
    train_loader, val_loader, test_loader = baseline_data_prep(
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

# --- 2. Main Catch22 + SVM Function ---

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
