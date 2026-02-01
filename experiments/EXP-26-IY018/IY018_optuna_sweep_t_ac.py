import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import numpy as np
import yaml
import gc
import pandas as pd
import optuna
from pathlib import Path
import os
from datetime import datetime

# Import your existing modules
from models.transformer import TransformerClassifier
from training.train import train_model
from dataloaders import load_loader_from_disk

# ==========================================
# CONFIGURATION
# ==========================================
YAML_CONFIG_PATH = "IY018_sweep_config_t_ac.yaml"
CSV_RESULTS_PATH = "IY018_optuna_sweep_t_ac_results.csv"
N_TRIALS = 100

# ==========================================
# GLOBAL CACHE
# ==========================================
EXPERIMENTS_LOOKUP = {} 
LOADER_CACHE = {}

def get_data_for_experiment(exp_name):
    """
    Fetches dataloaders for a specific experiment name.
    Uses a cache to avoid reloading from disk on every iteration.
    """
    global LOADER_CACHE, EXPERIMENTS_LOOKUP
    
    if exp_name in LOADER_CACHE:
        return LOADER_CACHE[exp_name]

    # Find the config for this experiment name
    exp_config = next((item for item in EXPERIMENTS_LOOKUP if item["name"] == exp_name), None)
    
    # Fallback to Baseline if not found
    if not exp_config:
        print(f"⚠️ Experiment '{exp_name}' not found. Defaulting to first available.")
        exp_config = EXPERIMENTS_LOOKUP[0]

    print(f"📂 Loading data for experiment: {exp_config['name']}")
    DATA_ROOT = Path(exp_config["data_dir"])
    train_save_path = DATA_ROOT / exp_config["train_save_path"]
    val_save_path   = DATA_ROOT / exp_config["val_save_path"]

    train_loader = load_loader_from_disk(train_save_path)
    val_loader = load_loader_from_disk(val_save_path)

    # Determine Input Dimensions from the data
    X_b, _ = next(iter(train_loader))
    input_size = X_b.shape[2]
    
    LOADER_CACHE[exp_name] = (train_loader, val_loader, input_size)
    return train_loader, val_loader, input_size

def save_to_csv(result_dict):
    """Appends a single trial result to the CSV file."""
    df = pd.DataFrame([result_dict])
    if not os.path.isfile(CSV_RESULTS_PATH):
        df.to_csv(CSV_RESULTS_PATH, index=False)
    else:
        df.to_csv(CSV_RESULTS_PATH, mode='a', header=False, index=False)

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================
def objective(trial):
    # 1. Parse Hyperparameters using Optuna
    with open(YAML_CONFIG_PATH, "r") as f:
        full_config = yaml.safe_load(f)
    
    params_def = full_config.get("parameters", {})
    config = {}

    # Map YAML params to Optuna suggestions
    for key, val_dict in params_def.items():
        if "values" in val_dict:
            config[key] = trial.suggest_categorical(key, val_dict["values"])
        elif "value" in val_dict:
            config[key] = val_dict["value"]
        elif "min" in val_dict and "max" in val_dict:
            if key == "lr":
                config[key] = trial.suggest_float(key, val_dict["min"], val_dict["max"], log=True)
            else:
                config[key] = trial.suggest_float(key, val_dict["min"], val_dict["max"])
    
    # Ensure Experiment Name exists
    if "experiment_name" not in config:
        config["experiment_name"] = "Baseline" 

    model = None
    optimizer = None
    scheduler = None
    best_val_acc = 0.0

    try:
        # --- PREPARE DATA ---
        train_loader, val_loader, input_size = get_data_for_experiment(config["experiment_name"])
        
        # Constants inferred from data
        X_b, _ = next(iter(train_loader))
        max_seq_length = X_b.shape[1] + 100
        num_classes = 2

        # --- CONSTRAINT CHECK ---
        # d_model must be divisible by nhead
        if config['d_model'] % config['nhead'] != 0:
            adjusted = (config['d_model'] // config['nhead']) * config['nhead']
            if adjusted == 0: adjusted = config['nhead']
            print(f"⚠️ Adjusted d_model {config['d_model']} -> {adjusted} (nhead={config['nhead']})")
            config['d_model'] = adjusted

        # --- MODEL & OPTIMIZER ---
        model = TransformerClassifier(
            input_size=input_size,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            dropout=config['dropout'],
            use_conv1d=config['use_conv1d'],
            max_seq_length=max_seq_length,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # Scheduler with hardcoded patience of 5 (as requested)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        loss_fn = nn.BCEWithLogitsLoss()

        # --- WANDB CONFIG PREPARATION ---
        # Constructing the rich configuration payload
        wandb_payload = {
            "entity": "grignard-reagent",
            "project": "IY018-baseline-sweep",
            "name": f"Trial{trial.number}_[{config['experiment_name']}]_L{config['num_layers']}_H{config['nhead']}_D{config['d_model']}",
            "group": "Optuna_Bayesian_Sweep_t_ac",
            "tags": ["Optuna", config["experiment_name"]],
            
            # --- Hyperparameters ---
            "dataset": config["experiment_name"],
            "batch_size": train_loader.batch_size, # Log the actual batch size from loader
            "input_size": input_size,
            "d_model": config['d_model'],
            "nhead": config['nhead'],
            "num_layers": config['num_layers'],
            "num_classes": num_classes,
            "dropout": config['dropout'],
            "use_conv1d": config['use_conv1d'],
            "epochs": 100,      # Hardcoded as requested
            "patience": 15,     # Hardcoded as requested
            "lr": config['lr'],
            
            # --- Object Types ---
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__,
            "loss_fn": type(loss_fn).__name__,
            "model": type(model).__name__,
            
            # --- Inferred Data Metadata ---
            # (Note: num_traj, groups, etc. are not available here as we load pre-baked .pt files)
            "max_seq_length": max_seq_length,
            
            # --- Full Optuna Config (Merge everything else) ---
            **config 
        }

        # --- TRAINING ---
        # wandb_logging=True activates the internal logger in train.py
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=100,      # Using direct values
            patience=15,     # Using direct values
            lr=config['lr'],
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            wandb_logging=True,        # <--- ENABLE INTERNAL LOGGING
            wandb_config=wandb_payload, # <--- PASS RICH CONFIG
            verbose=True
        )

        # --- SAVE RESULTS TO LOCAL CSV ---
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        best_epoch = history['val_acc'].index(best_val_acc) + 1 if history['val_acc'] else 0
        
        csv_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trial_id": trial.number,
            "experiment": config["experiment_name"],
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "final_train_loss": history['train_loss'][-1] if history['train_loss'] else None,
            "final_val_loss": history['val_loss'][-1] if history['val_loss'] else None,
            **config 
        }
        save_to_csv(csv_record)
        
        print(f"✅ Trial {trial.number} Finished. Best Val Acc: {best_val_acc:.2%}")
        return best_val_acc

    except Exception as e:
        print(f"❌ Trial {trial.number} Failed: {e}")
        return 0.0

    finally:
        # --- CLEANUP ---
        # Ensure the run is closed even if train_model crashed
        if wandb.run is not None:
            wandb.finish()
            
        if model: del model
        if optimizer: del optimizer
        if scheduler: del scheduler
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 Memory Cleared")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Experiments Definition Global
    with open(YAML_CONFIG_PATH, "r") as f:
        full_yaml = yaml.safe_load(f)
        EXPERIMENTS_LOOKUP = full_yaml.get("experiments_definition", [])
    
    print(f"🚀 Starting Optuna Bayesian Sweep for {N_TRIALS} rounds...")
    print(f"📂 Saving results to {CSV_RESULTS_PATH}")
    
    # 2. Create Optuna Study
    study = optuna.create_study(direction="maximize", study_name="IY018_Bayesian")
    
    # 3. Run Optimization
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n==========================================")
    print("🏆 Sweep Complete.")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value: {study.best_value}")
