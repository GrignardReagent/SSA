import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import numpy as np
import yaml
from pathlib import Path

# Import your existing modules
from models.transformer import TransformerClassifier
from training.train import train_model
from dataloaders import load_loader_from_disk

# ==========================================
# GLOBAL VARIABLES
# ==========================================
EXPERIMENTS_LOOKUP = {}  # Populated from YAML in __main__
LOADER_CACHE = {}        # Caches dataloaders: { 'ExperimentName': (train_loader, val_loader, input_size) }

def get_data_for_experiment(exp_name):
    """
    Fetches dataloaders for a specific experiment name.
    Uses a cache to avoid reloading from disk on every sweep iteration.
    """
    global LOADER_CACHE, EXPERIMENTS_LOOKUP
    
    # 1. Check Cache
    if exp_name in LOADER_CACHE:
        return LOADER_CACHE[exp_name]

    # 2. Look up configuration
    exp_config = next((item for item in EXPERIMENTS_LOOKUP if item["name"] == exp_name), None)
    if not exp_config:
        raise ValueError(f"Experiment '{exp_name}' not found in YAML 'experiments_definition'.")

    # 3. Resolve Paths
    print(f"Loading data for experiment: {exp_name}")
    DATA_ROOT = Path(exp_config["data_dir"])
    train_save_path = DATA_ROOT / exp_config["train_save_path"]
    val_save_path   = DATA_ROOT / exp_config["val_save_path"]

    # 4. Load Data
    train_loader = load_loader_from_disk(train_save_path)
    val_loader = load_loader_from_disk(val_save_path)

    # 5. Determine Input Dimensions from the data itself
    X_b, _ = next(iter(train_loader))
    input_size = X_b.shape[2]
    
    # 6. Cache and Return
    LOADER_CACHE[exp_name] = (train_loader, val_loader, input_size)
    return train_loader, val_loader, input_size

# ==========================================
# DEFINE TRAINING FUNCTION
# ==========================================
def run_sweep_agent():
    with wandb.init() as run:
        config = wandb.config

        # --- 1. GET DATASET BASED ON CONFIG ---
        try:
            exp_name = config.experiment_name
            train_loader, val_loader, input_size = get_data_for_experiment(exp_name)
        except AttributeError:
            # Fallback if experiment_name is missing from parameters
            print("⚠️ 'experiment_name' not found in config. Defaulting to 'Baseline'.")
            train_loader, val_loader, input_size = get_data_for_experiment("Baseline")

        # Constants inferred from data
        max_seq_length = 0
        # Peek at a batch to get sequence length (safer than hardcoding)
        X_b, _ = next(iter(train_loader))
        max_seq_length = X_b.shape[1] + 100
        num_classes = 2

        # --- 2. DYNAMIC NAMING ---
        # Name: Dataset - Hyperparams
        run_name = f"[{exp_name}] L{config.num_layers}-H{config.nhead}-D{config.d_model}-drop{config.dropout}"
        if config.use_conv1d:
            run_name += "-CNN"
        
        run.name = run_name

        # --- 3. CONSTRAINT CHECK ---
        # d_model must be divisible by nhead for PyTorch Transformer
        if config.d_model % config.nhead != 0:
            adjusted_d_model = (config.d_model // config.nhead) * config.nhead
            if adjusted_d_model == 0: adjusted_d_model = config.nhead
            d_model = adjusted_d_model
            print(f"⚠️ Adjusted d_model from {config.d_model} to {adjusted_d_model} to match nhead {config.nhead}")
        else:
            d_model = config.d_model

        # --- 4. INIT MODEL ---
        model = TransformerClassifier(
            input_size=input_size,
            d_model=d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            num_classes=num_classes,
            dropout=config.dropout,
            use_conv1d=config.use_conv1d,
            max_seq_length=max_seq_length,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # --- 5. SETUP OPTIMIZER ---
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        loss_fn = nn.BCEWithLogitsLoss()

        # --- 6. TRAIN ---
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=config.epochs,
            patience=config.patience,
            lr=config.lr,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            wandb_logging=False, # Manual logging below
            verbose=True
        )

        # --- 7. LOGGING TO SWEEP ---
        # Use zip to handle mismatched lengths safely (stops at the shortest list)
        for i, (t_loss, t_acc, v_loss, v_acc, current_lr) in enumerate(zip(
            history['train_loss'], 
            history['train_acc'], 
            history['val_loss'], 
            history['val_acc'],
            history['lr']
        )):
            wandb.log({
                "epoch": i + 1,
                "train_loss": t_loss,
                "train_acc": t_acc,
                "val_loss": v_loss,
                "val_acc": v_acc,
                "lr": current_lr  # Log the learning rate
            })
            
        # Optional: Log summary of best result
        if history['val_acc']:
            print(f"Sweep Run Finished. Best Val Acc: {max(history['val_acc']):.2%}")

# ==========================================
# EXECUTE SWEEP
# ==========================================
if __name__ == "__main__":
    # 1. Load YAML Configuration
    config_path = "IY018_baseline_sweep_config.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # 2. Extract Experiments Definition (Global)
    if "experiments_definition" in full_config:
        EXPERIMENTS_LOOKUP = full_config["experiments_definition"]
        # Remove it from the dict passed to wandb to keep the config clean
        del full_config["experiments_definition"]
    else:
        raise ValueError(f"Missing 'experiments_definition' in {config_path}")

    # 3. Initialize the sweep
    sweep_id = wandb.sweep(sweep=full_config, project="IY018-baseline-sweep")
    
    # 4. Start the agent
    wandb.agent(sweep_id, function=run_sweep_agent, count=100)