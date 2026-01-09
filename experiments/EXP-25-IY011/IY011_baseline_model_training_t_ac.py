#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

# ml
import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerClassifier
from training.eval import evaluate_model
from training.train import train_model 
from classifiers.svm_classifier import svm_classifier
from sklearn.svm import SVC

# data handling
from sklearn.preprocessing import StandardScaler
from utils.data_loader import baseline_data_prep, save_loader_to_disk

# simulation
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model

'''
Train a baseline transformer model on IY011 data with varying t_ac parameters.
'''

# Get the absolute path of the directory containing this script
script_dir = Path(__file__).resolve().parent
DATA_ROOT = script_dir / "temp_data_t_ac_variation"
RESULTS_PATH = DATA_ROOT / "IY011_simulation_t_ac_parameters_sobol.csv" #  this csv file stores all the simulation parameters used
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
)
# === Dataloader hyperparams & data prep ===

# === Save data for debugging later === 
# 1. Define paths
train_save_path = DATA_ROOT / "static_train.pt"
val_save_path   = DATA_ROOT / "static_val.pt"
test_save_path  = DATA_ROOT / "static_test.pt"

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

X_b, y_b = next(iter(train_loader))
print(X_b.shape, y_b.shape) # (Batch, Seq_Len, num_traj), (Batch, 1)

# === Model hyperparams ===
input_size = X_b.shape[2]  # number of input channels/features
num_classes = 2
d_model=64
nhead=4
num_layers=2
dropout=0.001
use_conv1d=False 
max_seq_length = X_b.shape[1] + 100  # e.g., 5020 + 100 = 5120

model = TransformerClassifier(
    input_size=input_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout, 
    use_conv1d=use_conv1d,
    max_seq_length=max_seq_length,
)
# Save location for the trained model
model_path = 'IY011_baseline_transformer_model_7_t_ac.pth'
# === Model hyperparams ===

# === Training hyperparams ===
epochs = 100
patience = 10
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)

### schedulers ### 
# simple scheduler choice
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5) 

loss_fn = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_clip = 1.0
save_path = None
verbose = True

model.to(device)
# === Training hyperparams ===

# === wandb config (required for tracking within train_model) ===
wandb_config = {
    "entity": "grignard-reagent",
    "project": "IY011-baseline-model",
    "name": f"num_groups_train_{num_groups_train}_traj_{num_traj}_batch_size_{batch_size} (t_ac variation)", # change this to what you want
    "dataset": DATA_ROOT.name,
    "batch_size": batch_size,
    "input_size": input_size,
    "d_model": d_model,
    "nhead": nhead,
    "num_layers": num_layers,
    "num_classes": num_classes,
    "dropout": dropout,
    "use_conv1d": use_conv1d,
    "epochs": epochs,
    "patience": patience,
    "lr": lr,
    "optimizer": type(optimizer).__name__,
    "scheduler": type(scheduler).__name__,
    "loss_fn": type(loss_fn).__name__,
    "model": type(model).__name__,
    "batch_size": train_loader.batch_size,
    "num_traj_per_group": num_traj,
    "num_groups_train": num_groups_train,
    "num_groups_val": num_groups_val,
    "num_groups_test": num_groups_test,
    "model_path": model_path,
}
# === wandb config === 

history = train_model(
    model,
    train_loader,
    val_loader,
    epochs=epochs,
    patience=patience,
    lr=lr,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    grad_clip=grad_clip,
    save_path=save_path,
    verbose=verbose,
    wandb_logging=True, # this enables wandb logging within train_model
    wandb_config=wandb_config, # pass the config dictionary
)

# save the trained model
torch.save(model.state_dict(), model_path)
print('Model saved to', model_path)


###########################################################################################
# EVALUATION
###########################################################################################

print("\n=== Evaluating on Test Set ===")
# evaluate on test set
test_loss, test_acc = evaluate_model(
    model,
    test_loader,
    loss_fn,
    device,
)

# ==========================================
# SVM BENCHMARK
# ==========================================

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


# 1. Extract Data dynamically from your loaders
X_train_svm, y_train_svm = extract_data_for_svm(train_loader)
X_test_svm, y_test_svm   = extract_data_for_svm(test_loader)

print(f"SVM Train Shape: {X_train_svm.shape}")
print(f"SVM Test Shape:  {X_test_svm.shape}")

# 2. Run the SVM Classifier
svm_accuracy = svm_classifier(
    X_train_svm,
    X_test_svm,
    y_train_svm,
    y_test_svm,
)

# ==========================================
# 1. GENERATE UNSEEN DATA
# ==========================================
def generate_unseen_classes(n_classes=5, n_trajs_per_class=20, seq_len=3000):
    """
    Generates N new 'mystery' datasets with random parameters.
    Returns a list of dictionaries, each representing a unique biological condition.
    """
    print(f"Generating {n_classes} unseen classes...")
    unseen_data = []
    
    # Randomly sample parameters (Mu, CV, Tac)
    # We pick ranges similar to your training data to ensure they are valid biological possibilities
    mus = np.random.uniform(1, 10000, n_classes)
    cvs = np.random.uniform(0.5, 2.0, n_classes)
    tacs = np.random.uniform(5, 120, n_classes)
    
    time_points = np.arange(0, seq_len, 1.0)

    for i in range(n_classes):
        try:
            # 1. Solve for physical parameters
            rho, d, sigma_b, sigma_u = find_tilda_parameters(mus[i], tacs[i], cvs[i])
            
            # 2. Simulate
            params = [{"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}]
            df = simulate_telegraph_model(params, time_points, n_trajs_per_class)
            
            # 3. Extract clean array (N_trajs, Time)
            # Assuming format: columns t_0...t_T, rows are trajectories
            # Adjust filtering based on your exact df structure
            trajs = df.drop(columns=['label'], errors='ignore').values
            
            unseen_data.append({
                "class_id": f"Mystery_Class_{i}",
                "parameters": {"mu": mus[i], "cv": cvs[i], "tac": tacs[i]},
                "trajectories": trajs
            })
            print(f"  Generated Class {i}: Mu={mus[i]:.1f}, CV={cvs[i]:.2f}")
            
        except Exception as e:
            print(f"  Skipped a class due to solver error: {e}")
            
    return unseen_data

# === RUN GENERATION ===
unseen_datasets = generate_unseen_classes(n_classes=5, n_trajs_per_class=20)

# ==========================================
# 2. EVALUATE BASELINE MODEL on UNSEEN CLASSES
# ==========================================
def instance_norm_np(x):
    """
    Normalize a single trajectory (Time,) or (Time, 1) to Mean=0, Std=1.
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

def prepare_group_input(pool, num_traj, crop_len, device):
    """
    Helper to sample, process, and concatenate a group of trajectories.
    Returns: Tensor of shape (1, Total_Seq_Len, 1)
    """
    # 1. Randomly sample 'num_traj' trajectories
    indices = np.random.choice(len(pool), num_traj, replace=True)
    selected = [pool[i] for i in indices]
    
    processed_parts = []
    for traj in selected:
        # A. Random Crop
        if traj.shape[0] > crop_len:
            start = np.random.randint(0, traj.shape[0] - crop_len)
            crop = traj[start : start + crop_len]
        else:
            # Pad if too short
            crop = np.pad(traj, (0, crop_len - len(traj)), mode='edge')
            
        # B. Instance Normalization (Critical!)
        norm = instance_norm_np(crop)
        
        # Ensure (Time, 1) shape
        if norm.ndim == 1:
            norm = norm.reshape(-1, 1)
            
        processed_parts.append(norm)
        
    # 2. Concatenate along Time Axis (Axis 0)
    # Result: (num_traj * crop_len, 1)
    concat_group = np.concatenate(processed_parts, axis=0)
    
    # 3. To Tensor & Batch Dim -> (1, Total_Time, 1)
    tensor_in = torch.tensor(concat_group, dtype=torch.float32).unsqueeze(0).to(device)
    
    return tensor_in

def test_baseline_performance(model, unseen_data, scaler=None, num_traj=10, crop_len=1811, samples_per_class=50):
    """
    Tests One-Shot Classification on unseen classes.
    Updated for (Batch, Time*N, 1) input shape + Instance Norm.
    
    Args:
        scaler: Ignored (kept for compatibility), Instance Norm is used internally.
    """
    model.eval()
    # Auto-detect device
    device = next(model.parameters()).device
    
    correct = 0
    total_queries = 0
    
    print(f"\n=== Testing Baseline Model (Feature Distance) ===")
    print(f"Model expects groups of {num_traj} trajectories (Concatenated Time).")
    
    # --- Phase 1: Build Reference Prototypes (Support Set) ---
    class_prototypes = {}
    
    for data in unseen_data:
        pool = data['trajectories']
        ref_embeddings = []
        
        # Average 5 examples to build a robust prototype
        for _ in range(5):
            tensor_in = prepare_group_input(pool, num_traj, crop_len, device)
            
            with torch.no_grad():
                # Get feature vector (1, d_model)
                emb = model.encode(tensor_in).cpu().numpy()
                ref_embeddings.append(emb)
        
        # Mean Prototype
        class_prototypes[data['class_id']] = np.mean(ref_embeddings, axis=0).flatten()
        
    print(f"✅ Built prototypes for {len(class_prototypes)} classes.")

    # --- Phase 2: Classification (Query Set) ---
    for data in unseen_data:
        true_class = data['class_id']
        pool = data['trajectories']
        
        for _ in range(samples_per_class):
            # Prepare Query
            tensor_in = prepare_group_input(pool, num_traj, crop_len, device)
            
            # Extract Feature
            with torch.no_grad():
                query_vec = model.encode(tensor_in).cpu().numpy().flatten()
            
            # Nearest Neighbor Classification
            best_dist = float('inf')
            predicted = None
            
            for cls_id, proto in class_prototypes.items():
                dist = np.linalg.norm(query_vec - proto)
                if dist < best_dist:
                    best_dist = dist
                    predicted = cls_id
            
            if predicted == true_class:
                correct += 1
            total_queries += 1

    acc = correct / total_queries if total_queries > 0 else 0
    print(f"Baseline One-Shot Accuracy: {acc:.2%}")
    return acc

print("\n=== Evaluating Baseline Model on Unseen Classes ===")
acc = test_baseline_performance(model, unseen_datasets, scaler, num_traj=num_traj)

# ==========================================
# SVM ON UNSEEN DATA BENCHMARK
# ==========================================

def test_svm_few_shot(unseen_data, crop_len=1811, n_support=5):
    """
    Tests how well a standard SVM performs on unseen classes given only
    a few examples (n_support) per class.
    
    This serves as a baseline: "Do we really need Deep Learning embeddings, 
    or are the raw time series distinct enough for an SVM?"
    """
    print(f"\n=== Testing SVM Few-Shot Baseline (Raw Data) ===")
    
    # 1. Prepare Training Data (Support Set)
    # We take the first 'n_support' trajectories from EACH class to train the SVM
    X_support = []
    y_support = []
    
    for label_idx, data in enumerate(unseen_data):
        # Take first 5 examples
        refs = data['trajectories'][:n_support] 
        
        for traj in refs:
            # Crop/Pad to fix length
            if len(traj) > crop_len:
                crop = traj[:crop_len]
            else:
                crop = np.pad(traj, (0, crop_len - len(traj)), mode='edge')
            
            X_support.append(crop)
            y_support.append(label_idx) # Use integer labels (0, 1, 2...)
            
    X_support = np.array(X_support)
    y_support = np.array(y_support)
    
    # 2. Fit the SVM
    # We essentially train a mini-classifier just for this specific "Unseen" problem
    scaler = StandardScaler()
    X_support_scaled = scaler.fit_transform(X_support)
    
    clf = SVC(kernel='rbf', C=1.0, gamma='scale') # Standard RBF SVM
    clf.fit(X_support_scaled, y_support)
    print(f"✅ Trained SVM on {len(X_support)} support samples.")
    
    # 3. Evaluate on Query Data (The rest of the trajectories)
    correct = 0
    total = 0
    
    for label_idx, data in enumerate(unseen_data):
        queries = data['trajectories'][n_support:]
        
        for traj in queries:
            # Same preprocessing
            if len(traj) > crop_len:
                start = np.random.randint(0, len(traj) - crop_len)
                crop = traj[start : start+crop_len]
            else:
                crop = np.pad(traj, (0, crop_len - len(traj)), mode='edge')
                
            # Flatten and Scale
            # Note: SVM expects (1, Features), so we reshape
            feat_vec = crop.reshape(1, -1) 
            feat_vec = scaler.transform(feat_vec)
            
            # Predict
            pred = clf.predict(feat_vec)[0]
            
            if pred == label_idx:
                correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"SVM Few-Shot Accuracy: {acc:.2%}")
    return acc

print("\n=== Evaluating SVM Few-Shot on Unseen Classes ===")
acc_svm_few_shot = test_svm_few_shot(unseen_datasets)

# ==========================================
# PERMUTATION TESTS
# ==========================================

def run_permutation_test(model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), sep_len=1, num_traj=2):
    """
    Evaluates the model on:
    1. Original Data (Preserves Temporal Patterns)
    2. Shuffled Data (Destroys Temporal Patterns, Preserves Stats & Structure)
    
    Generalized to handle any num_traj (2, 4, etc.).
    """
    model.eval()
    model.to(device)
    
    correct_original = 0
    correct_shuffled = 0
    total = 0
    
    print(f"Running Permutation Test (num_traj={num_traj}, sep_len={sep_len})...")
    
    with torch.no_grad():
        for X, y in test_loader:
            # Move to device
            X = X.to(device)
            y = y.to(device).squeeze() 
            
            # --- Helper to get predictions ---
            def get_preds(logits):
                if logits.shape[1] > 1:
                    return torch.argmax(logits, dim=1)
                else:
                    return (torch.sigmoid(logits) > 0.5).float().squeeze()

            # --- 1. Test on ORIGINAL (Ordered) ---
            logits_orig = model(X)
            preds_orig = get_preds(logits_orig)
            correct_original += (preds_orig == y).sum().item()
            
            # --- 2. Test on SHUFFLED (Structure Preserved) ---
            X_shuffled = X.clone()
            B, T, C = X.shape
            
            # Logic to determine Segment Length (L) for any num_traj
            # T = (num_traj * L) + ((num_traj - 1) * sep_len)
            
            if num_traj > 1:
                total_sep_space = (num_traj - 1) * sep_len
                remaining_space = T - total_sep_space
                L = remaining_space // num_traj
                
                # Validation: Check if the math adds up perfectly
                if L * num_traj + total_sep_space != T:
                    print(f"⚠️ Warning: Dimension mismatch! T={T} doesn't fit num_traj={num_traj}, sep={sep_len}. Shuffling globally.")
                    # Fallback: Global Shuffle
                    for i in range(B):
                        idx = torch.randperm(T)
                        X_shuffled[i] = X[i, idx, :]
                else:
                    # Component-wise Shuffle Loop
                    for i in range(B):
                        current_pos = 0
                        for k in range(num_traj):
                            # Define start and end of the k-th trajectory segment
                            seg_start = current_pos
                            seg_end = current_pos + L
                            
                            # Shuffle this specific segment
                            idx_segment = torch.randperm(L)
                            X_shuffled[i, seg_start:seg_end, :] = X[i, seg_start:seg_end, :][idx_segment]
                            
                            # Move pointer: Skip the segment we just shuffled + the separator
                            current_pos = seg_end + sep_len
            else:
                # num_traj=1 case (Simple global shuffle)
                for i in range(B):
                    idx = torch.randperm(T)
                    X_shuffled[i] = X[i, idx, :]
            
            logits_shuff = model(X_shuffled)
            preds_shuff = get_preds(logits_shuff)
            correct_shuffled += (preds_shuff == y).sum().item()
            
            total += B
    
    # Calculate Final Accuracy
    acc_orig = correct_original / total
    acc_shuff = correct_shuffled / total
    
    print(f"------------------------------------------------")
    print(f"Accuracy on ORIGINAL Data:  {acc_orig:.2%}")
    print(f"Accuracy on SHUFFLED Data:  {acc_shuff:.2%} (Structure Preserved)")
    print(f"------------------------------------------------")
    
    return acc_orig, acc_shuff

print("\n=== Running Permutation Test ===")
acc_orig, acc_shuff = run_permutation_test(model, test_loader, device=device)

# Test baseline performance on unseen data, both original and shuffled
acc = test_baseline_performance(model, unseen_datasets, scaler, num_traj=num_traj)