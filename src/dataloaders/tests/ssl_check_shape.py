import torch
import torch.nn as nn
from info_nce import InfoNCE
from models.ssl_transformer import SSL_Transformer

def run_diagnostics():
    print("=== STARTING SSL DIAGNOSTICS ===")
    
    # 1. Setup Dummy Data
    BATCH_SIZE = 64
    TIME_STEPS = 200
    INPUT_DIM = 1
    D_MODEL = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create random batch (Batch, Time, Feat)
    x1 = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_DIM).to(device)
    
    # Create a "Partner" batch
    # Case A: Identical (Should yield minimum loss)
    x2_identical = x1.clone()
    
    # Case B: Random (Should yield high loss)
    x2_random = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_DIM).to(device)

    # 2. Initialize Model
    model = SSL_Transformer(
        input_size=INPUT_DIM,
        d_model=D_MODEL,
        nhead=4,
        num_layers=2
    ).to(device)
    model.eval()

    # 3. Check Forward Pass & Shapes
    print("\n--- checking Shapes ---")
    with torch.no_grad():
        z1, z2 = model(x1, x2_random)
    
    print(f"Input Shape:  {x1.shape}")
    print(f"Output Shape: {z1.shape}")
    
    expected_shape = (BATCH_SIZE, D_MODEL)
    if z1.shape == expected_shape:
        print("✅ Shape Check PASSED: (Batch, d_model)")
    else:
        print(f"❌ Shape Check FAILED: Expected {expected_shape}, got {z1.shape}")
        # If this fails, it often means the backbone isn't pooling correctly (e.g. returning [Batch, Time, Dim])

    # 4. Check Loss Function Logic
    print("\n--- checking InfoNCE Loss Logic ---")
    loss_fn = InfoNCE(negative_mode='unpaired', temperature=0.1)
    
    with torch.no_grad():
        # A. Perfect Match (z1 vs z1)
        # Note: Even with identical vectors, SimCLR loss isn't 0 because it compares against negatives.
        # But it should be the mathematical minimum.
        # Min Loss ~= -log(1 / BatchSize) if temperature is perfect, but usually higher.
        # With Batch=64, random chance is -log(1/64) = 4.15.
        # Perfect alignment should be significantly lower than 4.15.
        
        z1, z1_clone = model(x1, x2_identical)
        loss_perfect = loss_fn(z1, z1_clone).item()
        
        # B. Random Noise (z1 vs z_random)
        z1, z2_rand = model(x1, x2_random)
        loss_random = loss_fn(z1, z2_rand).item()
        
    print(f"Loss (Perfect Match): {loss_perfect:.4f}")
    print(f"Loss (Random Inputs): {loss_random:.4f}")
    
    # Theoretical Random Loss calculation: -log(1/N)
    random_baseline = -torch.log(torch.tensor(1.0/BATCH_SIZE)).item()
    print(f"Theoretical Random Baseline (-ln(1/N)): {random_baseline:.4f}")

    if loss_perfect < loss_random:
        print("✅ Logic Check PASSED: Perfect match loss is lower than random.")
    else:
        print("❌ Logic Check FAILED: Loss is not discriminating correctly.")

    if loss_random < random_baseline:
        print("⚠️  Warning: Random Loss is lower than theoretical random baseline. Check normalization.")
    
    # 5. Check Normalization
    print("\n--- checking Embedding Normalization ---")
    norm_z1 = torch.norm(z1, dim=1).mean().item()
    print(f"Average Norm of Embeddings (before explicit normalize): {norm_z1:.4f}")
    
    # InfoNCE usually normalizes internally, but if your vectors are exploding, 
    # it might cause numerical instability. Ideally, this should be reasonable (e.g., around sqrt(d_model) or 1).

if __name__ == "__main__":
    run_diagnostics()