import math
import torch
import torch.nn.functional as F
from info_nce import InfoNCE

def pure_math_infonce(z1, z2, temperature=0.1):
    """
    Calculates InfoNCE loss using basic loops and raw math.
    No matrix multiplication, no PyTorch shortcuts.
    """
    batch_size = len(z1)
    vector_dim = len(z1[0])
    total_loss = 0.0
    
    # --- Step 1: Manually Normalize Vectors ---
    def normalize(v):
        magnitude = math.sqrt(sum(val * val for val in v))
        return [val / magnitude for val in v]
    
    z1_norm = [normalize(v) for v in z1.tolist()]
    z2_norm = [normalize(v) for v in z2.tolist()]
    
    # --- Step 2: Loop through each sample in the batch ---
    for i in range(batch_size):
        anchor = z1_norm[i]
        
        # Calculate Numerator (Positive Match)
        positive = z2_norm[i]
        sim_pos = sum(anchor[d] * positive[d] for d in range(vector_dim))        # Manual Dot Product
        numerator = math.exp(sim_pos / temperature)
        
        # Calculate Denominator (All Matches)
        denominator = 0.0
        for j in range(batch_size):
            candidate = z2_norm[j]
            sim_candidate = sum(anchor[d] * candidate[d] for d in range(vector_dim))            # Manual Dot Product
            denominator += math.exp(sim_candidate / temperature)
            
        # Calculate Probability and Loss
        probability = numerator / denominator
        loss_i = -math.log(probability)
        
        total_loss += loss_i
        
    # Average the loss across the batch
    return total_loss / batch_size

def pytorch_shortcut(z1, z2, temperature=0.1):
    """The PyTorch version for comparison."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    return F.cross_entropy(logits, labels).item()

def infonce_package(z1, z2, temperature=0.1):
    """Using the imported InfoNCE package."""
    # negative_mode='unpaired' matches the standard SimCLR logic 
    # where all other items in the batch are negatives
    loss_fn = InfoNCE(temperature=temperature, negative_mode='unpaired')
    return loss_fn(z1, z2).item()

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Dummy data
    BATCH = 16
    DIM = 8
    TEMP = 0.1
    
    z_a = torch.randn(BATCH, DIM)
    z_b = torch.randn(BATCH, DIM)
    
    loss_pure_math = pure_math_infonce(z_a, z_b, TEMP)
    loss_pytorch = pytorch_shortcut(z_a, z_b, TEMP)
    loss_package = infonce_package(z_a, z_b, TEMP)
    
    print(f"Pure Math Loss: {loss_pure_math:.8f}")
    print(f"PyTorch Loss:   {loss_pytorch:.8f}")
    print(f"InfoNCE Pkg:    {loss_package:.8f}")