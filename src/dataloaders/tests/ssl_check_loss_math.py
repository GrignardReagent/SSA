import torch
import torch.nn.functional as F
from info_nce import InfoNCE

def manual_simclr_loss(z1, z2, temperature=0.1):
    """
    Manually calculates InfoNCE loss (SimCLR style) from scratch.
    
    Logic:
    1. Normalize vectors.
    2. Compute Similarity Matrix (Dot Product).
    3. Scale by Temperature.
    4. Apply Cross Entropy Loss.
       - Positive: Diagonal elements (z1[i] matches z2[i]).
       - Negatives: All off-diagonal elements.
    """
    # 1. Normalize (Cosine Similarity)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 2. Similarity Matrix (Batch x Batch)
    # logits[i, j] = similarity between z1[i] and z2[j]
    logits = torch.matmul(z1, z2.T)
    
    # 3. Temperature Scaling
    logits /= temperature
    
    # 4. Cross Entropy
    # The "label" for row i is column i (the diagonal)
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size).to(z1.device)
    
    # CrossEntropyLoss automatically does LogSoftmax + NLL
    loss = F.cross_entropy(logits, labels)
    
    return loss

def check_loss_package():
    print("=== INFO NCE MATH CHECK ===")
    
    # 1. Setup Data
    BATCH_SIZE = 64
    DIM = 64
    TEMP = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random embeddings (un-normalized to test if functions normalize correctly)
    z1 = torch.randn(BATCH_SIZE, DIM).to(device)
    z2 = torch.randn(BATCH_SIZE, DIM).to(device)
    
    print(f"Batch Size: {BATCH_SIZE}, Temp: {TEMP}, Device: {device}")

    # 2. Calculate via Package
    # negative_mode='unpaired' is the standard SimCLR setting (batch negatives)
    package_loss_fn = InfoNCE(negative_mode='unpaired', temperature=TEMP)
    loss_pkg = package_loss_fn(z1, z2).item()
    
    # 3. Calculate Manually
    loss_manual = manual_simclr_loss(z1, z2, temperature=TEMP).item()
    
    # 4. Compare
    print(f"\nPackage Loss: {loss_pkg:.8f}")
    print(f"Manual Loss:  {loss_manual:.8f}")
    
    diff = abs(loss_pkg - loss_manual)
    print(f"Difference:   {diff:.8f}")
    
    if diff < 1e-5:
        print("\n✅ SUCCESS: The package calculates loss correctly.")
    else:
        print("\n❌ FAILURE: Mismatch detected.")
        print("Possible reasons:")
        print("1. The package might assume inputs are already normalized.")
        print("2. The package might be using a symmetric loss (z1->z2 AND z2->z1).")

    # 5. Symmetric Check (Optional)
    # Some implementations average (Loss(z1,z2) + Loss(z2,z1)) / 2
    loss_manual_sym = (manual_simclr_loss(z1, z2, TEMP) + manual_simclr_loss(z2, z1, TEMP)) / 2
    print(f"\n(Reference) Symmetric Manual Loss: {loss_manual_sym.item():.8f}")
    
    if abs(loss_pkg - loss_manual_sym.item()) < 1e-5:
         print("💡 Note: The package is using Symmetric Loss.")

if __name__ == "__main__":
    check_loss_package()