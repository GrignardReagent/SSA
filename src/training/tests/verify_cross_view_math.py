import math
import torch
import torch.nn.functional as F
from training.train import cross_view_info_nce

def pure_math_cross_view_info_nce(z1, z2, temperature=0.1):
    """
    Calculates cross-view InfoNCE loss using basic loops and raw math only.
    No matrix multiplication, no PyTorch shortcuts.

    Construction (mirrors cross_view_info_nce exactly):
        all_q = [z1[0..B-1], z2[0..B-1]]  (2B vectors)
        all_k = [z2[0..B-1], z1[0..B-1]]  (2B vectors, views swapped)

    For each of the 2B queries i:
        - Positive   : column i           (diagonal, guaranteed by the swap in K)
        - Self-comp. : column (i+B) % 2B  (same vector as query — excluded)
        - Negatives  : all remaining 2(B-1) columns
    """
    batch_size = len(z1)
    vector_dim = len(z1[0])
    N = 2 * batch_size

    # --- Step 1: L2-normalise each vector manually ---
    def normalize(v):
        magnitude = math.sqrt(sum(val * val for val in v))
        return [val / magnitude for val in v]  # unit vectors → dot product == cosine similarity

    def dot(a, b):
        return sum(a[d] * b[d] for d in range(vector_dim))

    z1_norm = [normalize(v) for v in z1.tolist()]
    z2_norm = [normalize(v) for v in z2.tolist()]

    # --- Step 2: Build query and key banks (note the swap in K) ---
    all_q_norm = z1_norm + z2_norm   # rows 0..B-1 = z1,  B..2B-1 = z2
    all_k_norm = z2_norm + z1_norm   # rows 0..B-1 = z2,  B..2B-1 = z1  ← swapped so positives = diagonal

    total_loss = 0.0

    # --- Step 3-5: Loop over all 2B queries ---
    for i in range(N):
        query    = all_q_norm[i]
        self_col = (i + batch_size) % N  # column where all_q[i] also appears in all_k

        # Numerator: positive is at column i (diagonal entry)
        sim_pos   = dot(query, all_k_norm[i])
        numerator = math.exp(sim_pos / temperature)

        # Denominator: sum over all columns EXCEPT the self-comparison column
        # (self-col is masked to -inf in the PyTorch version → exp(-inf)=0 → skip)
        denominator = 0.0
        for j in range(N):
            if j == self_col:
                continue
            sim_j        = dot(query, all_k_norm[j])
            denominator += math.exp(sim_j / temperature)

        # Cross-entropy contribution for this query: -log(p_positive)
        loss_i     = -math.log(numerator / denominator)
        total_loss += loss_i

    # Average over all 2B queries
    return total_loss / N


def pytorch_shortcut(z1, z2, temperature=0.1):
    """
    The same loss using the PyTorch implementation in training.train.
    Uses F.normalize, matmul, scatter-mask to -inf, and F.cross_entropy.
    The labels vector [0,1,...,2B-1] encodes 'positive is on the diagonal'.
    """
    B = z1.size(0)
    N = 2 * B

    # L2-normalise so cosine similarity == dot product (standard for InfoNCE)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Build the cross-view query / key banks by concatenating both views.
    # Result: positives lie on the main diagonal of the (2B, 2B) logit matrix.
    all_q = torch.cat([z1, z2], dim=0)   # (2B, E)
    all_k = torch.cat([z2, z1], dim=0)   # (2B, E)

    # Full pairwise cosine-similarity matrix, scaled by temperature
    logits = torch.matmul(all_q, all_k.T) / temperature   # (2B, 2B)

    # Mask self-comparisons: query i appears as a key at column (i+B) % 2B.
    # Setting to -inf excludes the position from the softmax without affecting
    # gradient flow through any other entry.
    self_cols = (torch.arange(N, device=z1.device) + B) % N
    logits[torch.arange(N, device=z1.device), self_cols] = float('-inf')

    # Positives are the diagonal; cross-entropy over 2B classes.
    # labels[i] = i means "the positive for query i is at column i".
    # F.cross_entropy(logits, labels) computes -log(softmax(logits[i])[i])
    # averaged over all rows — which is exactly the InfoNCE sum.
    labels = torch.arange(N, device=z1.device)
    return F.cross_entropy(logits, labels).item()


if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH = 16
    DIM   = 8
    TEMP  = 0.1

    z_a = torch.randn(BATCH, DIM)
    z_b = torch.randn(BATCH, DIM)

    loss_pure_math = pure_math_cross_view_info_nce(z_a, z_b, TEMP)
    loss_pytorch   = pytorch_shortcut(z_a, z_b, TEMP)
    loss_package   = cross_view_info_nce(z_a, z_b, TEMP).item()

    print(f"Pure Math Loss : {loss_pure_math:.8f}")
    print(f"PyTorch Loss   : {loss_pytorch:.8f}")
    print(f"cross_view_nce : {loss_package:.8f}")

    match = abs(loss_pure_math - loss_pytorch) < 1e-5
    print(f"Match (<1e-5)  : {'PASS' if match else 'FAIL'}")
