"""Model diagnostic tests based on controlled input perturbations."""

from __future__ import annotations

import torch


def _logits_to_class_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Convert binary or multiclass logits to class predictions."""
    if logits.ndim > 1 and logits.shape[1] > 1:
        return torch.argmax(logits, dim=1)
    return (torch.sigmoid(logits) > 0.5).float().squeeze()


@torch.no_grad()
def run_permutation_test(
    model,
    test_loader,
    device=None,
    sep_len: int = 1,
    num_traj: int = 2,
) -> tuple[float, float]:
    """Compare accuracy on original data and within-trajectory-shuffled data."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct_original = 0
    correct_shuffled = 0
    total = 0

    print(f"Running Permutation Test (num_traj={num_traj}, sep_len={sep_len})...")

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device).squeeze()

        logits_orig = model(X)
        preds_orig = _logits_to_class_predictions(logits_orig)
        correct_original += (preds_orig == y).sum().item()

        X_shuffled = X.clone()
        batch_size, total_time, _ = X.shape

        if num_traj > 1:
            total_sep_space = (num_traj - 1) * sep_len
            remaining_space = total_time - total_sep_space
            segment_len = remaining_space // num_traj

            if segment_len * num_traj + total_sep_space != total_time:
                print(
                    "Warning: dimension mismatch; "
                    f"T={total_time}, num_traj={num_traj}, sep={sep_len}. "
                    "Shuffling globally."
                )
                for i in range(batch_size):
                    X_shuffled[i] = X[i, torch.randperm(total_time, device=device), :]
            else:
                for i in range(batch_size):
                    current_pos = 0
                    for _ in range(num_traj):
                        seg_start = current_pos
                        seg_end = current_pos + segment_len
                        idx_segment = torch.randperm(segment_len, device=device)
                        X_shuffled[i, seg_start:seg_end, :] = X[
                            i, seg_start:seg_end, :
                        ][idx_segment]
                        current_pos = seg_end + sep_len
        else:
            for i in range(batch_size):
                X_shuffled[i] = X[i, torch.randperm(total_time, device=device), :]

        logits_shuff = model(X_shuffled)
        preds_shuff = _logits_to_class_predictions(logits_shuff)
        correct_shuffled += (preds_shuff == y).sum().item()
        total += batch_size

    acc_orig = correct_original / total
    acc_shuff = correct_shuffled / total

    print("------------------------------------------------")
    print(f"Accuracy on ORIGINAL Data:  {acc_orig:.2%}")
    print(f"Accuracy on SHUFFLED Data:  {acc_shuff:.2%} (Structure Preserved)")
    print("------------------------------------------------")

    return acc_orig, acc_shuff
