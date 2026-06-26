#!/usr/bin/python
from typing import Optional, Tuple 
import torch
import torch.nn as nn
import torch.optim as optim

# === Model Evaluation ===
@torch.no_grad()
def evaluate_model(
    model, 
    test_loader, 
    loss_fn=None, 
    device=None, 
    verbose=True
    ):
    """
    Supports batches shaped:
      - (X, y)
      - (X, y, mask)  -> passed as src_key_padding_mask=mask
      - (X1, X2, y)   -> passed as model(X1, X2)
    Handles BCE/BCEWithLogits and CrossEntropy.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    total_loss, correct, total = 0.0, 0, 0
    
    for batch in test_loader:
        outputs, y_batch = _handle_different_batch_shapes(model, batch, device)
        loss, preds, tgt = _compute_loss_and_accuracy(outputs, y_batch, loss_fn)
        batch_size = tgt.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == tgt).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    loss = (total_loss / total) if (loss_fn and total > 0) else None
    acc = correct / total

    if verbose:
        loss_str = f"{loss:.2f}" if loss is not None else "N/A"
        print(f"Test — loss: {loss_str} | acc: {acc:.2f}")
    return loss, acc

def _handle_different_batch_shapes(
    model: torch.nn.Module,
    batch,
    device: torch.device,
):
    """
    Handle different batch shapes:

      - (X, y)
      - (X, y, mask)           -> src_key_padding_mask=mask
      - (X1, X2, y)            -> pairwise model: model(X1, X2)

    Returns:
        outputs: model outputs tensor
        y_batch: target tensor
    """
    if len(batch) == 2:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)

    elif len(batch) >= 3:
        a, b, c = batch

        # A padding mask is a boolean/byte tensor with shape (batch, time).
        # Otherwise a 3-item batch is treated as pairwise: (X1, X2, y).
        if isinstance(c, torch.Tensor) and c.dtype in (torch.bool, torch.uint8) and c.dim() == 2:
            X_batch, y_batch, mask = a, b, c
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = mask.to(device)
            outputs = model(X_batch, src_key_padding_mask=mask)
        else:
            X1_batch, X2_batch, y_batch = a, b, c
            X1_batch, X2_batch, y_batch = X1_batch.to(device), X2_batch.to(device), y_batch.to(device)
            outputs = model(X1_batch, X2_batch)

    else:
        raise ValueError("Batch must be (X,y), (X,y,mask) or (X1,X2,y).")
    
    return outputs, y_batch


def _compute_loss_and_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Optional[nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified helper for computing loss + predictions + targets
    that supports:
      - nn.BCEWithLogitsLoss (logits)
      - nn.BCELoss (probabilities)
      - nn.CrossEntropyLoss (logits)
    Returns:
        loss: scalar tensor (or 0 if loss_fn=None)
        preds: tensor of predicted class indices (N,)
        tgt: tensor of target class indices (N,)
    """
    if loss_fn is None:
        # no loss function — skip
        loss = torch.tensor(0.0, device=outputs.device)
        preds = outputs.argmax(dim=1)
        tgt = targets
        return loss, preds, tgt

    y_batch_mod = targets
    outputs_mod = outputs
    
    # -------- BCEWithLogitsLoss --------  
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):  
        # make targets float, shape (N,1) when needed  
        y_batch_mod = y_batch_mod.float()
        if y_batch_mod.dim() == 1:
            y_batch_mod = y_batch_mod.unsqueeze(1)               # (N,1)

        # special case: model outputs 2 logits (N,2) → take positive class  
        if outputs_mod.dim() == 2 and outputs_mod.size(1) == 2 and y_batch_mod.size(1) == 1:
            outputs_mod = outputs_mod[:, 1].unsqueeze(1)     # (N,1)
            
        # compute loss
        loss = loss_fn(outputs_mod, y_batch_mod)
        
        preds = (outputs_mod.view(-1) > 0).long()
        tgt = targets.view(-1).long()
        return loss, preds, tgt
    
    # -------- BCELoss --------
    elif isinstance(loss_fn, nn.BCELoss):
        # force float targets
        y_batch_mod = y_batch_mod.float()

        # match shapes (N,1)
        if y_batch_mod.dim() == 1:
            y_batch_mod = y_batch_mod.unsqueeze(1)

        # BCELoss expects probabilities, so use model outputs directly.
        if outputs_mod.dim() == 1 or (outputs_mod.dim() == 2 and outputs_mod.size(1) == 1):
            probs = outputs_mod.view(-1, 1)      # (N,1)

        # Case B: model outputs (N,2) → take positive class probability
        elif outputs_mod.dim() == 2 and outputs_mod.size(1) == 2:
            probs = outputs_mod[:, 1].unsqueeze(1)  # (N,1)
        else:
            raise ValueError(
                "BCELoss expects model outputs with shape (N,), (N, 1), or (N, 2)."
            )

        # compute loss
        loss = loss_fn(probs, y_batch_mod)

        # predictions: threshold probs
        preds = (probs.view(-1) > 0.5).long()
        tgt = targets.view(-1).long()
        return loss, preds, tgt

    # -------- CrossEntropy or others --------
    else:
        loss = loss_fn(outputs, targets)
        preds = outputs.argmax(dim=1)
        tgt = targets
        return loss, preds, tgt
# === Model Evaluation ===


# === Probability Prediction ===
@torch.no_grad()
def predict_proba(model, loader, loss_fn, device):
    model.eval()
    all_probs, all_targets = [], []

    for X, y in loader: #TODO: Correctly Handle X1, X2, y batches
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        probs = _compute_probabilities(logits, loss_fn)

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())

    return torch.cat(all_probs), torch.cat(all_targets)

def _compute_probabilities(
    outputs: torch.Tensor,
    loss_fn: Optional[nn.Module],
) -> torch.Tensor:
    """
    Convert raw model outputs to probabilities in (0,1) or (0,1)^C.
    For:
      - BCEWithLogitsLoss: expects logits -> use sigmoid on positive logit
      - BCELoss: expects probs already; if you passed logits, we sigmoid them
      - CrossEntropyLoss: expects logits -> use softmax
    Returns:
      probs: (N,) for binary, (N,C) for multiclass
    """
    
    # No loss_fn: make a best-effort guess based on shape
    if loss_fn is None:
        if outputs.dim() == 2 and outputs.size(1) > 1:
            return torch.softmax(outputs, dim=1)       # (N,C)
        else:
            return torch.sigmoid(outputs.view(-1))     # (N,)

    # Binary with logits
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
        if outputs.dim() == 2 and outputs.size(1) == 2:
            logits_pos = outputs[:, 1]                 # (N,)
        else:
            logits_pos = outputs.view(-1)              # (N,)
        return torch.sigmoid(logits_pos)               # (N,)

    # Binary with BCELoss: outputs are already probabilities.
    if isinstance(loss_fn, nn.BCELoss):
        if outputs.dim() == 2 and outputs.size(1) == 2:
            return outputs[:, 1]
        return outputs.view(-1)

    # Multiclass CE
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return torch.softmax(outputs, dim=1)           # (N,C)
# === Probability Prediction ===


def logits_to_class_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Convert binary or multiclass logits to class predictions."""
    if logits.ndim > 1 and logits.shape[1] > 1:
        return torch.argmax(logits, dim=1)
    return (torch.sigmoid(logits) > 0.5).float().squeeze()
