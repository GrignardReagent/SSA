#!/usr/bin/python
from typing import Optional, Tuple 
import torch
import torch.nn as nn
import torch.optim as optim

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
    Handles BCE/BCEWithLogits and CrossEntropy.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    total_loss, correct, total = 0.0, 0, 0
    
    for batch in test_loader:
        if len(batch) == 2: # handle (X, y)
            X_batch, y_batch = batch
            mask = None
        elif len(batch) >= 3: # handle (X, y, mask)
            X_batch, y_batch, mask = batch[0], batch[1], batch[2]
        else:
            raise ValueError("Batch must be (X,y) or (X,y,mask, ...).")
        
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch) if mask is None else model(X_batch, src_key_padding_mask=mask.to(device))
        
        loss, preds, tgt = _compute_loss_and_accuracy(outputs, y_batch, loss_fn)
        total_loss += loss.item() * X_batch.size(0)
        correct += (preds == tgt).sum().item()
        total += tgt.size(0)

    loss = (total_loss / total) if (loss_fn and total > 0) else None
    acc = correct / total

    if verbose:
        loss_str = f"{loss:.2f}" if loss is not None else "N/A"
        print(f"Test — loss: {loss_str} | acc: {acc:.2f}")
    return loss, acc


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

        # Convert model outputs → probabilities
        # Case A: model outputs (N,) or (N,1) single logit → use sigmoid
        if outputs_mod.dim() == 1 or (outputs_mod.dim() == 2 and outputs_mod.size(1) == 1):
            probs = torch.sigmoid(outputs_mod.view(-1, 1))      # (N,1)

        # Case B: model outputs (N,2) → take positive class prob via softmax
        elif outputs_mod.dim() == 2 and outputs_mod.size(1) == 2:
            probs = torch.softmax(outputs_mod, dim=1)[:, 1].unsqueeze(1)  # (N,1)

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


@torch.no_grad()
def predict_proba(model, loader, loss_fn, device):
    model.eval()
    all_probs, all_targets = [], []

    for X, y in loader:
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

    # Binary with BCELoss (we treat outputs as logits for consistency)
    if isinstance(loss_fn, nn.BCELoss):
        return torch.sigmoid(outputs.view(-1))         # (N,)

    # Multiclass CE
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return torch.softmax(outputs, dim=1)           # (N,C)
