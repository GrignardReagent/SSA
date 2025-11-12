#!/usr/bin/python
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)

        # compute loss
        if loss_fn is not None:
            y_batch_mod = y_batch
            #TODO: adjust targets if BCE-type loss
            if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                y_batch_mod = y_batch_mod.float().unsqueeze(1) if y_batch_mod.dim() == 1 else y_batch_mod.float()
                if outputs.dim() == 2 and outputs.size(1) == 2 and y_batch_mod.size(1) == 1:
                    outputs = outputs[:, 1].unsqueeze(1)
            total_loss += loss_fn(outputs, y_batch_mod).item() * X_batch.size(0)

        # compute accuracy
        if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).long()
            tgt = y_batch.view(-1).long()
        else:
            preds = outputs.argmax(1)
            tgt = y_batch
        correct += (preds == tgt).sum().item()
        total += tgt.size(0)

    loss = (total_loss / total) if (loss_fn and total > 0) else None
    acc = correct / total

    if verbose:
        loss_str = f"{loss:.2f}" if loss is not None else "N/A"
        print(f"Test — loss: {loss_str} | acc: {acc:.2f}")
    return loss, acc