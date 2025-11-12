#!/usr/bin/python
from training.eval import evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    patience=10,
    lr=1e-2,
    optimizer=None,
    loss_fn=None,
    device=None,
    grad_clip=1.0,
    save_path=None,
    verbose=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or nn.CrossEntropyLoss()

    if verbose:
        print("Starting training...")

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            #TODO: adjust targets if BCE-type loss
            y_batch_mod = y_batch
            if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                y_batch_mod = y_batch_mod.float().unsqueeze(1) if y_batch_mod.dim() == 1 else y_batch_mod.float()
                if outputs.dim() == 2 and outputs.size(1) == 2 and y_batch_mod.size(1) == 1:
                    outputs = outputs[:, 1].unsqueeze(1)

            loss = loss_fn(outputs, y_batch_mod)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

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

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # Validation
        val_loss, val_acc = (None, None)             
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, loss_fn=loss_fn, device=device, verbose=False)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"✅ Model saved at {save_path} (Best Val Acc: {best_val_acc:.4f})")
            else:
                epochs_no_improve += 1
                print(f"No improvement ({epochs_no_improve}/{patience}).")
                
            if epochs_no_improve >= patience:
                print("🛑 Early stopping.")
                break

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            msg = f"Epoch [{epoch+1}/{epochs}] | train_loss {train_loss:.2f} | train_acc {train_acc:.2f}"
            if val_loader is not None:
                msg += f" | val_loss {val_loss:.2f} | val_acc {val_acc:.2f}"
            print(msg)
            
    print("Training complete.")
    return history