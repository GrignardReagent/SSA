#!/usr/bin/python
import time
from training.eval import evaluate_model, _compute_loss_and_accuracy
from training.wandb_utils import init_wandb_run, wandb_log, finish_wandb_run
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    patience=10,
    lr=1e-3,
    optimizer=None,
    scheduler=None, 
    loss_fn=None,
    device=None,
    grad_clip=1.0,
    save_path=None,
    wandb_logging=False,
    wandb_config=None,
    verbose=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    
    # -------- initialize wandb (optional) --------
    run, start_time = None, None
    if wandb_logging:
        if wandb_config is None:
            raise ValueError("wandb_logging=True but no wandb_config provided.")
        run = init_wandb_run(wandb_config)
        start_time = time.time()
    # --------------------------------------------

    if verbose:
        print("Starting training...")

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch in train_loader:
            if len(batch) == 2: # handle (X, y)
                X_batch, y_batch = batch
            
            elif len(batch) >= 3 and isinstance(batch[2], torch.Tensor): # handle (X, y, mask)
                X_batch, y_batch, mask = batch[0], batch[1], batch[2]
            
            elif len(batch) >=3: # handle (X1, X2, y) - pairwise
                X1_batch, X2_batch, y_batch = batch[0], batch[1], batch[2]
                X_batch = (X1_batch.to(device), X2_batch.to(device))
                y_batch = y_batch.to(device)
                
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss, preds, tgt = _compute_loss_and_accuracy(outputs, y_batch, loss_fn)
            
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # This rescales all gradients so their total norm never exceeds grad_clip
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # ========= Validation =========
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
        
        # ---- LR scheduler step ----
        if scheduler is not None:
            # schedulers that depend on val_loss (e.g., ReduceLROnPlateau)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        # ----------------------------

        
        # -------- wandb logging --------
        if wandb_logging and run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": float(train_loss),
                "train/acc": float(train_acc),
            }
            if val_loss is not None:
                log_dict["val/loss"] = float(val_loss)
                log_dict["val/acc"] = float(val_acc)

            # learning rate
            try:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            except Exception:
                pass

            # gradient norm
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            log_dict["grad/norm"] = total_grad_norm ** 0.5 if total_grad_norm > 0 else 0.0

            wandb_log(run, log_dict)
        # --------------------------------

        if verbose:
            msg = f"Epoch [{epoch+1}/{epochs}] | train_loss {train_loss:.4f} | train_acc {train_acc:.4f}"
            if val_loader is not None:
                msg += f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
            print(msg)
            
    # -------- finish wandb --------
    if wandb_logging and run is not None:
        finish_wandb_run(run, best_val_acc, start_time)
    # ------------------------------
   
            
    print("Training complete.")
    return history


def train_siamese_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    patience=10,
    lr=1e-3,
    optimizer=None,
    scheduler=None,
    loss_fn=None,
    device=None,
    grad_clip=1.0,
    save_path=None,
    wandb_logging=False,
    wandb_config=None,
    verbose=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()

    # -------- initialize wandb (optional) --------
    run, start_time = None, None
    if wandb_logging:
        if wandb_config is None:
            raise ValueError("wandb_logging=True but no wandb_config provided.")
        run = init_wandb_run(wandb_config)
        start_time = time.time()
    # --------------------------------------------

    if verbose:
        print("Starting siamese training...")

    best_val_acc = -1.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        # ======== TRAIN LOOP (X1, X2, y) ========
        for batch in train_loader:
            if len(batch) != 3:
                raise ValueError("Siamese train_loader must yield (X1, X2, y).")
            X1_batch, X2_batch, y_batch = batch
            X1_batch = X1_batch.to(device)
            X2_batch = X2_batch.to(device)
            y_batch  = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X1_batch, X2_batch)   # model must accept (X1, X2)

            loss, preds, tgt = _compute_loss_and_accuracy(outputs, y_batch, loss_fn)

            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_size = tgt.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == tgt).sum().item()
            total += batch_size

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ========= VALIDATION (optional) =========
        val_loss, val_acc = (None, None)
        if val_loader is not None:
            model.eval()
            val_total_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) != 3:
                        raise ValueError("Siamese val_loader must yield (X1, X2, y).")
                    X1_batch, X2_batch, y_batch = batch
                    X1_batch = X1_batch.to(device)
                    X2_batch = X2_batch.to(device)
                    y_batch  = y_batch.to(device)

                    outputs = model(X1_batch, X2_batch)
                    vloss, vpreds, vtgt = _compute_loss_and_accuracy(outputs, y_batch, loss_fn)

                    bs = vtgt.size(0)
                    val_total_loss += vloss.item() * bs
                    val_correct += (vpreds == vtgt).sum().item()
                    val_total += bs

            val_loss = val_total_loss / val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # ===== Early stopping =====
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"✅ Model saved at {save_path} (Best Val Acc: {best_val_acc:.4f})")
            else:
                epochs_no_improve += 1
                if verbose:
                    print(f"No improvement ({epochs_no_improve}/{patience}).")

            if epochs_no_improve >= patience:
                print("🛑 Early stopping.")
                # scheduler might still want a final step, but we bail out here
                break
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        # ---- LR scheduler step ----
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        # ----------------------------

        # -------- wandb logging --------
        if wandb_logging and run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": float(train_loss),
                "train/acc": float(train_acc),
            }
            if val_loss is not None:
                log_dict["val/loss"] = float(val_loss)
                log_dict["val/acc"] = float(val_acc)

            # learning rate
            try:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            except Exception:
                pass

            # gradient norm
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            log_dict["grad/norm"] = total_grad_norm ** 0.5 if total_grad_norm > 0 else 0.0

            wandb_log(run, log_dict)
        # --------------------------------

        if verbose:
            msg = f"[Siamese] Epoch [{epoch+1}/{epochs}] | train_loss {train_loss:.4f} | train_acc {train_acc:.4f}"
            if val_loader is not None and val_loss is not None:
                msg += f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
            print(msg)

        if val_loader is not None and epochs_no_improve >= patience:
            break

    # -------- finish wandb --------
    if wandb_logging and run is not None:
        finish_wandb_run(run, best_val_acc, start_time)
    # ------------------------------

    print("Siamese training complete.")
    return history
