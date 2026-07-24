#!/usr/bin/python
import time
from training.eval import evaluate_model, _compute_loss_and_accuracy
from training.wandb_utils import init_wandb_run, wandb_log, finish_wandb_run
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
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
                "lr": current_lr,
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

# Siamese / Contrastive Learning Training Loop
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


from info_nce import InfoNCE 
from training.wandb_utils import init_wandb_run, wandb_log, finish_wandb_run

def calculate_batch_accuracy(z1, z2):
    """
    Computes top-1 accuracy for the batch.
    For each sample in z1, checks if the corresponding sample in z2 
    is the closest neighbor compared to all other z2 samples.
    """
    # Normalize for cosine similarity
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute similarity matrix (Batch x Batch)
    logits = torch.matmul(z1, z2.T)
    
    # The correct match for z1[i] is z2[i] (the diagonal)
    batch_size = z1.size(0)
    targets = torch.arange(batch_size).to(z1.device)
    
    # Check if the highest score is on the diagonal
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float().sum()
    
    return correct, batch_size


def cross_view_info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Cross-view InfoNCE loss with self-comparison masking.

    Motivation
    ----------
    Standard InfoNCE(z1, z2, negative_mode='unpaired') treats z1 as queries and
    z2 as the key bank.  For query z1[i] the negatives are {z2[j] | j≠i} — the
    entire z1 side of the batch is never used as negatives, so same-side cross-file
    pairs (z1[i] ↔ z1[j]) contribute zero contrastive signal.

    Concretely, if file-1 yields views A, B and file-2 yields views C, D:
        standard loss uses     negatives: A↔D, B↔C            (diagonal only)
        cross-view loss uses   negatives: A↔C, A↔D, B↔C, B↔D  (all cross pairs)

    Construction
    ------------
    Concatenate both views into a single query/key bank:
        all_q = cat([z1, z2])   shape (2B, E)
        all_k = cat([z2, z1])   shape (2B, E)

    The (2B×2B) similarity matrix has positives on the main diagonal:
        • row i   (z1[i] query) → positive at col i   (z2[i] key)
        • row B+i (z2[i] query) → positive at col B+i (z1[i] key)

    Self-comparison mask
    --------------------
    all_q[i] also appears as a key at column (i+B) % 2B:
        • z1[i] query (row i)   maps to z1[i] key at column B+i
        • z2[i] query (row B+i) maps to z2[i] key at column i
    For L2-normalised embeddings sim(z, z)=1, so exp(1/τ) ≈ 148 at τ=0.2
    would inflate the denominator without contributing any learning signal
    (gradient of a constant w.r.t. θ is zero).  We mask these positions to
    -inf so they are excluded from the softmax denominator entirely.

    Parameters
    ----------
    z1, z2      : (B, E) embedding tensors from the two views
    temperature : InfoNCE temperature τ (default matches info-nce-pytorch)

    Returns
    -------
    Scalar loss tensor with gradients attached to z1 and z2.
    """
    B = z1.size(0)
    N = 2 * B

    # L2-normalise so cosine similarity == dot product (standard for InfoNCE)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

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

    # Positives are the diagonal; cross-entropy over 2B classes
    # label i means "the positive for query i is at column i" (the diagonal), it tells cross-entropy where to look for the positive in each row's logit vector.
    labels = torch.arange(N, device=z1.device)
    return F.cross_entropy(logits, labels)


def train_ssl_model(
    model,
    train_loader,
    val_loader=None,
    epochs=100,
    patience=20,
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
    
    # Initialize Optimizer
    optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
    
    # Initialize Loss.
    # Default: cross-view InfoNCE with self-comparison masking.
    # This replaces the previous InfoNCE(negative_mode='unpaired') default, which only
    # used z2[j≠i] as negatives for query z1[i], leaving all same-side cross-file pairs
    # (z1[i] ↔ z1[j]) unused.  The cross-view version captures every cross-file pair
    # as a negative — see cross_view_info_nce() for the full explanation.
    # Temperature 0.1 matches the info-nce-pytorch library default.
    if loss_fn is None:
        loss_fn = lambda q, k: cross_view_info_nce(q, k, temperature=0.1)
    
    # -------- initialize wandb (optional) --------
    run, start_time = None, None
    if wandb_logging:
        if wandb_config is None:
            raise ValueError("wandb_logging=True but no wandb_config provided.")
        run = init_wandb_run(wandb_config)
        start_time = time.time()
    # --------------------------------------------

    if verbose:
        print("Starting SSL training...")

    best_val_acc = -1.0  # We will use Contrastive Accuracy for early stopping
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        # ======== TRAIN LOOP (X1, X2, _) ========
        for batch in train_loader:
            # We unpack 3 items but IGNORE the label 'y'
            if len(batch) != 3:
                raise ValueError("Loader must yield (X1, X2, y).")
            
            X1_batch, X2_batch, _ = batch 
            X1_batch = X1_batch.to(device)
            X2_batch = X2_batch.to(device)

            optimizer.zero_grad()
            
            # Forward Pass: Get Embeddings/Projections
            z1, z2 = model(X1_batch, X2_batch) 

            # Compute Loss (InfoNCE takes query, positive)
            loss = loss_fn(z1, z2)

            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Track Metrics
            batch_size = X1_batch.size(0)
            total_loss += loss.item() * batch_size
            
            # Compute Batch Accuracy (Did we match the right pair?)
            with torch.no_grad():
                correct, count = calculate_batch_accuracy(z1, z2)
                total_correct += correct.item()
                total_samples += count

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ========= VALIDATION (optional) =========
        val_loss, val_acc = (None, None)
        if val_loader is not None:
            model.eval()
            val_total_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    X1_batch, X2_batch, _ = batch
                    X1_batch = X1_batch.to(device)
                    X2_batch = X2_batch.to(device)

                    z1, z2 = model(X1_batch, X2_batch)
                    
                    vloss = loss_fn(z1, z2)
                    vcorrect, vcount = calculate_batch_accuracy(z1, z2)

                    val_total_loss += vloss.item() * vcount
                    val_correct += vcorrect.item()
                    val_total += vcount

            val_loss = val_total_loss / val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # ===== Early stopping based on Contrastive Accuracy =====
            # TODO: Consider using SimCLR + kNN downstream evaluation for early stopping instead of raw contrastive accuracy, as the latter may not always correlate with downstream task performance.
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
                break
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        # ---- LR scheduler step ----
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use Validation Loss or Accuracy for scheduler
                metric = val_loss if val_loss is not None else train_loss
                scheduler.step(metric)
            else:
                scheduler.step()

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

            try:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            except Exception:
                pass

            wandb_log(run, log_dict)

        if verbose:
            msg = f"[SSL] Epoch [{epoch+1}/{epochs}] | train_loss {train_loss:.4f} | train_acc {train_acc:.4f}"
            if val_loader is not None and val_loss is not None:
                msg += f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
            print(msg)

    if wandb_logging and run is not None:
        finish_wandb_run(run, best_val_acc, start_time)

    print("SSL training complete.")
    return history


# Supervised-Contrastive (SupCon) Training Loop
def train_supcon_model(
    model,
    train_loader,
    val_loader=None,
    epochs=200,
    patience=20,
    lr=1e-3,
    optimizer=None,
    scheduler=None,
    loss_fn=None,
    augment_fn=None,
    n_views=2,
    encode_fn=None,
    device=None,
    grad_clip=1.0,
    save_path=None,
    eval_fn=None,
    eval_every=10,
    eval_metric_key=None,
    wandb_logging=False,
    wandb_config=None,
    verbose=True,
):
    """SupCon training loop: batches are (X, y) single cells/trajectories.

    Each step builds `n_views` independently-augmented views of X via
    `augment_fn`, encodes each via `encode_fn`, concatenates into
    (n_views*B, D) features / (n_views*B,) labels, and scores them with a
    metric-learning `loss_fn(feats, labels)` (e.g. pytorch-metric-learning's
    SupConLoss). 
    
    `eval_fn`: if given, is a caller-supplied downstream probe
    (e.g. SVM/KNN readout) invoked every `eval_every` epochs; its returned
    dict drives both wandb logging (under "downstream/") and, via
    `eval_metric_key`, checkpoint selection -- mirroring how `train_model`
    selects on val_acc, except the selection metric here is domain-specific
    and supplied by the caller rather than computed internally.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optimizer or optim.AdamW(model.parameters(), lr=lr)
    if loss_fn is None:
        raise ValueError("train_supcon_model requires an explicit loss_fn, e.g. SupConLoss(...).")
    if augment_fn is None:
        raise ValueError("train_supcon_model requires an explicit augment_fn producing SupCon views.")
    encode_fn = encode_fn or (lambda m, x: m.projection_head(m.backbone.encode(x)))

    if eval_fn is not None and save_path is not None and eval_metric_key is None:
        raise ValueError("eval_metric_key is required when eval_fn + save_path are both given.")

    # -------- initialize wandb (optional) --------
    run, start_time = None, None
    if wandb_logging:
        if wandb_config is None:
            raise ValueError("wandb_logging=True but no wandb_config provided.")
        run = init_wandb_run(wandb_config)
        start_time = time.time()
    # --------------------------------------------

    def _supcon_forward(xb, yb):
        """n_views augmented views of one cell batch -> concatenated feats/labels -> loss."""
        # data augmentation: n_views independent augmentations of the same batch
        views = [augment_fn(xb) for _ in range(n_views)]
        # encode each view into a feature vector
        zs = [encode_fn(model, v) for v in views]
        feats = torch.cat(zs, dim=0)
        # supcon is supervised so we need to repeat the labels for each view
        labels = torch.cat([yb] * n_views, dim=0)
        return loss_fn(feats, labels)

    if verbose:
        print("Starting SupCon training...")

    best_eval_value = -float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "eval_epoch": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            loss = _supcon_forward(xb, yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            # batch total loss
            running += loss.item() * len(yb)
            n_seen += len(yb)
        # summing across batches and dividing by the total sample count (n_seen) gives a correct per-sample average for the epoch — rather than just averaging the per-batch means, which would be slightly wrong if the last batch is a different size e.g., if drop_last = False.
        train_loss = running / n_seen
        current_lr = optimizer.param_groups[0]["lr"]

        # ========= Validation (monitoring only -- no early stopping unless patience is set) =========
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_running, v_seen = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vloss = _supcon_forward(xb, yb)
                    v_running += vloss.item() * len(yb)
                    v_seen += len(yb)
            val_loss = v_running / v_seen
            model.train()

        # ---- LR scheduler step ----
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()
        # ----------------------------

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # ========= Periodic downstream eval + checkpoint selection =========
        eval_metrics = None
        if eval_fn is not None and (epoch % eval_every == 0 or epoch == epochs):
            eval_metrics = eval_fn(model)
            history["eval_epoch"].append(epoch)
            for k, v in eval_metrics.items():
                history.setdefault(f"eval/{k}", []).append(v)

            # Checkpoint selection based on eval_metric_key
            if eval_metric_key is not None:
                if eval_metric_key not in eval_metrics:
                    raise ValueError(f"eval_metric_key '{eval_metric_key}' not found in eval_metrics: {list(eval_metrics.keys())}")
                current_value = eval_metrics[eval_metric_key]
                if current_value > best_eval_value:
                    best_eval_value = current_value
                    epochs_no_improve = 0
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                        if verbose:
                            print(f"✅ Model saved at {save_path} (best {eval_metric_key}: {best_eval_value:.4f})")
                else:
                    epochs_no_improve += 1
                    if verbose:
                        print(f"No improvement in {eval_metric_key} ({epochs_no_improve}/{patience}).")
                
                # ==== Early stopping based on (downstream) eval_metric_key ====
                if patience is not None and epochs_no_improve >= patience:
                    if verbose:
                        print("🛑 Early stopping.")
                    break
                
        # -------- wandb logging (all log_dict construction lives here) --------
        if wandb_logging and run is not None:
            log_dict = {"epoch": epoch, "train/loss": float(train_loss), "lr": current_lr}
            if val_loss is not None:
                log_dict["val/loss"] = float(val_loss)
            if eval_metrics is not None:
                log_dict.update({f"downstream/{k}": float(v) for k, v in eval_metrics.items()})
            wandb_log(run, log_dict)

        if verbose:
            msg = f"[SupCon] Epoch [{epoch}/{epochs}] | train_loss {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val_loss {val_loss:.4f}"
            if eval_metrics is not None:
                msg += " | " + " | ".join(f"{k} {v:.4f}" for k, v in eval_metrics.items())
            print(msg)

    if wandb_logging and run is not None:
        finish_wandb_run(run, best_eval_value if eval_fn is not None else None, start_time)

    print("SupCon training complete.")
    return history