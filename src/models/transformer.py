#!/usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import warnings

# Suppress the nested tensor warning for norm_first=True
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")

class PositionalEncoding(nn.Module):
    """Fixed sin/cos positional encodings; buffer has shape [max_len, d_model]."""
    def __init__(self, d_model, max_seq_length=4096):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)      # even dims
        pe[:, 1::2] = torch.cos(position * div_term)      # odd dims
        self.register_buffer("pe", pe, persistent=False)   # not a parameter

    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        if T > self.pe.size(0):
            raise ValueError(f"seq_len {T} > max_seq_length {self.pe.size(0)}")
        pe = self.pe[:T].unsqueeze(0).to(x)               # match device & dtype
        return x + pe
    
class TransformerClassifier(nn.Module):
    def __init__(
        self, 
        input_size, 
        d_model, 
        nhead=4, 
        num_layers=2, 
        num_classes=2,
        dropout=1e-3, 
        use_conv1d=False
        ):
        super().__init__()
        self.use_conv1d = use_conv1d

        # Optional Conv1D stem
        if use_conv1d:
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.stem = None

        self.input_proj = nn.Linear(input_size, d_model)
        self.pe = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B,T,F]
        if self.stem is not None: # Conv1D stem
            x = x.permute(0, 2, 1)          # [B,F,T]
            x = self.stem(x)
            x = x.permute(0, 2, 1)          # [B,T,F]

        x = self.input_proj(x)               # [B,T,D]
        x = self.pe(x)                       # add positions
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # mask: [B,T] bool

        x = x.mean(dim=1)                    # mean pool over time
        # x = self.dropout(x)
        return self.head(x)                  # [B,C]

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