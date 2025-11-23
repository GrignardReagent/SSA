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
        
    # encode() returns pooled embedding [B, D] (no final head)
    def encode(self, x, src_key_padding_mask=None):
        # x: [B,T,F]
        if self.stem is not None: # Conv1D stem
            x = x.permute(0, 2, 1)          # [B,F,T]
            x = self.stem(x)
            x = x.permute(0, 2, 1)          # [B,T,F]

        x = self.input_proj(x)               # [B,T,D] → unbounded
        x = self.pe(x)                       # add positions → unbounded
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # mask: [B,T] boolean True = ignore this position
        x = x.mean(dim=1)                    # mean pool over time → [B,D]
        return x

    def forward(self, x, src_key_padding_mask=None):
        # x: [B,T,F]
        if self.stem is not None: # Conv1D stem
            x = x.permute(0, 2, 1)          # [B,F,T]
            x = self.stem(x)
            x = x.permute(0, 2, 1)          # [B,T,F]

        x = self.input_proj(x)               # [B,T,D] → unbounded
        x = self.pe(x)                       # add positions → unbounded
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # mask: [B,T] boolean True = ignore this position
        # print("Encoder output shape:", x.shape)  # should be [B,T,D] → unbounded

        x = x.mean(dim=1)                    # mean pool over time → unbounded
        # x = self.dropout(x)
        out = self.head(x)                  # [B, num_classes] → unbounded
        return out