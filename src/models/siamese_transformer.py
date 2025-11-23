#!/usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import warnings
from models.transformer import TransformerClassifier

# Siamese transformer built on top of TransformerClassifier
class SiameseTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead=4,
        num_layers=2,
        dropout=1e-3,
        use_conv1d=False,
    ):
        """
        Siamese wrapper:
          - shared Transformer backbone → embeddings z1, z2  (each [B,D])
          - concatenation [z1, z2] → linear head → logits [B,1] (same/different)
        """
        super().__init__()
        # shared backbone (ignores its own head via .encode())
        self.backbone = TransformerClassifier(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=2,      # unused in siamese, but required by ctor
            dropout=dropout,
            use_conv1d=use_conv1d,
        )
        # We concatenate:
        # 1. z1 (Feature A)
        # 2. z2 (Feature B)
        # === Explicit Difference Feature ===
        # 3. |z1 - z2| (The distance/difference) <--- CRITICAL for convergence: tells model that the answer depends on how different features A and B are. 
        self.fc = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1) # Binary output
        )

    def forward(self, x1, x2, src_key_padding_mask1=None, src_key_padding_mask2=None):
        # Encode both inputs
        z1 = self.backbone.encode(x1, src_key_padding_mask=src_key_padding_mask1)
        z2 = self.backbone.encode(x2, src_key_padding_mask=src_key_padding_mask2)

        # Compute difference
        diff = torch.abs(z1 - z2)
        
        # Concatenate
        h = torch.cat([z1, z2, diff], dim=1)
        
        logits = self.fc(h)
        return logits