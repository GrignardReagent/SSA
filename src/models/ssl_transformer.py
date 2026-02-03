#!/usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import warnings
from models.transformer import TransformerClassifier

# SSL_Transformer built on top of TransformerClassifier
class SSL_Transformer(nn.Module):
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
        SSL_Transformer wrapper:
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
        # Maps backbone features to the space where contrastive loss is applied
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x1, x2):
            # 1. Get Representations
            h1 = self.backbone.encode(x1)
            h2 = self.backbone.encode(x2)
            
            # 2. Project
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            
            # Return projected vectors! No logits, no difference calc.
            return z1, z2