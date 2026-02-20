#!/usr/bin/python

import torch
import torch.nn as nn
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
        
# class SSL_Transformer(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         d_model,
#         nhead=4,
#         num_layers=2,
#         dropout=1e-3,
#         use_conv1d=False,
#     ):
#         """
#         SimCLR-style Transformer:
#           - Backbone: Extracts features (Sequence -> Sequence)
#           - Pooling: Aggregates time (Sequence -> Vector h)
#           - Projection Head: Maps for loss (Vector h -> Vector z)
#         """
#         super().__init__()
        
#         # 1. The Backbone (f)
#         # We use the encoder part of your existing classifier
#         self.backbone = TransformerClassifier(
#             input_size=input_size,
#             d_model=d_model,
#             nhead=nhead,
#             num_layers=num_layers,
#             num_classes=2, 
#             dropout=dropout,
#             use_conv1d=use_conv1d,
#         )
        
#         # 2. The Projection Head (g)
#         # Learnings from Chen et al. (2020):
#         # - Nonlinearity (ReLU) improves quality.
#         # - Batch Norm is beneficial in the head.
#         # - Output dim usually smaller or equal to d_model.
#         self.projection_head = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.BatchNorm1d(d_model), # <--- Added for SimCLR stability
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#             nn.BatchNorm1d(d_model)  # <--- Optional, but common in SimCLR v2
#         )

#     def pool(self, x):
#         """Global Average Pooling: [Batch, Time, Dim] -> [Batch, Dim]"""
#         # Assumes x is (Batch, Time, Feats)
#         return x.mean(dim=1)

#     def encode(self, x):
#         """
#         Returns the Representation 'h' (Backbone + Pooling).
#         Use THIS for Downstream Tasks (SVM, Classification, etc.).
#         Do NOT use the projection head output for evaluation.
#         """
#         # 1. Backbone (Sequence features)
#         features_seq = self.backbone.encode(x) 
        
#         # 2. Pooling (Vector representation h)
#         h = self.pool(features_seq)
#         return h

#     def forward(self, x1, x2):
#         """
#         Training Step: Returns projected vectors 'z' for the Loss Function.
#         """
#         # 1. Get Representations (h)
#         h1 = self.encode(x1)
#         h2 = self.encode(x2)
        
#         # 2. Project to Contrastive Space (z)
#         z1 = self.projection_head(h1)
#         z2 = self.projection_head(h2)
        
#         return z1, z2