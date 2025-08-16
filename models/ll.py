# models/ll.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence

class MultiTaskLL(nn.Module):
    """
    Fully-connected baseline for ST-GCN comparison.
    Accepts x with shape (N, C, T, V), flattens to (N, C*T*V), then MLP.
    Returns:
      {
        "rul": (N, 1)  in [0,1]  (Sigmoid),
        "status_logits": [ (N, Ci), ... ] raw logits (NO softmax)
      }
    """
    def __init__(
        self,
        in_channels: int = 24,
        T: int = 6,
        V: int = 17,
        status_classes: List[int] = (3, 4, 3, 4),
        shared_layers: Sequence[int] = (512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        in_features = in_channels * T * V

        # Shared MLP trunk
        layers = []
        prev = in_features
        for h in shared_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.shared = nn.Sequential(*layers)
        trunk_dim = prev  # last shared width

        # RUL head (mirrors stgcn head: hidden 256 + Sigmoid)
        self.rul_head = nn.Sequential(
            nn.Linear(trunk_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1), nn.Sigmoid()
        )

        # 4 status heads -> RAW LOGITS (no softmax)
        self.status_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, n_cls)
            ) for n_cls in status_classes
        ])

    def forward(self, x: torch.Tensor):
        # x: (N, C, T, V) -> (N, C*T*V)
        x = x.view(x.size(0), -1)
        z = self.shared(x)
        rul = self.rul_head(z)  # (N,1) in [0,1]
        status_logits = [head(z) for head in self.status_heads]  # raw logits
        return {"rul": rul, "status_logits": status_logits}
