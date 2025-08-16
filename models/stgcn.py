#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal, runnable ST-GCN in PyTorch with detailed comments.

What you get:
- A small but faithful ST-GCN stack (spatial graph conv -> BN -> temporal conv -> ReLU)
- Clear shape annotations at each step
- A toy adjacency with K=3 partitions (self, inward, outward) for quick testing
- A main() that forwards random data and prints output shapes

Input format:
    X: (N, C_in, T, V)
        N: batch size
        C_in: channels per node (e.g., 3 for [x, y, confidence])
        T: temporal length (frames)
        V: number of nodes per frame (e.g., skeleton joints)

Adjacency:
    A: (K, V, V)
        K: number of partitions (e.g., 3: self, inward, outward)
"""

from typing import Sequence, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric degree normalization per partition:
        \hat{A}_k = D^{-1/2} A_k D^{-1/2}
    so nodes with many neighbors don't dominate the sum.

    Args:
        A: (K, V, V) raw/partitioned adjacency (float tensor)

    Returns:
        A_norm: (K, V, V) normalized adjacency
    """
    assert A.dim() == 3, "A must be (K, V, V)"
    K, V, V2 = A.shape
    assert V == V2, "Adjacency must be square per partition"

    A_norm = torch.empty_like(A)
    for k in range(K):
        Ak = A[k]                              # (V, V)
        deg = Ak.sum(dim=1)                    # (V,) degree per node (row sum)
        deg = torch.clamp(deg, min=1e-6)       # avoid divide-by-zero
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))  # (V, V)
        A_norm[k] = D_inv_sqrt @ Ak @ D_inv_sqrt
    return A_norm


# --------------------------------------------------------------------------------------
# Core layers
# --------------------------------------------------------------------------------------

class SpatialGraphConv(nn.Module):
    """
    Spatial graph "convolution" = channel-mixing (1x1) + neighbor aggregation via A.

    Given:
        x: (N, C_in, T, V)

    Steps:
        1) Channel mix with 1x1 conv to produce K*C_out channels (K partitions)
        2) Split along channels into K tensors z_k, each (N, C_out, T, V)
        3) For each k, aggregate neighbor info:
               y_k(n, c, t, v_out) = sum_{v_in} z_k(n, c, t, v_in) * A_k[v_in, v_out]
           Implemented as einsum over node dimension.
        4) Sum over k partitions.

    Output:
        y: (N, C_out, T, V)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 K: int,
                 bias: bool = True,
                 edge_importance: bool = False):
        super().__init__()
        self.K = K

        # Produce K * C_out channels in one go, then split into K "heads"
        self.conv1x1 = nn.Conv2d(in_channels, out_channels * K, kernel_size=1, bias=bias)

        # Optional scalar reweighting per partition (learnable)
        self.edge_importance = nn.Parameter(torch.ones(K, 1, 1)) if edge_importance else None

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T, V)
            A: (K, V, V) normalized adjacency per partition

        Returns:
            (N, C_out, T, V)
        """
        N, _, T, V = x.shape
        # 1) Channel mix: (N, C_in, T, V) -> (N, K*C_out, T, V)
        z = self.conv1x1(x)

        # 2) Reshape to split K partitions cleanly:
        #    (N, K*C_out, T, V) -> (N, K, C_out, T, V)
        z = z.view(N, self.K, -1, T, V)

        # 3) Aggregate neighbors for each partition, then sum
        y = None
        for k in range(self.K):
            zk = z[:, k, ...]             # (N, C_out, T, V)
            Ak = A[k]                     # (V, V)
            if self.edge_importance is not None:
                Ak = Ak * self.edge_importance[k]  # scale the whole partition

            # Neighbor aggregation over V:
            #   (n,c,t,v) x (v,w) -> (n,c,t,w)
            yk = torch.einsum('nctv,vw->nctw', zk, Ak)

            y = yk if y is None else (y + yk)  # sum over partitions

        # y: (N, C_out, T, V)
        return y


class TemporalConv(nn.Module):
    """
    Temporal-only convolution:
        Conv2d with kernel=(k_t, 1), stride=(s_t, 1), no padding; T_out = T_in - (k_t-1).
    This models motion patterns along time for each node independently.

    Input/Output shape:
        x in:  (N, C, T, V)
        x out: (N, C, T', V)  where T' depends on stride along time
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int = 9,
                 stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=(kernel_size, 1),
                              stride=(stride, 1),
                              padding=(0, 0),
                              bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)    # (N, C, T', V)
        x = self.bn(x)
        x = self.drop(x)
        return x


class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
        SpatialGraphConv -> BN -> TemporalConv -> ReLU
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,                   # (K, V, V), will be stored normalized
                 temporal_kernel: int = 9,
                 temporal_stride: int = 1,
                 dropout: float = 0.0,
                 edge_importance: bool = False):
        super().__init__()

        # Store normalized adjacency as a non-trainable buffer
        A_norm = normalize_adjacency(A.detach().clone())
        self.register_buffer('A', A_norm)         # (K, V, V)

        K, V, _ = A.shape
        self.spatial = SpatialGraphConv(in_channels, out_channels, K, edge_importance=edge_importance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.temporal = TemporalConv(out_channels,
                                     kernel_size=temporal_kernel,
                                     stride=temporal_stride,
                                     dropout=dropout)



        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C_in, T, V)
        returns: (N, C_out, T', V)
        """
        # Spatial aggregation over the graph
        y = self.spatial(x, self.A)   # (N, C_out, T, V)
        y = self.bn(y)

        # Temporal modeling along T (may change T via stride)
        y = self.temporal(y)          # (N, C_out, T', V)

        y = self.act(y)
        return y


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------

class STGCN(nn.Module):
    """
    A compact ST-GCN stack with global average pooling and a classifier head.

    Args:
        num_class: number of output classes
        A: (K, V, V) adjacency (partitioned); will be normalized internally in each block
        in_channels: input channels per node (e.g., 3 for [x, y, confidence])
        channels: tuple of channels per block (depth = len(channels))
        temporal_kernel: temporal conv kernel size (odd; e.g., 9)
        dropout: dropout used in temporal conv blocks
        edge_importance: whether to learn a scalar per adjacency partition
    """
    def __init__(self,
                 A: torch.Tensor,
                 status_classes: List[int],
                 in_channels: int = 24,
                 channels: Sequence[int] = (64, 32, 10),
                 temporal_kernel: Sequence[int] = (3,3,2),
                 dropout: float = 0.0,
                 edge_importance: bool = False):
        super().__init__()

        # Optional input BN across (C_in * V) at each time step (common in ST-GCN repos)
        V = A.shape[1]
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # Build backbone blocks
        blocks = []
        c_prev = in_channels
        for i, c_out in enumerate(channels):
            stride = 1
            blocks.append(
                STGCNBlock(c_prev, c_out, A,
                           temporal_kernel=temporal_kernel[i],
                           temporal_stride=stride,
                           dropout=dropout,
                           edge_importance=edge_importance)
            )
            c_prev = c_out
        self.backbone = nn.Sequential(*blocks)

        self.attn_norm = nn.LayerNorm(c_prev)
        self.attention = nn.MultiheadAttention(embed_dim=c_prev, num_heads=1, dropout=0.0 , batch_first=True)
        # self attention result shape (N, V, C_last/C_prev)
        
        hidden = 256
        self.rul_head = nn.Sequential(
            nn.Linear(c_prev, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # RUL output in [0, 1]
        )
        self.status_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c_prev, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, n_cls)
            ) for n_cls in status_classes
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C_in, T, V)
        returns logits: (N, num_class)
        """
        N, C, T, V = x.shape
        assert (T == 6) # Ensure T is 6
        # Input BN across (C*V) per time step:
        #   (N, C, T, V) -> (N*T, C*V) -> BN -> back
        x = x.permute(0, 2, 1, 3).contiguous()   # (N, T, C, V)
        x = x.view(N * T, C * V)                 # (N*T, C*V)
        x = self.data_bn(x)
        x = x.view(N, T, C, V).permute(0, 2, 1, 3).contiguous()  # (N, C, T, V)

        # Spatiotemporal backbone
        x = self.backbone(x)             # (N, C_last, T', V)

        N, C_last, T_prime, V = x.shape
        assert (C_last == 10) and (T_prime == 1), f"Expected C_last==10 and T'==1, got C_last={C_last}, T'={T_prime}"

        # (N, C_last, T', V) -> (N, C_last, V)
        x = x.squeeze(2)


        # Attention expects (N, V, C_last) since batch_first=True
        x_nodes = x.permute(0, 2, 1).contiguous()          # (N, V, C_last)

        # stabilize attention
        x_nodes = self.attn_norm(x_nodes)                         # (N, V, C_last)
        x_nodes = torch.nan_to_num(x_nodes, nan=0.0, posinf=1e4, neginf=-1e4)

        # Self-attention: q = k = v
        x_attn, _= self.attention(x_nodes, x_nodes, x_nodes)  # x_attn: (N, V, C_last)

        # Keep naming consistent with the rest of your pipeline
        x = x_attn                                           # (N, V, C_last)
        

        g = x_attn.mean(dim=1)                        # (N, C_last), permutation-invariant readout
        rul = self.rul_head(g)                        # (N, 1)
        status_logits = [head(g) for head in self.status_heads]
        y = {"rul": rul, "status_logits": status_logits}
        return y


