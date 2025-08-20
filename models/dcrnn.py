# models/dcrnn.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _to_device_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.to(device=ref.device, dtype=ref.dtype)


def _row_normalize(A: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize adjacency: P = D^{-1} A  (random-walk normalization).
    A: (V,V) nonnegative
    """
    with torch.no_grad():
        d = A.sum(dim=1).clamp(min=1e-8)
        Dinv = torch.diag(1.0 / d)
    return Dinv @ A


def _prepare_supports(A: Optional[torch.Tensor], V: int) -> List[torch.Tensor]:
    """
    Build diffusion supports [P_forward, P_backward] where P = D^{-1}A.
    If A is None, default to identity (no spatial mixing).
    Accepts A with shape (V,V) or (K,V,V); if K>1, partitions are summed.
    Returns list of (V,V) tensors.
    """
    if A is None:
        I = torch.eye(V)
        return [I, I]  # forward/backward same (no-op)
    if A.dim() == 3:
        A_agg = A.sum(dim=0)  # (V,V)
    else:
        A_agg = A
    A_agg = A_agg.clamp(min=0)

    P_f = _row_normalize(A_agg)         # forward
    P_b = _row_normalize(A_agg.t())     # backward
    return [P_f, P_b]


class DiffusionConv(nn.Module):
    """
    Diffusion convolution over graph:
      H = concat( X, S1 X, S1^2 X, ..., S_m^K X ) * W   (per-node linear)
    where Si are supports (e.g., forward/backward random-walk matrices).
    X: (N, V, C_in)  ->  H: (N, V, C_out)
    """
    def __init__(self, c_in: int, c_out: int, K: int, supports: List[torch.Tensor]):
        super().__init__()
        assert K >= 0
        self.K = K
        self.num_sup = len(supports)
        self.register_buffer("supports_0", supports[0].clone())
        self.register_buffer("supports_1", supports[1].clone() if len(supports) > 1 else supports[0].clone())
        # total stacks = 1 (X) + K * num_sup
        self.total_stacks = 1 + K * self.num_sup
        self.linear = nn.Linear(c_in * self.total_stacks, c_out, bias=False)

    def _apply_one_support(self, S: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # S: (V,V), X: (N,V,C) -> (N,V,C)
        return torch.einsum('vw,nwc->nvc', S, X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, V, C_in)
        """
        N, V, C = X.shape
        stacks = [X]  # k=0 identity term

        if self.K > 0:
            S_list = [self.supports_0, self.supports_1]
            for S in S_list[:self.num_sup]:
                Xk = X
                for _ in range(self.K):
                    Xk = self._apply_one_support(S, Xk)
                    stacks.append(Xk)

        H = torch.cat(stacks, dim=-1)  # (N,V,C * total_stacks)
        H = self.linear(H)             # per-node linear
        return H


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.
    h_t = (1 - z) ⊙ h_{t-1} + z ⊙ tanh( DConv([x_t, r ⊙ h_{t-1}]) )
    Gates z,r computed via diffusion conv on [x_t, h_{t-1}].
    """
    def __init__(self, c_in: int, c_hidden: int, K: int, supports: List[torch.Tensor], dropout: float = 0.0):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # gates: concat(x, h) -> 2*c_hidden
        self.dconv_gates = DiffusionConv(c_in + c_hidden, 2 * c_hidden, K, supports)
        # candidate: concat(x, r ⊙ h) -> c_hidden
        self.dconv_cand  = DiffusionConv(c_in + c_hidden, c_hidden, K, supports)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_t:   (N, V, c_in)
        h_prev:(N, V, c_hidden)
        """
        inp = torch.cat([x_t, h_prev], dim=-1)  # (N,V,c_in+c_hidden)
        gates = self.dconv_gates(inp)           # (N,V,2*c_hidden)
        z_t, r_t = gates.chunk(2, dim=-1)
        z_t = torch.sigmoid(z_t)
        r_t = torch.sigmoid(r_t)

        cand_inp = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_tilde = torch.tanh(self.dconv_cand(cand_inp))
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return self.dropout(h_t)


class DCRNNLayer(nn.Module):
    """
    One DCRNN layer unrolled over T steps.
    """
    def __init__(self, c_in: int, c_hidden: int, K: int, supports: List[torch.Tensor], dropout: float = 0.0):
        super().__init__()
        self.cell = DCGRUCell(c_in, c_hidden, K, supports, dropout)

    def forward(self, X: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        X: (N, T, V, C_in)
        h0: (N, V, c_hidden) or None
        Returns last hidden: (N, V, c_hidden)
        """
        N, T, V, _ = X.shape
        if h0 is None:
            h_t = X.new_zeros((N, V, self.cell.c_hidden))
        else:
            h_t = h0
        for t in range(T):
            h_t = self.cell(X[:, t, :, :], h_t)
        return h_t  # last hidden


class DCRNNBackbone(nn.Module):
    """
    DCRNN backbone that matches the training script interface.
    Expects:
      A: adjacency (V,V) or (K,V,V); if None, uses identity (no spatial mixing)
      x: (N, C, T, V)
    Returns:
      {"rul": (N,1), "status_logits": [ (N,num_cls_i), ... ]}
    """
    def __init__(self,
                 A: Optional[torch.Tensor],
                 in_channels: int,
                 T: int,
                 V: int,
                 status_classes: List[int],
                 hidden: int = 64,
                 num_layers: int = 2,
                 k: int = 1,
                 dropout: float = 0.1,
                 head_hidden: int = 256,
                 head_dropout: float = 0.2):
        super().__init__()
        self.T = T
        self.V = V
        # supports (forward/backward random-walk)
        supports = _prepare_supports(A, V)
        # register as buffers to move w/ model.to(device)
        self.register_buffer("support_f", supports[0])
        self.register_buffer("support_b", supports[1])

        # project per-node input channels to hidden for first layer (optional).
        # We can embed via a per-node linear before recurrent stack:
        self.in_proj = nn.Linear(in_channels, hidden, bias=False)

        layers = []
        c_in = hidden
        for _ in range(num_layers):
            layers.append(DCRNNLayer(c_in=c_in, c_hidden=hidden, K=k, supports=[self.support_f, self.support_b],
                                     dropout=dropout))
            c_in = hidden
        self.layers = nn.ModuleList(layers)

        # Heads (same style as ST-GCN)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1), nn.Sigmoid()
        )
        self.status_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_cls)
            ) for n_cls in status_classes
        ])

        # Optional norm for stability
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, T, V)
        """
        N, C, T, V = x.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"
        assert V == self.V, f"Expected V={self.V}, got {V}"

        # (N, C, T, V) -> (N, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # per-node input projection
        x = self.in_proj(x)  # (N, T, V, hidden)

        # unroll through stacked DCGRU layers
        h = None
        for layer in self.layers:
            h = layer(x, h0=h)  # (N, V, hidden)
            # you can feed only hidden to next layer (standard stacked RNN)
            x = h.unsqueeze(1).repeat(1, self.T, 1, 1)  # broadcast over T for next layer

        h = self.node_norm(h)           # (N, V, hidden)
        g = h.mean(dim=1)               # global node pooling -> (N, hidden)

        rul = self.rul_head(g)          # (N,1)
        status_logits = [head(g) for head in self.status_heads]
        return {"rul": rul, "status_logits": status_logits}
