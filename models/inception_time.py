# models/inception_time.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence


def _same_pad_1d(x: torch.Tensor, k: int, d: int = 1) -> torch.Tensor:
    # symmetric "same" padding for stride=1
    pad = ((k - 1) * d) // 2
    return F.pad(x, (pad, pad))


class InceptionBlock(nn.Module):
    """
    One InceptionTime block (non-causal):
      - optional 1x1 bottleneck
      - parallel Conv1d branches with different kernel sizes
      - maxpool + 1x1 branch
      - concat -> BN -> (optional residual) -> ReLU -> Dropout
    Keeps time length T (stride=1, SAME padding).
    """
    def __init__(
        self,
        in_ch: int,
        nb_filters: int,
        bottleneck_channels: int = 32,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.kernel_sizes = tuple(kernel_sizes)

        # Bottleneck 1x1 on input (only if in_ch > bottleneck)
        self.use_bottleneck = in_ch > bottleneck_channels
        bch = bottleneck_channels if self.use_bottleneck else in_ch
        self.bottleneck = nn.Conv1d(in_ch, bch, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()
        self.bottleneck_bn = nn.BatchNorm1d(bch) if self.use_bottleneck else nn.Identity()

        # Conv branches with different kernels
        self.br_convs = nn.ModuleList([
            nn.Conv1d(bch, nb_filters, kernel_size=k, padding=0, bias=False) for k in self.kernel_sizes
        ])
        self.br_bns = nn.ModuleList([nn.BatchNorm1d(nb_filters) for _ in self.kernel_sizes])

        # MaxPool branch (3) + 1x1 conv to nb_filters
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_ch, nb_filters, kernel_size=1, bias=False)
        self.pool_bn = nn.BatchNorm1d(nb_filters)

        # Residual path to match concat channels
        total_out = nb_filters * (len(self.kernel_sizes) + 1)
        self.res_proj = nn.Conv1d(in_ch, total_out, kernel_size=1, bias=False) if (use_residual and in_ch != total_out) else None
        self.res_bn = nn.BatchNorm1d(total_out) if (use_residual and in_ch != total_out) else nn.Identity()

        self.out_bn = nn.BatchNorm1d(total_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        inp = x

        # Bottleneck
        xb = self.bottleneck_bn(self.bottleneck(x)) if self.use_bottleneck else x

        # Conv branches (SAME)
        conv_outs = []
        for conv, bn, k in zip(self.br_convs, self.br_bns, self.kernel_sizes):
            y = conv(_same_pad_1d(xb, k))
            y = bn(y)
            y = F.relu(y, inplace=True)
            conv_outs.append(y)

        # MaxPool + 1x1 branch
        yp = self.pool_conv(self.maxpool(inp))
        yp = self.pool_bn(yp)
        yp = F.relu(yp, inplace=True)
        conv_outs.append(yp)

        y = torch.cat(conv_outs, dim=1)   # (N, total_out, T)
        y = self.out_bn(y)

        if self.use_residual:
            res = inp
            if self.res_proj is not None:
                res = self.res_bn(self.res_proj(res))
            y = y + res

        y = F.relu(y, inplace=True)
        y = self.drop(y)
        return y  # (N, total_out, T)


class InceptionTimeBackbone(nn.Module):
    """
    InceptionTime backbone adapted for multivariate sensor windows:
      - expects x: (N, C, T, V)
      - flattens nodes -> (N, C*V, T)
      - stacks Inception blocks
      - global average over time
      - multi-task heads:
          rul: Sigmoid in [0,1]  (use MSE)
          status_logits: list of raw logits (use CE with label smoothing)
    """
    def __init__(
        self,
        in_channels: int = 24,
        V: int = 17,
        T: int = 6,
        status_classes: List[int] = (3, 4, 3, 4),
        nb_filters: int = 32,
        depth: int = 6,
        bottleneck_channels: int = 32,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
        use_residual: bool = True,
        head_hidden: int = 256,
        head_dropout: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__()
        assert depth >= 1
        assert pool in ("mean", "last")
        self.V = V
        self.T = T
        self.pool = pool

        in_ch = in_channels * V
        self.input_bn = nn.BatchNorm1d(in_ch)

        blocks = []
        c_prev = in_ch
        for _ in range(depth):
            block = InceptionBlock(
                in_ch=c_prev,
                nb_filters=nb_filters,
                bottleneck_channels=bottleneck_channels,
                kernel_sizes=kernel_sizes,
                use_residual=use_residual,
                dropout=dropout,
            )
            blocks.append(block)
            c_prev = nb_filters * (len(kernel_sizes) + 1)
        self.backbone = nn.Sequential(*blocks)

        last_feat = c_prev
        # Heads (same style as your ST-GCN/TCN/DCRNN)
        self.rul_head = nn.Sequential(
            nn.Linear(last_feat, head_hidden), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1), nn.Sigmoid()
        )
        self.status_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(last_feat, head_hidden), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_cls)
            ) for n_cls in status_classes
        ])

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, T, V)
        """
        N, C, T, V = x.shape
        assert V == self.V, f"Expected V={self.V}, got {V}"
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # Flatten nodes; treat (C*V) as channels for temporal convs
        x = x.view(N, C * V, T)          # (N, C*V, T)
        x = self.input_bn(x)

        y = x
        y = self.backbone(y)             # (N, F, T)

        if self.pool == "mean":
            g = y.mean(dim=2)            # (N, F)
        else:
            g = y[:, :, -1]              # (N, F)

        rul = self.rul_head(g)           # (N, 1)
        status_logits = [head(g) for head in self.status_heads]  # list[(N,num_cls)]
        return {"rul": rul, "status_logits": status_logits}
