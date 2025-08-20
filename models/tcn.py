# models/tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List


class TemporalBlock(nn.Module):
    """
    A standard TCN block: Conv1d (dilated, same padding) -> BN -> ReLU -> Dropout -> Conv1d -> BN -> ReLU -> Dropout
    with a residual/skip connection (1x1 conv if channels change).
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        assert kernel_size >= 1
        assert not (kernel_size % 2 == 0 and not causal), \
            "Use odd kernel_size for non-causal SAME padding."

        self.causal = causal
        self.k = kernel_size
        self.d = dilation

        # We'll set padding inside forward (for causal we need left-only)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=0, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else None

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        # causal: left-pad only by (k-1)*d; non-causal: symmetric SAME padding
        if self.causal:
            pad = (self.k - 1) * self.d
            return F.pad(x, (pad, 0))
        else:
            pad = (self.k - 1) * self.d // 2
            return F.pad(x, (pad, pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self._pad(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        y = self.dropout1(y)

        y = self._pad(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            residual = self.downsample(residual)

        y = F.relu(y + residual, inplace=True)
        y = self.dropout2(y)
        return y


class TCNBackbone(nn.Module):
    """
    TCN over time for multivariate sequences (NO graph).
    Expects x: (N, C, T, V). Internally flattens nodes => (N, C*V, T), applies TCN blocks,
    pools over time, then uses the same multi-task heads as your ST-GCN.

    Returns:
      {"rul": (N,1), "status_logits": [ (N,num_cls_i), ... ]}
    """
    def __init__(self,
                 status_classes: List[int],
                 in_channels: int = 24,
                 V: int = 17,
                 tcn_channels: Sequence[int] = (64, 32, 10),
                 kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2, 4),
                 dropout: float = 0.1,
                 causal: bool = False,
                 head_hidden: int = 256,
                 head_dropout: float = 0.2,
                 pool: str = "mean"):  # "mean" or "last"
        super().__init__()

        assert len(tcn_channels) >= 1
        assert len(dilations) == len(tcn_channels), "dilations must match tcn_channels length."

        self.V = V
        self.pool = pool

        in_ch = in_channels * V  # flatten nodes -> treat as feature channels
        self.input_bn = nn.BatchNorm1d(in_ch)

        blocks = []
        c_prev = in_ch
        for c_out, d in zip(tcn_channels, dilations):
            blocks.append(TemporalBlock(c_prev, c_out,
                                        kernel_size=kernel_size,
                                        dilation=d,
                                        dropout=dropout,
                                        causal=causal))
            c_prev = c_out
        self.tcn = nn.Sequential(*blocks)

        # Heads (same spirit as your ST-GCN heads)
        last_feat = tcn_channels[-1]
        self.rul_head = nn.Sequential(
            nn.Linear(last_feat, head_hidden), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid()  # keep your [0,1] RUL
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
        assert V == self.V, f"Got V={V}, expected V={self.V}"

        # Flatten nodes; keep time as Conv1d length
        x = x.view(N, C * V, T)             # (N, C*V, T)
        x = self.input_bn(x)

        y = self.tcn(x)                      # (N, C_last, T)

        if self.pool == "mean":
            g = y.mean(dim=2)               # global average over time -> (N, C_last)
        elif self.pool == "last":
            g = y[:, :, -1]                 # last timestep
        else:
            raise ValueError("pool must be 'mean' or 'last'")

        rul = self.rul_head(g)              # (N, 1)
        status_logits = [head(g) for head in self.status_heads]
        return {"rul": rul, "status_logits": status_logits}
