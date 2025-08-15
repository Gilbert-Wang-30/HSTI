import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# FiLM building blocks
# -----------------------------
class FiLM1D(nn.Module):
    """
    FiLM modulation for 1D feature maps.
    condition: (B, cond_dim) -> per-channel gamma, beta
    x: (B, C, T) -> gamma*x + beta, where gamma,beta are (B, C, 1)
    """
    def __init__(self, cond_dim: int, channels: int, hidden: int = 64):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * channels)
        )
        # Init to be near identity modulation at start: gamma≈1, beta≈0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # params: (B, 2C) -> split -> (B, C) each
        params = self.net(condition)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta  = beta.unsqueeze(-1)   # (B, C, 1)
        # start near identity: add 1.0 to gamma at runtime
        return (1.0 + gamma) * x + beta


class FiLMConvBlock(nn.Module):
    """
    Conv1d -> Norm -> FiLM -> ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, cond_dim: int,
                 norm: str = "bn", padding: str = "same"):
        super().__init__()
        pad = (kernel_size // 2) if padding == "same" else 0
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm == "ln":
            self.norm = nn.GroupNorm(1, out_ch)  # LN proxy for 1D maps
        else:
            self.norm = nn.Identity()
        self.film = FiLM1D(cond_dim=cond_dim, channels=out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        return F.relu(x, inplace=True)


# -----------------------------
# Expert: FiLM-conditioned 1D CNN
# -----------------------------
class FiLM1DCNNExpert(nn.Module):
    """
    One expert that corrects the raw time-series so that feature_i moves toward LR target.
    Inputs:
      signal:   (B, 1, T)
      lstm_feat:(B, 1)  scalar
      lr_feat:  (B, 1)  scalar (may be NaN; you should prefilter or send mask=0)
      mask:     (B, 1)  0/1 availability (optional; default = 1)
    Output:
      corrected: (B, 1, T) = signal + residual
    """
    def __init__(self,
                 hidden_channels: int = 32,
                 n_blocks: int = 3,
                 kernel_size: int = 5,
                 cond_uses: str = "diff+mask",  # "diff", "diff+mask", or "raw+mask"
                 norm: str = "bn"):
        super().__init__()
        self.cond_uses = cond_uses

        # Determine conditioning vector dimensionality
        if cond_uses == "diff":
            cond_dim = 1
        elif cond_uses == "diff+mask":
            cond_dim = 2
        elif cond_uses == "raw+mask":
            cond_dim = 3  # lstm_feat, lr_feat, mask
        else:
            raise ValueError("cond_uses must be one of {'diff','diff+mask','raw+mask'}")

        blocks = []
        # First block: 1 -> hidden
        blocks.append(FiLMConvBlock(1, hidden_channels, kernel_size, cond_dim, norm=norm))
        # Middle blocks: hidden -> hidden
        for _ in range(n_blocks - 1):
            blocks.append(FiLMConvBlock(hidden_channels, hidden_channels, kernel_size, cond_dim, norm=norm))
        self.blocks = nn.ModuleList(blocks)

        # Final projection to residual Δ
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # Initialize head to zero so initial output ≈ identity (signal + 0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _build_condition(self, lstm_feat: torch.Tensor, lr_feat: torch.Tensor, mask: torch.Tensor):
        if mask is None:
            mask = torch.ones_like(lstm_feat)
        if self.cond_uses == "diff":
            cond = (lr_feat - lstm_feat)
        elif self.cond_uses == "diff+mask":
            cond = torch.cat([lr_feat - lstm_feat, mask], dim=1)
        else:  # "raw+mask"
            cond = torch.cat([lstm_feat, lr_feat, mask], dim=1)
        # Replace NaNs (if any slipped through) with zeros to avoid NaN propagation
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        return cond

    def forward(self,
                signal: torch.Tensor,     # (B,1,T)
                lstm_feat: torch.Tensor,  # (B,1)
                lr_feat: torch.Tensor,    # (B,1)
                mask: torch.Tensor = None # (B,1)
                ) -> torch.Tensor:
        cond = self._build_condition(lstm_feat, lr_feat, mask)
        x = signal
        for blk in self.blocks:
            x = blk(x, cond)
        residual = self.head(x)
        return signal + residual  # residual correction


# -----------------------------
# Bank of 14 experts (per frequency group)
# -----------------------------
class ExpertBank(nn.Module):
    """
    Holds 14 feature-specific experts.
    - forward_feature(...) routes to expert[feat_idx]
    - forward_all(...) runs all experts and returns a list/stack of corrected signals
    """
    def __init__(self, num_features: int = 14, **expert_kwargs):
        super().__init__()
        self.num_features = num_features
        self.experts = nn.ModuleList([
            FiLM1DCNNExpert(**expert_kwargs) for _ in range(num_features)
        ])

    @torch.no_grad()
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward_feature(self,
                        feat_idx: int,
                        signal: torch.Tensor,     # (B,1,T)
                        lstm_feat: torch.Tensor,  # (B,1)
                        lr_feat: torch.Tensor,    # (B,1)
                        mask: torch.Tensor = None # (B,1)
                        ) -> torch.Tensor:
        return self.experts[feat_idx](signal, lstm_feat, lr_feat, mask)

    def forward_all(self,
                    signal: torch.Tensor,        # (B,1,T)
                    lstm_feats: torch.Tensor,    # (B,14)
                    lr_feats: torch.Tensor,      # (B,14)
                    masks: torch.Tensor = None   # (B,14) optional
                    ):
        """
        Runs every expert i with its scalar pair (lstm_feats[:,i], lr_feats[:,i], masks[:,i])
        Returns:
          outputs: list of 14 tensors each (B,1,T) OR a single stacked tensor (B,14,T) if you prefer.
        """
        outs = []
        if masks is None:
            masks = torch.ones_like(lstm_feats)
        for i in range(self.num_features):
            y = self.experts[i](
                signal,
                lstm_feats[:, i:i+1],
                lr_feats[:, i:i+1],
                masks[:, i:i+1]
            )
            outs.append(y)
        # Example: return stacked corrections of shape (B, 14, T)
        return torch.stack([o.squeeze(1) for o in outs], dim=1)
