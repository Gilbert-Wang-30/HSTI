# models/LinearInterp.py
# Predict next window by fitting y ~ a * t + b on the previous window (per sequence),
# then extrapolating to the next L time steps: t = L .. 2L-1

import torch
import torch.nn as nn

class LinearInterpModel(nn.Module):
    """
    Linear interpolation / extrapolation baseline.
    Input:  (B, L, 1)  previous window
    Output: (B, L, 1)  predicted next window
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        B, L, C = x.shape
        assert C == 1, "LinearInterpModel expects input_size=1"
        device = x.device
        # times 0..L-1 and  L..2L-1
        t_prev = torch.arange(L, dtype=x.dtype, device=device).view(1, L, 1)      # (1,L,1)
        t_next = torch.arange(L, 2*L, dtype=x.dtype, device=device).view(1, L, 1) # (1,L,1)

        y = x  # (B,L,1)
        # closed-form OLS for y = a*t + b on previous window
        t_mean = t_prev.mean(dim=1, keepdim=True)   # (1,1,1)
        y_mean = y.mean(dim=1, keepdim=True)        # (B,1,1)
        t_center = t_prev - t_mean                  # (1,L,1)
        y_center = y - y_mean                       # (B,L,1)

        # slope a = cov(t,y)/var(t)
        var_t = (t_center ** 2).sum(dim=1, keepdim=True) + 1e-12  # (1,1,1)
        cov_ty = (t_center * y_center).sum(dim=1, keepdim=True)   # (B,1,1)
        a = cov_ty / var_t                                        # (B,1,1)
        b = y_mean - a * t_mean                                   # (B,1,1)

        y_next = a * t_next + b                                   # (B,L,1)
        return y_next
