# models/ARIMA.py
# Per-sequence ARIMA baseline using statsmodels.
# NOTE: This runs on CPU and fits (p,d,q) for each series in the batch.
# Heavy but fine as a classical baseline and for small L (e.g., 10/100).

import torch
import torch.nn as nn

class ARIMAModel(nn.Module):
    """
    ARIMA(p,d,q) baseline.
    Input:  (B, L, 1)  previous window
    Output: (B, L, 1)  predicted next window
    """
    def __init__(self, p: int = 1, d: int = 1, q: int = 0, enforce_stationarity: bool = True, enforce_invertibility: bool = True):
        super().__init__()
        self.p, self.d, self.q = p, d, q
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # Lazy import to avoid hard dependency if user doesn't use ARIMA
        try:
            from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
        except Exception as e:
            raise ImportError("ARIMAModel requires statsmodels. Install via: pip install statsmodels") from e

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        from statsmodels.tsa.arima.model import ARIMA
        B, L, C = x.shape
        assert C == 1, "ARIMAModel expects input_size=1"

        preds = []
        for b in range(B):
            series = x[b, :, 0].detach().cpu().numpy()
            try:
                model = ARIMA(series, order=(self.p, self.d, self.q),
                              enforce_stationarity=self.enforce_stationarity,
                              enforce_invertibility=self.enforce_invertibility)
                fitted = model.fit(method_kwargs={"warn_convergence": False})
                # forecast next L steps
                y_next = fitted.forecast(steps=L)
            except Exception:
                # Fallback: naive copy of last value
                y_next = series[-1] * torch.ones(L).numpy()
            preds.append(torch.tensor(y_next, dtype=x.dtype))

        y_pred = torch.stack(preds, dim=0).unsqueeze(-1)  # (B, L, 1)
        return y_pred.to(x.device)
