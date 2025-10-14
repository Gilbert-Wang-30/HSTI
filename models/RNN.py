# models/RNN.py
# Vanilla RNN (tanh) + linear head, seq-to-seq

import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """
    Sequence-to-sequence vanilla RNN (tanh) baseline.
    Input:  (B, L, 1)
    Output: (B, L, 1)
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, nonlinearity: str = "tanh"):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          nonlinearity=nonlinearity, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        out, _ = self.rnn(x)     # (B, L, H)
        out = self.fc(out)       # (B, L, 1)
        return out
