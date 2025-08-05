import torch
import torch.nn as nn
class LSTMModel(nn.Module):
    """
    Sequence-to-sequence LSTM model for next-window prediction.
    Given a window (e.g., shape [batch, seq_len, n_features]), predicts the next window.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # predict entire feature(s) at each step

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        # Apply fc to every time step to predict full window (not just last step)
        out = self.fc(out)  # (batch, seq_len, input_size)
        return out