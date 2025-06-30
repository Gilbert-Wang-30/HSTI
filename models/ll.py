import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(LinearLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer to prevent overfitting
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, out_features)  # outputs 1 value (e.g. RUL)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # output shape: (batch_size,)
    
