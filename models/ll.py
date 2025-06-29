# simple ll model for testing
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, 50)
        self.linear_out = nn.Linear(50, out_features)

    def forward(self, x):
        return self.linear(x)
    
