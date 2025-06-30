import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
    def __init__(self, in_features, status_classes):
        super(MultiTaskModel, self).__init__()
        # Shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, 500),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer to prevent overfitting
            nn.Linear(500, 250),
            nn.ReLU()
        )
        # RUL head
        self.rul_head = nn.Linear(250, 1)
        # Status classification heads (one linear layer per status)
        self.status_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(250, 300),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(300, 50),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(50, num_classes)
            ) for num_classes in status_classes
        ])
        # (Each status_head will output logits for that status's classes)



    def forward(self, x):
        features = self.shared_net(x)  # output shape: (batch_size, 50)
        rul_output = self.rul_head(features).squeeze(-1)  # RUL output shape: (batch,)
        # Compute logits for each status head
        status_logits = [head(features) for head in self.status_heads]  # list of tensors
        status_probs = [F.softmax(logit, dim=1) for logit in status_logits]
        return rul_output, status_logits, status_probs
