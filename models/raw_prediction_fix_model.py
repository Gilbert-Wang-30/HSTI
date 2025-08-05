import torch
import torch.nn as nn

class RawFix(nn.Module):
    def __init__(self, input_size, num_features):
        super(RawFix, self).__init__()
        # Per-feature correction network
        self.feature_fix = nn.Sequential(
            nn.Linear(input_size + 1, input_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size, input_size)
        )
        # Shared network to combine all per-feature corrections
        self.shared_tail = nn.Sequential(
            nn.Linear(num_features * input_size, input_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size * 2, input_size)
        )

        self.input_size = input_size
        self.num_features = num_features

    def forward(self, x, x_features, lr_features):
        """
        x: [batch_size, input_size] - raw LSTM outputs
        x_features: [batch_size, num_features] - LSTM predicted features
        lr_features: [batch_size, num_features] - LR predicted features
        """
        delta_features = lr_features - x_features  # [batch, num_features]
        batch_size = x.shape[0]
        fixes = []

        for i in range(self.num_features):
            delta_this = delta_features[:, i].unsqueeze(1)  # [batch, 1]
            mask = ~torch.isnan(delta_this)  # [batch, 1], True where not nan
            delta_this_clean = torch.where(mask, delta_this, torch.zeros_like(delta_this))  # nan->0
            x_with_feature = torch.cat([x, delta_this_clean], dim=1)  # [batch, input_size+1]
            fix = self.feature_fix(x_with_feature)  # [batch, input_size]
            # Now mask: set fix to zero where LR is nan
            fix = fix * mask  # if mask is 0, fix is zeroed out
            fixes.append(fix.unsqueeze(1))  # [batch, 1, input_size]

        fixes = torch.cat(fixes, dim=1)  # [batch, num_features, input_size]
        fixes_flat = fixes.view(batch_size, -1)  # [batch, num_features*input_size]
        delta_x = self.shared_tail(fixes_flat)  # [batch, input_size]
        y = x + delta_x  # [batch, input_size]
        return y
