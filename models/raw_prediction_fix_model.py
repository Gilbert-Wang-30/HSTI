import torch
import torch.nn as nn

class RawFix(nn.Module):
    def __init__(self, input_size, num_features):
        super(RawFix, self).__init__()
        # Per-feature correction network
        self.feature_fix = nn.Sequential(
            nn.Linear(input_size + 1, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, input_size),
            nn.ReLU()
        )
        # Shared network to combine all per-feature corrections
        self.shared_tail = nn.Sequential(
            nn.Linear(num_features * input_size, input_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU()
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

        # For each feature, create a fix
        batch_size = x.shape[0]
        fixes = []
        for i in range(self.num_features):
            # Each fix uses the raw input and the delta for this feature
            delta_this = delta_features[:, i].unsqueeze(1)  # [batch, 1]
            # if delta_this is NaN, we skip this feature
            # Replace NaNs in delta_this with 0 (per-sample, not whole batch)
            delta_this = torch.where(
                torch.isnan(delta_this),
                torch.zeros_like(delta_this),  # replace NaN with zero
                delta_this
            )
            x_with_feature = torch.cat([x, delta_this], dim=1)  # [batch, input_size+1]
            fix = self.feature_fix(x_with_feature)  # [batch, input_size]
            fixes.append(fix.unsqueeze(1))  # [batch, 1, input_size]

        
        # Stack fixes: [batch, num_features, input_size] -> flatten features
        fixes = torch.cat(fixes, dim=1)  # [batch, num_features, input_size]
        fixes_flat = fixes.view(batch_size, -1)  # [batch, num_features*input_size]

        # Combine all fixes into one delta for x
        delta_x = self.shared_tail(fixes_flat)  # [batch, input_size]
        y = x + delta_x  # [batch, input_size]
        return y
