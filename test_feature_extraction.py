import torch
import numpy as np
from data_loader import data_loader
from high_level_feature_extraction import extract_high_level_features, extract_cycle_features  # change import path as needed

# Set data directory
data_dir = "/home/wangyuxiao/project/gilbert_copy/HSTI/data"

# Load raw full dataset
dataset = data_loader(data_dir)

# Pick sample index from raw dataset directly
idx = 0
(tensor_100, tensor_10, tensor_1), rul = dataset[idx]  # raw, aligned with extract_high_level_features

# Apply extract_cycle_features on that sample
features1, rul1 = extract_cycle_features(tensor_100, tensor_10, tensor_1, rul.item())

# Apply extract_high_level_features on [idx, idx]
features2, rul2 = extract_high_level_features(data_dir, start_idx=idx, end_idx=idx)

# Compare
f1 = np.array(features1)
f2 = np.array(features2[:6])  # Only the first 6 windows

# Debug
if np.array_equal(f1, f2, equal_nan=True):
    print("✅ Success: extract_cycle_features matches extract_high_level_features[:6], including NaNs.")
else:
    diff = np.nanmax(np.abs(f1 - f2))
    print("❌ Mismatch: max absolute difference =", diff)

