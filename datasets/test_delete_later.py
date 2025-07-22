import os
import numpy as np
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import random

import sys
sys.path.insert(0, "/home/wangyuxiao/project/gilbert_copy/HSTI")
from features.high_level_feature_extraction import extract_cycle_features
from packet_loss_data_loader import SensorWindowDataset

# Assume SensorWindowDataset from your previous code

# Load dataset and features
raw_data_dir = "/home/wangyuxiao/project/gilbert_copy/HSTI/data/raw"    # <-- set path as appropriate
features_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/features/features.pkl"
dataset = SensorWindowDataset(raw_data_dir, features_path)

# Load high-level feature extraction function
from features.high_level_feature_extraction import extract_cycle_features

def verify_window_features(idx):
    # 1. Extract raw data for window idx
    d100 = torch.from_numpy(dataset.data_100[idx])
    d10 = torch.from_numpy(dataset.data_10[idx])
    d1 = torch.from_numpy(dataset.data_1[idx])
    # 2. Get stored features
    stored_features = dataset.features[idx]  # (17, 14)

    # 3. The window data here is for a single 10s window (windowed already)
    #   - reshape d100: (7, 1000) --> (7, 1, 1000)
    #   - reshape d10:  (2, 100)  --> (2, 1, 100)
    #   - reshape d1:   (8, 10)   --> (8, 1, 10)
    d100_windowed = d100.unsqueeze(1).numpy()  # (7, 1, 1000)
    d10_windowed  = d10.unsqueeze(1).numpy()   # (2, 1, 100)
    d1_windowed   = d1.unsqueeze(1).numpy()    # (8, 1, 10)

    # 4. Extract features for this window
    # extract_cycle_features expects torch.FloatTensor as input
    fmat, _ = extract_cycle_features(
        torch.from_numpy(d100_windowed),
        torch.from_numpy(d10_windowed),
        torch.from_numpy(d1_windowed),
        0  # rul_value, not used for this test
    )  # fmat: (14*17, 1), i.e., (238, 1) or sometimes (170, 1) depending on version

    # Adapt shape for comparison
    # fmat may be (features, n_windows), here n_windows==1
    if fmat.shape[-1] == 1:
        recomputed = fmat[:, 0]
    else:
        recomputed = fmat.reshape(-1)
    # stored_features is (17, 14); flatten for direct comparison
    stored_flat = stored_features.reshape(-1)

    # 5. Compare
    same = np.allclose(stored_flat, recomputed, atol=1e-5, equal_nan=True)
    max_diff = np.nanmax(np.abs(stored_flat - recomputed))
    print(f"[IDX {idx}] Features match: {same} (max abs diff: {max_diff:.4g})")
    if not same:
        print("First few differences:")
        print("stored:", stored_flat[:10])
        print("recomputed:", recomputed[:10])

if __name__ == "__main__":
    total = len(dataset.features)
    # Test a few random windows
    for idx in [0, 1, 10, 100, 1000, total-1]:
        verify_window_features(idx)
