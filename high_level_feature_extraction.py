# high_level_feature_extraction.py
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from data_loader import data_loader # for loading pkl files
from scipy.stats import skew, kurtosis

# This function loads training data from a pickle file and returns a DataLoader.
def load_train_data(pkl_path, batch_size=16, shuffle=False):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# This function extracts statistical features from a slice of data.
def extract_stat_features(slice_data):  # shape: (B, 6, C, T)
    # Convert to numpy for scipy
    data = slice_data.numpy()  # shape: (B, 6, C, T)
    B, W, C, T = data.shape
    data = data.reshape(B * W * C, T)  # reshape to (B*6*C, T)

    features = {
        'mean': np.mean(data, axis=1),
        'var': np.var(data, axis=1),
        'std': np.std(data, axis=1),
        'skew': skew(data, axis=1),
        'kurt': kurtosis(data, axis=1),
        'max': np.max(data, axis=1),
        'min': np.min(data, axis=1),
        'pulse_count': np.sum(np.abs(np.diff(data, axis=1)) > 0, axis=1),
        'peak': np.max(np.abs(data), axis=1),
        'amplitude': np.max(data, axis=1) - np.min(data, axis=1)
    }

    # Concatenate features along axis 1
    all_feats = np.stack(list(features.values()), axis=1)  # shape: (B*6*C, 10)
    return all_feats.reshape(B, W, C, -1)  # (B, 6, C, 10)

# This function extracts high-level features from the data loader.
def extract_high_level_features(data_loader):
    feature_batches = []
    label_batches = []
    raw_x100_batches = []
    raw_x10_batches = []
    raw_x1_batches = []


    for (x100, x10, x1), labels in data_loader:
        with torch.no_grad():
            f100 = extract_stat_features(x100)  # (B, 6, 7, 10)
            f10 = extract_stat_features(x10)    # (B, 6, 2, 10)
            f1  = extract_stat_features(x1)     # (B, 6, 8, 10)

            combined = np.concatenate([f100, f10, f1], axis=2)  # (B, 6, 17, 10)

            feature_batches.append(torch.tensor(combined, dtype=torch.float32))
            label_batches.append(labels)
          # Save raw inputs for future use
            raw_x100_batches.append(x100)
            raw_x10_batches.append(x10)
            raw_x1_batches.append(x1)

    return feature_batches, label_batches, raw_x100_batches, raw_x10_batches, raw_x1_batches

if __name__ == "__main__":
    train_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train.pkl"
    train_data = load_train_data(train_path, batch_size=16, shuffle=True)

    # Iterate through one batch to check shapes
    for i, ((x100, x10, x1), y) in enumerate(train_data):
        print(f"Batch {i}:")
        print(f"  100Hz shape: {x100.shape}, 10Hz shape: {x10.shape}, 1Hz shape: {x1.shape}")
        print(f"  Labels shape: {y.shape}")
        break  # only load one batch

    # Extract high-level features
    feature_batches, label_batches, raw_x100_batches, raw_x10_batches, raw_x1_batches = extract_high_level_features(train_data)
    # print("Low-level features shape:", (x100.shape, x10.shape, x1.shape))  # should be (B, 6, C, T) for each
    print("Raw Data Shape:", raw_x100_batches[0].shape, raw_x10_batches[0].shape, raw_x1_batches[0].shape)  # should be (B, 6, C, T) for each
    print("Extracted features shape:", feature_batches[0].shape)  # should be (N, 6, 17, 10)
    print("Extracted labels shape:", label_batches[0].shape)
    # 1. Check number of batches (i.e., list lengths)
    n_batches = len(label_batches)
    assert len(feature_batches) == n_batches, "Mismatch in number of feature batches"
    assert len(raw_x100_batches) == n_batches, "Mismatch in number of x100 batches"
    assert len(raw_x10_batches) == n_batches, "Mismatch in number of x10 batches"
    assert len(raw_x1_batches) == n_batches, "Mismatch in number of x1 batches"

    # 2. Check sample size within each batch
    for i in range(n_batches):
        B = label_batches[i].shape[0]
        assert feature_batches[i].shape[0] == B, f"Batch {i}: feature-label size mismatch"
        assert raw_x100_batches[i].shape[0] == B, f"Batch {i}: x100-label size mismatch"
        assert raw_x10_batches[i].shape[0] == B, f"Batch {i}: x10-label size mismatch"
        assert raw_x1_batches[i].shape[0] == B, f"Batch {i}: x1-label size mismatch"

    print(f"[OK] Extracted {n_batches} batches — each batch is consistent.")
    print("Feature extraction complete. Ready for next steps.")