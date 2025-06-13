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
def extract_stat_features(slice_data):  # shape: (B, C, 6, T)
    # Convert to numpy for scipy
    data = slice_data.numpy()  # shape: (B, C, 6, T)
    B, C, W, T = data.shape
    data = data.reshape(B * C * 6, T)  # reshape to (B*C*6, T)

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
    return all_feats.reshape(B, C, W, -1)  # (B, C, 6, 10)

# This function extracts high-level features from the data loader.
def extract_high_level_features(data_loader):
    feature_batches = []
    label_batches = []
    raw_x100_batches = []
    raw_x10_batches = []
    raw_x1_batches = []


    for (x100, x10, x1), labels in data_loader:
        with torch.no_grad():
            f100 = extract_stat_features(x100)  # (B, 7, 6, 10)
            f10 = extract_stat_features(x10)    # (B, 2, 6, 10)
            f1  = extract_stat_features(x1)     # (B, 8, 6, 10)

            combined = np.concatenate([f100, f10, f1], axis=1)  # (B, 17, 6, 10)
            # print(f"Combined features shape: {combined.shape}")  # should be (B, 17, 6, 10)
            feature_batches.append(torch.tensor(combined, dtype=torch.float32))
            label_batches.append(labels)
          # Save raw inputs for future use
            raw_x100_batches.append(x100)
            raw_x10_batches.append(x10)
            raw_x1_batches.append(x1)

    return feature_batches, label_batches, raw_x100_batches, raw_x10_batches, raw_x1_batches

if __name__ == "__main__":
    train_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train.pkl"
    train_data = load_train_data(train_path, batch_size=1, shuffle=False)
    feature_batches, label_batches, raw_x100_batches, raw_x10_batches, raw_x1_batches = extract_high_level_features(train_data)
    # Squeeze batches axis
    # Convert to numpy and squeeze batch dim
    X_raw = [f.squeeze(0).numpy() for f in feature_batches]  # shape: (17, 6, 10) each
    y_raw = [l.squeeze(0).item() for l in label_batches]      # scalar RUL per sample

    # Transform all to (6, 170) format
    X_transposed = [x.transpose(1, 0, 2).reshape(6, 170) for x in X_raw]  # shape: (6, 170) per sample

    print(f"X_transposed shape: {len(X_transposed)} samples, each of shape {X_transposed[0].shape}")
    # Save to pickle
    output_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train_features.pkl"
    with open(output_path, "wb") as f:
        pickle.dump((X_transposed, y_raw), f)

    print(f"[INFO] Saved {len(X_transposed)} samples in (6, 170) format to {output_path}")