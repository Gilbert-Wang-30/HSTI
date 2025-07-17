#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sympy import denom
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import sys
from pathlib import Path

from features.high_level_feature_extraction import extract_high_level_features  # Assuming this function is defined in your project

# Import the feature extraction function (assumed to be provided in the project)
# from your_project.features import extract_high_level_features  # Uncomment and adjust import as needed

def main():
    parser = argparse.ArgumentParser(description="Reconstruct missing sensor features using causal adjacency and compute MSE.")
    parser.add_argument("--sensor", required=True, type=str, 
                        help="Sensor name (one of PS1-PS6, EPS1, FS1-FS2, TS1-TS4, VS1, SE, CE, CP). Case-insensitive.")
    parser.add_argument("--run_id", default=None, type=str, 
                        help="Optional run identifier for TensorBoard log directory (default: current timestamp).")
    parser.add_argument("--data_dir", default="data", type=str, 
                        help="Base data directory containing raw data and causality subfolder.")
    args = parser.parse_args()
    sensor_name = args.sensor
    run_id = args.run_id or "run_" + __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = Path(__file__).resolve().parent
    data_dir = BASE_DIR / "data"

    # Determine sensor segment indices (10 features each)
    sensors = ["PS1","PS2","PS3","PS4","PS5","PS6",
               "EPS1","FS1","FS2","TS1","TS2","TS3","TS4",
               "VS1","SE","CE","CP"]
    sensor_lower = sensor_name.lower()
    sensor_names_lower = [s.lower() for s in sensors]
    if sensor_lower not in sensor_names_lower:
        raise ValueError(f"Unknown sensor name '{args.sensor}'. Must be one of {sensors}.")
    sensor_index = sensor_names_lower.index(sensor_lower)
    seg_start = sensor_index * 7
    seg_end = seg_start + 7  # exclusive
    assert seg_end <= 119, f"Segment indices out of bounds: {seg_start}-{seg_end}"

    # 1. Extract high-level features for all cycles (as overlapping pairs)
    # (Assuming extract_high_level_features is available in scope or imported from project)
    features_array, _ = extract_high_level_features(f"{data_dir}/raw", start_idx=0, end_idx=2204)

    # Remove unwanted features (var, skew, kurtosis)
    # Original order per sensor: mean, var, std, skew, kurt, max, min, pulse, peak, amp
    remove_indices = np.array([1, 3, 4])  # positions of feat_var, feat_skew, feat_kurt
    keep_indices = np.array([i for i in range(10) if i not in remove_indices])

    # Apply removal for each sensor's 10-feature group across all sensors
    selected_indices = np.concatenate([keep_indices + 10 * sensor for sensor in range(17)])

    # Filter the features array
    features_array = features_array[:, selected_indices]

    # Now each cycle has 119 features instead of 170 (17 sensors * 7 features per sensor)
    assert features_array.shape[1] == 119, f"Feature array should now have shape (N, 119), got {features_array.shape}"

    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    assert features_array.shape[0] % 2 == 0, "Feature count must be even for pairing"

    N_pairs = features_array.shape[0] // 2
    features_array = features_array.reshape(N_pairs, 2, 119)


    # Ensure the shape is (N_pairs, 2, 119)
    assert features_array.shape[1:] == (2, 119), "Feature array should have shape (N, 2, 170)"
    n_pairs = features_array.shape[0]

    # Feature standardization (critical!)
    feature_means = features_array.mean(axis=(0, 1))
    feature_stds = features_array.std(axis=(0, 1)) + 1e-8
    features_scaled = (features_array - feature_means) / feature_stds


    # 2. Identify observed (non-missing) feature indices
    all_indices = np.arange(119)
    target_indices = np.arange(seg_start, seg_end)
    observed_idx = np.setdiff1d(all_indices, target_indices)

    # 3. Load causal adjacency matrix (no thresholding) and reconstruct missing features
    adj_path = f"{data_dir}/causality/pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    with open(adj_path, "rb") as f:
        adj_matrix = pickle.load(f)
    adj_matrix = np.array(adj_matrix)  # ensure it's a numpy array for slicing and dot



    # remove the row for the sensor to predict
    adj_matrix = adj_matrix[selected_indices][:, selected_indices]
    assert adj_matrix.shape == (119,119), f"Adjacency matrix shape mismatch, got {adj_matrix.shape}"
    
    # Normalize adjacency matrix row-wise 
    row_sums = np.abs(adj_matrix).sum(axis=1, keepdims=True) + 1e-8
    adj_matrix = adj_matrix / row_sums
    adj_matrix *= 17  # Scale for better interpretability (optional, adjust as needed)

    # Calculate differences between consecutive cycles
    diffs = features_scaled[:, 1, :] - features_scaled[:, 0, :]
    diffs_obs = diffs[:, observed_idx]
    # Get weights from observed features to target features (shape: 7 x 112)
    W = adj_matrix[seg_start:seg_end, :][:, observed_idx]               # shape (7, 112)


    # Set K - choose only top-K causal features (e.g., top 10)
    K = 10  

    # Get the absolute weights for sorting (importance)
    abs_W = np.abs(W)

    # Initialize a zero-weighted matrix of same shape as W
    W_topk = np.zeros_like(W)

    # Select top-K causal connections for each target feature separately
    for target_feature_idx in range(W.shape[0]):
        # Get indices of top-K causal features for the current target feature
        top_k_idx = np.argsort(abs_W[target_feature_idx])[-K:]
        # Copy only top-K causal feature weights
        W_topk[target_feature_idx, top_k_idx] = W[target_feature_idx, top_k_idx]


    # Predict the deltas for target features for all pairs
    # Predicted scaled deltas
    pred_deltas_scaled = diffs_obs.dot(W_topk.T)

    # Reconstruct scaled target features
    actual_t_scaled = features_scaled[:, 0, seg_start:seg_end]
    reconstructed_t1_scaled = actual_t_scaled + pred_deltas_scaled

    # Actual target features at t+1 (ground truth)
    reconstructed_t1 = reconstructed_t1_scaled * feature_stds[seg_start:seg_end] + feature_means[seg_start:seg_end]
    actual_t1 = features_array[:, 1, seg_start:seg_end]
    naive_pred = features_array[:, 0, seg_start:seg_end]

    # 4. Compute precision for each feature dimension (averaged over all pairs)
    # 1. Causal reconstruction as before (already in precision_per_feature)
    # Replace with RAE to avoid unstable divisions:
    precision_causal = []
    precision_naive = []
    for j in range(seg_end - seg_start):
        num_causal = np.abs(reconstructed_t1[:, j] - actual_t1[:, j])
        num_naive = np.abs(naive_pred[:, j] - actual_t1[:, j])
        denom = np.abs(actual_t1[:, j] - np.mean(actual_t1[:, j])) + 1e-8
        precision_causal.append(100 * (1 - num_causal.sum() / denom.sum()))
        precision_naive.append(100 * (1 - num_naive.sum() / denom.sum()))

    import matplotlib
    matplotlib.use("Agg")  # Always safe for scripts

    fig = plt.figure(figsize=(7, 5))
    bar_width = 0.35
    feature_ids = np.arange(1, 8)
    plt.bar(feature_ids - bar_width/2, precision_causal, width=bar_width, color='skyblue', label='Causal')
    plt.bar(feature_ids + bar_width/2, precision_naive, width=bar_width, color='orange', label='Naive')
    plt.xticks(feature_ids)
    plt.xlabel(f'Feature # (of sensor {sensor_name.upper()})')
    plt.ylabel('Precision (%)')
    plt.title(f'Reconstruction Precision (%) for sensor {sensor_name.upper()}')
    plt.legend()
    plt.tight_layout()

    print(f"Precision for sensor {sensor_name.upper()}:")
    for i in range(seg_start, seg_end):
        print(f"Feature {i - seg_start + 1}: Causal: {precision_causal[i - seg_start]:.2f}%, Naive: {precision_naive[i - seg_start]:.2f}%")

    log_dir = f"runs/causal_reconstruction/{sensor_name.upper()}/{run_id}_{datetime.now().strftime('%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_figure("reconstruction_precision", fig)
    writer.close()
    plt.close(fig)
    
if __name__ == "__main__":
    main()
