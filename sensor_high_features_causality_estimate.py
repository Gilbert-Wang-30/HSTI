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
    seg_start = sensor_index * 10
    seg_end = seg_start + 10  # exclusive

    # 1. Extract high-level features for all cycles (as overlapping pairs)
    # (Assuming extract_high_level_features is available in scope or imported from project)
    features_array, _ = extract_high_level_features(f"{data_dir}/raw", start_idx=0, end_idx=2204)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    assert features_array.shape[0] % 2 == 0, "Feature count must be even for pairing"

    N_pairs = features_array.shape[0] // 2
    features_array = features_array.reshape(N_pairs, 2, 170)


    # Ensure the shape is (N_pairs, 2, 170)
    assert features_array.shape[1:] == (2, 170), "Feature array should have shape (N, 2, 170)"
    n_pairs = features_array.shape[0]

    # 2. Identify observed (non-missing) feature indices
    all_indices = np.arange(170)
    target_indices = np.arange(seg_start, seg_end)
    observed_idx = np.setdiff1d(all_indices, target_indices)

    # 3. Load causal adjacency matrix (no thresholding) and reconstruct missing features
    adj_path = f"{data_dir}/causality/pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    with open(adj_path, "rb") as f:
        adj_matrix = pickle.load(f)
    adj_matrix = np.array(adj_matrix)  # ensure it's a numpy array for slicing and dot

    # Calculate differences between consecutive cycles
    diffs = features_array[:, 1, :] - features_array[:, 0, :]           # shape (n_pairs, 170)
    diffs_obs = diffs[:, observed_idx]                                  # shape (n_pairs, 160)

    # Get weights from observed features to target features (shape: 10 x 160)
    W = adj_matrix[seg_start:seg_end, :][:, observed_idx]               # shape (10, 160)

    # Predict the deltas for target features for all pairs
    pred_deltas = diffs_obs.dot(W.T)                                    # shape (n_pairs, 10)

    # Reconstruct target features for t+1 by adding predicted deltas to t features
    actual_t = features_array[:, 0, seg_start:seg_end]                  # shape (n_pairs, 10)
    reconstructed_t1 = actual_t + pred_deltas                           # shape (n_pairs, 10)

    # Actual target features at t+1 (ground truth)
    actual_t1 = features_array[:, 1, seg_start:seg_end]                 # shape (n_pairs, 10)

    # 4. Compute precision for each feature dimension (averaged over all pairs)
    # 1. Causal reconstruction as before (already in precision_per_feature)
    epsilon = 1e-8
    denom = np.abs(actual_t1) + epsilon
    mask = denom > 1e-2
    naive_pred = actual_t  # Just use t as prediction for t+1

    precision_causal = []
    precision_naive = []
    for j in range(10):
        valid = mask[:, j]
        if valid.sum() > 0:
            mape_c = np.abs((reconstructed_t1[valid, j] - actual_t1[valid, j]) / (np.abs(actual_t1[valid, j]) + epsilon)) * 100
            mape_n = np.abs((naive_pred[valid, j] - actual_t1[valid, j]) / (np.abs(actual_t1[valid, j]) + epsilon)) * 100
            precision_causal.append(100 - mape_c.mean())
            precision_naive.append(100 - mape_n.mean())
        else:
            precision_causal.append(np.nan)
            precision_naive.append(np.nan)
    precision_causal = np.array(precision_causal)
    precision_naive = np.array(precision_naive)

    # Print for inspection
    print("Precision (causal) per feature (%):", precision_causal)
    print("Precision (naive, use t as t+1) per feature (%):", precision_naive)
    # 5. Log the precision values to TensorBoard as a bar chart   

    print("Precision per feature (%):", precision_causal)


    print("Adjacency matrix stats:", np.mean(W), np.std(W), np.min(W), np.max(W))

    print("First cycle feature vector (sensor):", actual_t[0])
    print("Next cycle feature vector (sensor):", actual_t1[0])


    print("Any NaN in features_array?", np.isnan(features_array).any())
    print("Any Inf in features_array?", np.isinf(features_array).any())
    print("Any NaN in diffs_obs?", np.isnan(diffs_obs).any())
    print("Any Inf in diffs_obs?", np.isinf(diffs_obs).any())
    print("Any NaN in W?", np.isnan(W).any())
    print("Any Inf in W?", np.isinf(W).any())


    import matplotlib
    matplotlib.use("Agg")  # Always safe for scripts

    fig = plt.figure(figsize=(7, 5))
    bar_width = 0.35
    feature_ids = np.arange(1, 11)
    # plt.bar(feature_ids - bar_width/2, precision_causal, width=bar_width, color='skyblue', label='Causal')
    plt.bar(feature_ids + bar_width/2, precision_naive, width=bar_width, color='orange', label='Naive')
    plt.xticks(feature_ids)
    plt.xlabel(f'Feature # (of sensor {sensor_name.upper()})')
    plt.ylabel('Precision (%)')
    plt.title(f'Reconstruction Precision (%) for sensor {sensor_name.upper()}')
    plt.legend()
    plt.tight_layout()

    log_dir = f"runs/causal_reconstruction/{sensor_name.upper()}/{run_id}_{datetime.now().strftime('%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_figure("reconstruction_precision", fig)
    writer.close()
    plt.close(fig)

    print("Some true values for feature 1:", actual_t1[:20, 0])
    print("Some naive predictions for feature 1:", naive_pred[:20, 0])
    print("Some causal predictions for feature 1:", reconstructed_t1[:20, 0])
    print("Mean abs true for f1:", np.mean(np.abs(actual_t1[:, 0])))
    print("Mean abs error (naive) for f1:", np.mean(np.abs(naive_pred[:, 0] - actual_t1[:, 0])))
    print("Mean abs error (causal) for f1:", np.mean(np.abs(reconstructed_t1[:, 0] - actual_t1[:, 0])))
    print("How many valid points per feature:", mask.sum(axis=0))

    print(f"Logged reconstruction precision bar chart to TensorBoard under {log_dir}")
if __name__ == "__main__":
    main()
