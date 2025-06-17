#!/usr/bin/env python3
"""
PCMCI causal discovery on high-level features (cycles 0-209).

This script extracts features and runs the PCMCI algorithm to find causal links.
Default parameters: time lag = 1, significance level = 0.01.
To adjust, modify the MAX_LAG and SIGNIFICANCE_LEVEL variables below.
"""

# Necessary imports
import numpy as np
import pickle
import os
import argparse

# Import tigramite PCMCI and independence test
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError as e:
    raise ImportError("Tigramite library is required. Please install via pip (pip install tigramite).") from e

# Import the feature extraction function (assumed to be implemented elsewhere)
try:
    from high_level_feature_extraction import extract_high_level_features
except ImportError as e:
    raise ImportError("Function extract_high_level_features not found. "
                      "Ensure high_level_feature_extraction.py is in the path.") from e



parser = argparse.ArgumentParser(description="PCMCI causal discovery over selected cycles")
parser.add_argument("--start", type=int, required=True, help="Start cycle index")
parser.add_argument("--end", type=int, required=True, help="End cycle index (inclusive)")
parser.add_argument("--lag", type=int, default=1, help="Max lag to consider (default=1)")
args = parser.parse_args()



# Parameters (adjustable)
START_CYCLE = args.start
END_CYCLE = args.end
MAX_LAG = args.lag
PATH_DIR = "/home/wangyuxiao/project/gilbert_copy/HSTI/data"
SIGNIFICANCE_LEVEL = 0.001   # Significance level for causal links (default 0.01)

# 1. Feature extraction
print(f"Extracting high-level features for cycles {START_CYCLE} to {END_CYCLE}...")
features, extracted_rul = extract_high_level_features(data_dir=PATH_DIR, start_idx=START_CYCLE, end_idx=END_CYCLE)

# === Clean up feature matrix and track original indices ===
N_total = features.shape[1]
valid_mask = (np.std(features, axis=0) > 1e-6) & (~np.isnan(features).any(axis=0))
kept_indices = np.where(valid_mask)[0].tolist()
removed_indices = np.where(~valid_mask)[0].tolist()
features_clean = features[:, kept_indices]

print(f"[Cleanup] Removed {len(removed_indices)} features → {features_clean.shape[1]} remaining.")
features = features_clean

# Expect features shape to be (time_points, num_features), e.g., (1260, 170)
print(f"Feature matrix shape: {features.shape}")

# 2. Set up Tigramite DataFrame and PCMCI
dataframe = pp.DataFrame(features)
# Use partial correlation as the conditional independence test
parcorr_test = ParCorr(significance='analytic')  # analytic p-value computation
# Initialize PCMCI object
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr_test, verbosity=0)

# Run PCMCI algorithm with specified lag and significance threshold
print(f"Running PCMCI with tau_max={MAX_LAG} and alpha={SIGNIFICANCE_LEVEL}...")
tau_min = 0 if MAX_LAG == 0 else 1
results = pcmci.run_pcmci(tau_min=tau_min, tau_max=MAX_LAG, pc_alpha=None, alpha_level=SIGNIFICANCE_LEVEL)
p_matrix, val_matrix = results['p_matrix'], results['val_matrix']

# 3. Construct adjacency matrix (binary) from p_matrix and print results
# Note: If multiple lags, p_matrix and val_matrix have shape (N, N, tau_max+1) including lag 0.
# We consider lags 0, 1..MAX_LAG. We'll create adjacency matrices for each lag.
# 3. Construct and save adjacency matrix/matrices
OUT_DIR = "/home/wangyuxiao/project/gilbert_copy/HSTI/pcmci"
os.makedirs(OUT_DIR, exist_ok=True)

if MAX_LAG == 0:
    # Instantaneous causality (lag 0 only)
    adj_small = (p_matrix[..., 0] <= SIGNIFICANCE_LEVEL).astype(int)
    strength_small = val_matrix[..., 0]
    # 2. Reconstruct full 170×170 matrix with zeros at removed indices
    adj_matrix = np.zeros((N_total, N_total), dtype=int)
    strength_matrix = np.zeros((N_total, N_total), dtype=float)

    for i, orig_i in enumerate(kept_indices):
        for j, orig_j in enumerate(kept_indices):
            adj_matrix[orig_i, orig_j] = adj_small[i, j]
            strength_matrix[orig_i, orig_j] = strength_small[i, j]


    # Save matrix
    adj_matrix_path = os.path.join(OUT_DIR, f"pcmci_instant_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag0.pkl")
    strength_matrix_path = os.path.join(OUT_DIR, f"pcmci_instant_strength_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag0.pkl")
    # Save the adjacency matrix to disk
    with open(adj_matrix_path, 'wb') as f:
        pickle.dump(adj_matrix, f)

    # Save the strength matrix to disk
    with open(strength_matrix_path, 'wb') as f:
        pickle.dump(strength_matrix, f)


elif MAX_LAG == 1:
    # Lag 1 only
    adj_small = (p_matrix[..., 1] <= SIGNIFICANCE_LEVEL).astype(int)
    strength_small = val_matrix[..., 1]
        # 2. Reconstruct full 170×170 matrix with zeros at removed indices
    adj_matrix = np.zeros((N_total, N_total), dtype=int)
    strength_matrix = np.zeros((N_total, N_total), dtype=float)

    for i, orig_i in enumerate(kept_indices):
        for j, orig_j in enumerate(kept_indices):
            adj_matrix[orig_i, orig_j] = adj_small[i, j]
            strength_matrix[orig_i, orig_j] = strength_small[i, j]


    # Save matrix
    adj_matrix_path = os.path.join(OUT_DIR, f"pcmci_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag1.pkl")
    strength_matrix_path = os.path.join(OUT_DIR, f"pcmci_strength_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag1.pkl")
    # Save the adjacency matrix to disk 
    with open(adj_matrix_path, 'wb') as f:
        pickle.dump(adj_matrix, f)

    # Save the strength matrix to disk
    with open(strength_matrix_path, 'wb') as f:
        pickle.dump(strength_matrix, f)

else:
    # Multi-lag: save all lags 1..MAX_LAG
    adjacency_small = (p_matrix[..., 1:MAX_LAG+1] <= SIGNIFICANCE_LEVEL).astype(int)
    strength_small = val_matrix[..., 1:MAX_LAG+1]

    adj_matrix = np.zeros((N_total, N_total, MAX_LAG), dtype=int)
    strength_matrix = np.zeros((N_total, N_total, MAX_LAG), dtype=float)

    for tau in range(1, MAX_LAG + 1):
        adj_tau = adjacency_small[..., tau - 1]
        strength_tau = strength_small[..., tau - 1]
            # 2. Reconstruct full 170×170 matrix with zeros at removed indices

        for i, orig_i in enumerate(kept_indices):
            for j, orig_j in enumerate(kept_indices):
                adj_matrix[orig_i, orig_j, tau - 1] = adj_tau[i, j]
                strength_matrix[orig_i, orig_j, tau - 1] = strength_tau[i, j]

        # Save each tau-lag matrix separately
        out_adj = os.path.join(OUT_DIR, f"pcmci_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag{tau}.pkl")
        out_strength = os.path.join(OUT_DIR, f"pcmci_strength_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag{tau}.pkl")
        with open(out_adj, "wb") as f:
            pickle.dump(adj_matrix[..., tau - 1], f)
        with open(out_strength, "wb") as f:
            pickle.dump(strength_matrix[..., tau - 1], f)

print(f"Adjacency matrices saved under: {OUT_DIR}")