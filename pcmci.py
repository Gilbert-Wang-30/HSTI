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

# Parameters (adjustable)
START_CYCLE = 1464
END_CYCLE = 1663
PATH_DIR = "/home/wangyuxiao/project/gilbert_copy/HSTI/data"
MAX_LAG = 0                 # Maximum time lag to consider (default 1)
SIGNIFICANCE_LEVEL = 0.001   # Significance level for causal links (default 0.01)

# 1. Feature extraction
print(f"Extracting high-level features for cycles {START_CYCLE} to {END_CYCLE}...")
features, extracted_rul = extract_high_level_features(data_dir=PATH_DIR, start_idx=START_CYCLE, end_idx=END_CYCLE)

# Remove features that are constant or contain NaNs
valid_cols = (np.std(features, axis=0) > 1e-6) & (~np.isnan(features).any(axis=0))
features = features[:, valid_cols]
print(f"[Cleanup] Removed {np.sum(~valid_cols)} constant/NaN features → New shape: {features.shape}")

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
    adj_matrix = (p_matrix[..., 0] <= SIGNIFICANCE_LEVEL).astype(int)
    strength_matrix = val_matrix[..., 0]
    print("\nCausal adjacency matrix (lag 0, binary 0/1):")
    print(adj_matrix)
    print("\nStrength matrix for lag 0 (partial correlation values):")
    print(strength_matrix)

    # Save matrix
    outfile = os.path.join(OUT_DIR, f"pcmci_instant_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag0.pkl")
    with open(outfile, "wb") as f:
        pickle.dump(adj_matrix, f)

elif MAX_LAG == 1:
    # Lag 1 only
    adj_matrix = (p_matrix[..., 1] <= SIGNIFICANCE_LEVEL).astype(int)
    strength_matrix = val_matrix[..., 1]
    print("\nCausal adjacency matrix (lag 1, binary 0/1):")
    print(adj_matrix)
    print("\nStrength matrix for lag 1 (partial correlation values):")
    print(strength_matrix)

    # Save matrix
    outfile = os.path.join(OUT_DIR, f"pcmci_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag1.pkl")
    with open(outfile, "wb") as f:
        pickle.dump(adj_matrix, f)

else:
    # Multi-lag: save all lags 1..MAX_LAG
    adjacency_matrices = (p_matrix[..., 1:MAX_LAG+1] <= SIGNIFICANCE_LEVEL).astype(int)
    strength_matrices = val_matrix[..., 1:MAX_LAG+1]

    for tau in range(1, MAX_LAG + 1):
        adj_tau = adjacency_matrices[..., tau - 1]
        strength_tau = strength_matrices[..., tau - 1]
        print(f"\nCausal adjacency matrix for lag {tau}:")
        print(adj_tau)
        print(f"Strength matrix for lag {tau}:")
        print(strength_tau)

        # Save each tau-lag matrix separately
        out_tau = os.path.join(OUT_DIR, f"pcmci_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}_lag{tau}.pkl")
        with open(out_tau, "wb") as f:
            pickle.dump(adj_tau, f)

print(f"Adjacency matrices saved under: {OUT_DIR}")