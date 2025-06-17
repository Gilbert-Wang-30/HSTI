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
START_CYCLE = 0
END_CYCLE = 209
PATH_DIR = "/home/wangyuxiao/project/gilbert_copy/HSTI/data"
MAX_LAG = 1                 # Maximum time lag to consider (default 1)
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
results = pcmci.run_pcmci(tau_min=1, tau_max=MAX_LAG, pc_alpha=None, alpha_level=SIGNIFICANCE_LEVEL)
p_matrix, val_matrix = results['p_matrix'], results['val_matrix']

# 3. Construct adjacency matrix (binary) from p_matrix and print results
# Note: If multiple lags, p_matrix and val_matrix have shape (N, N, tau_max+1) including lag0.
# We consider lags 1..MAX_LAG. We'll create adjacency matrices for each lag.
adjacency_matrices = (p_matrix[..., 1:MAX_LAG+1] <= SIGNIFICANCE_LEVEL).astype(int)
strength_matrices  = val_matrix[..., 1:MAX_LAG+1]

# Print adjacency and strength matrices
if MAX_LAG == 1:
    # For single lag, adjacency_matrices and strength_matrices are 3D with size 1 in last axis
    adj_matrix = adjacency_matrices[..., 0]  # shape (N, N)
    strength_matrix = strength_matrices[..., 0]  # shape (N, N)
    print("Causal adjacency matrix (lag 1, binary 0/1):")
    print(adj_matrix)
    print("Strength matrix for lag 1 (partial correlation values):")
    print(strength_matrix)
else:
    # For multiple lags, iterate and print each matrix
    num_vars = adjacency_matrices.shape[0]
    for tau in range(1, MAX_LAG+1):
        adj_mat_tau = adjacency_matrices[..., tau-1]
        strength_mat_tau = strength_matrices[..., tau-1]
        print(f"\nCausal adjacency matrix for lag {tau}:")
        print(adj_mat_tau)
        print(f"Strength matrix for lag {tau}:")
        print(strength_mat_tau)

# Save the binary adjacency matrix (or matrices) to a .pkl file
outfile = f"pcmci_adj_matrix_cycles_{START_CYCLE}_to_{END_CYCLE}.pkl"
with open(outfile, "wb") as f:
    pickle.dump(adjacency_matrices, f)
print(f"\nBinary adjacency matrix saved to {outfile}")