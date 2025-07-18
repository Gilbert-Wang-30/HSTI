#!/usr/bin/env python3
"""
Linear regression on features using PCMCI causal parents (threshold-based).
Project-specific: No parser except --threshold, paths are hardcoded.
"""

import numpy as np
import pickle
import os
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- Parse only threshold argument ---
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.8, help='Causality strength threshold (default 0.8)')
args = parser.parse_args()
THRESHOLD = args.threshold

# --- Configure project paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "features" / "features.pkl"
CAUSALITY_DIR = BASE_DIR / "data" / "causality"
ADJ_PATH = CAUSALITY_DIR / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
STRENGTH_PATH = CAUSALITY_DIR / "pcmci_instant_strength_matrix_cycles_0_to_2204_lag0.pkl"
COEF_OUTPUT = BASE_DIR / "features" / f"linear_regression_coefs_thr_{THRESHOLD:.2f}.pkl"

print(f"[LOAD] Loading features from {FEATURE_PATH}")
with open(FEATURE_PATH, "rb") as f:
    feat_obj = pickle.load(f)
features = feat_obj["features"]  # (T, D)
print(f"[LOAD] Feature matrix shape: {features.shape}")

print(f"[LOAD] Loading adjacency matrix from {ADJ_PATH}")
with open(ADJ_PATH, "rb") as f:
    adj = pickle.load(f)  # (D, D)

print(f"[LOAD] Loading strength matrix from {STRENGTH_PATH}")
with open(STRENGTH_PATH, "rb") as f:
    strength = pickle.load(f)  # (D, D)

n_samples, n_features = features.shape

# --- Fit linear regression for each feature using strong parents ---
coefs_all = []
parents_all = []

for j in range(n_features):
    # Find parent indices (other features) with strong absolute causality to feature j
    strong_parents = np.where(np.abs(strength[j, :]) >= THRESHOLD)[0]
    # Exclude self-loop
    strong_parents = strong_parents[strong_parents != j]
    parents_all.append(strong_parents.tolist())
    if len(strong_parents) == 0:
        coefs_all.append(None)
        print(f"Feature {j:3d}: [No parent | Skipped]")
        continue

    X = features[:, strong_parents]
    y = features[:, j]
    # Fit OLS linear regression with intercept
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    coefs_all.append({
        "coef": model.coef_,
        "intercept": model.intercept_,
        "parents": strong_parents.tolist(),
        "r2": model.score(X, y),
        "n_parents": len(strong_parents)
    })
    # Print summary
    print(f"Feature {j:3d}: n_parents={len(strong_parents):2d}, R2={model.score(X, y):.3f} | parents={strong_parents.tolist()}")

# --- Save regression coefficients ---
save_obj = {
    "coefs_all": coefs_all,
    "parents_all": parents_all,
    "threshold": THRESHOLD,
    "feature_dim": n_features,
}
with open(COEF_OUTPUT, "wb") as f:
    pickle.dump(save_obj, f)
print(f"[SAVE] All coefficients saved to: {COEF_OUTPUT}")
