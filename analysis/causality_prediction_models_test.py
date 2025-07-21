#!/usr/bin/env python3
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

def mape_abs(y_true, y_pred):
    # Avoid divide by zero by masking zeros
    mask = np.abs(y_true) != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(np.abs(y_true[mask]) - np.abs(y_pred[mask])) / np.abs(y_true[mask]))


feature_dir = Path(__file__).resolve().parent.parent / "features"
features_pkl = feature_dir / "features.pkl"

# 1. Load all models
# model_files = sorted(feature_dir.glob("linear_regression_r2_thr_*.pkl"))
# model_files = sorted(feature_dir.glob("linear_regression_coefs_thr_*.pkl"))
# model_files = sorted(feature_dir.glob("linear_regression_coefs_r2_thr_*.pkl"))
model_files = sorted(feature_dir.glob("linear_regression_coefs_r2*.pkl"))


model_names = [f.stem for f in model_files]

# 2. Load feature matrix
with open(features_pkl, "rb") as f:
    features_obj = pickle.load(f)
X_full = features_obj["features"]   # shape: (n_samples, n_features)
print(f"[INFO] Loaded feature matrix {X_full.shape} from {features_pkl}")

results_per_model = []

n_features = X_full.shape[1]
n_samples = X_full.shape[0]


# 3. For each model
for model_path, model_name in zip(model_files, model_names):
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)
    # Each model_obj is a dict: keys = "coefs_all", "parents_all"
    coefs_all = model_obj["coefs_all"]
    parents_all = model_obj["parents_all"]


    pred_features = np.zeros_like(X_full)
    mask_valid = np.ones_like(X_full, dtype=bool)  # Mark where prediction is valid

    feature_mae = []
    for j in range(n_features):   
        parents = parents_all[j]
        coefs = coefs_all[j]
        if coefs is None or len(parents) == 0:
            mask_valid[:, j] = False
            feature_mae.append(-1)
            continue
        Xp = X_full[:, parents]
        y_true = X_full[:, j]
        # --- Unpack coefficients/intercept flexibly ---
        if isinstance(coefs, dict) and "coefs" in coefs:
            coefs_arr = coefs["coefs"]
            intercept = coefs["intercept"]
        elif isinstance(coefs, dict) and "coef" in coefs:
            coefs_arr = coefs["coef"]
            intercept = coefs["intercept"]
        elif isinstance(coefs, (tuple, list)) and len(coefs) == 2:
            coefs_arr, intercept = coefs
        else:
            raise ValueError(f"Unknown coef format for feature {j}: {coefs}")
        y_pred = np.dot(Xp, coefs_arr) + intercept
        pred_features[:, j] = y_pred
        err = mape_abs(y_true, y_pred)
        feature_mae.append(err)


    # <-- APPEND RESULTS ONLY ONCE PER MODEL, NOT PER FEATURE
    results_per_model.append({
        "model": model_name,
        "mae_per_feature": np.array(feature_mae),
        "mask_valid": mask_valid,
    })
    mae_valid = [m for m in feature_mae if m != -1 and not np.isnan(m)]
    mean_mae = np.mean(mae_valid) if mae_valid else float('nan')
    print(f"[{model_name}] Per-feature MAPE: {np.round(feature_mae[:10], 4)} ... mean: {mean_mae:.4f}")

print("\n=== Summary Table: ===")
print(f"{'Model':50s} | {'Mean MAPE (all features)':>25}  | {'Valid Features':>15}")
print("-"*60)
for res in results_per_model:
    model = res["model"]
    valid_mae = [m for m in res["mae_per_feature"] if m != -1 and not np.isnan(m)]
    mean_mae = np.mean(valid_mae) if valid_mae else float('nan')
    n_valid_features = np.sum(~np.isnan(res["mae_per_feature"]) & (res["mae_per_feature"] != -1))
    print(f"{model:50s} | {mean_mae:25.5f}  | {n_valid_features:15d}")


print(f"{'Model':50s} | {'Median MAPE (all features)':>25}")
print("-"*60)
for res in results_per_model:
    model = res["model"]
    valid_mae = [m for m in res["mae_per_feature"] if m != -1 and not np.isnan(m)]
    median_mae = np.median(valid_mae) if valid_mae else float('nan')
    n_valid_features = np.sum(~np.isnan(res["mae_per_feature"]) & (res["mae_per_feature"] != -1))
    print(f"{model:50s} | {median_mae:25.5f}  | {n_valid_features:15d}")



# Optionally, show for each feature across models
print("\n=== Per Feature Comparison (first 10 features): ===")
for j in range(n_features):
    line = f"Feature {j:2d}:"
    valid = False
    for res in results_per_model:
        val = res['mae_per_feature'][j]
        if val != -1 and not np.isnan(val):
            valid = True
        line += f" {val:8.4f}" if val != -1 and not np.isnan(val) else "    [NA]  "
    if valid:
        print(line)
