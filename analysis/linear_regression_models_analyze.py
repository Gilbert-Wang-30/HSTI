#!/usr/bin/env python3
import pickle
from pathlib import Path

# ------------------ Config ---------------------
# You can change this to any specific file you want to analyze:
MODEL_PATH = Path(__file__).resolve().parent.parent / "features" / "linear_regression_r2_0.50_mape_0.20.pkl"

# Set this to your number of sensors/features-per-sensor:
N_SENSORS = 17
N_FEATURES_PER_SENSOR = 14

# ------------------ Load and Check ---------------------
with open(MODEL_PATH, "rb") as f:
    model_obj = pickle.load(f)
coefs_all = model_obj["coefs_all"]
parents_all = model_obj["parents_all"]
print(f"[INFO] Loaded {MODEL_PATH}")
print(f"[INFO] Features: {len(coefs_all)}")

# ------------------ Per-Feature & Per-Sensor Analysis ---------------------
valid_per_feature = [(c is not None and len(parents_all[i]) > 0) for i, c in enumerate(coefs_all)]

# Print valid for each feature in each sensor
print("\n=== Valid features per sensor ===")
for s in range(N_SENSORS):
    start = s * N_FEATURES_PER_SENSOR
    end = start + N_FEATURES_PER_SENSOR
    valid_count = sum(valid_per_feature[start:end])
    valid_indices = [i for i in range(start, end) if valid_per_feature[i]]
    print(f"Sensor {s:2d} [{start:3d}-{end-1:3d}]:  {valid_count:2d} valid features | Valid indices: {valid_indices}")

# Print valid count per feature position across all sensors (e.g., all "feature 0" across sensors)
print("\n=== Valid count per feature position (across sensors) ===")
for f in range(N_FEATURES_PER_SENSOR):
    count = sum(valid_per_feature[f + s*N_FEATURES_PER_SENSOR] for s in range(N_SENSORS))
    print(f"Feature {f:2d}: {count:2d} valid across all sensors")

# Total valid models
total_valid = sum(valid_per_feature)
print(f"\n=== Overall: {total_valid} valid models out of {len(coefs_all)} features ({100*total_valid/len(coefs_all):.2f}%) ===")
