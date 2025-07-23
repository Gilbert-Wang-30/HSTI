import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pickle
from pathlib import Path

class SensorWindowDataset(Dataset):
    def __init__(self, raw_data_dir, features_path=None):
        self.sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
        self.sensors_10hz = ["FS1", "FS2"]
        self.sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
        self.all_sensors = self.sensors_100hz + self.sensors_10hz + self.sensors_1hz
        self.n_sensors = len(self.all_sensors)   # 17

        # Load raw data
        data_100_list, data_10_list, data_1_list = [], [], []
        for sensor in self.sensors_100hz:
            arr = np.loadtxt(os.path.join(raw_data_dir, f"{sensor}.txt"), delimiter='\t', dtype=np.float32)
            data_100_list.append(arr)
        self.data_100 = np.stack(data_100_list, axis=1)

        for sensor in self.sensors_10hz:
            arr = np.loadtxt(os.path.join(raw_data_dir, f"{sensor}.txt"), delimiter='\t', dtype=np.float32)
            data_10_list.append(arr)
        self.data_10 = np.stack(data_10_list, axis=1)

        for sensor in self.sensors_1hz:
            arr = np.loadtxt(os.path.join(raw_data_dir, f"{sensor}.txt"), delimiter='\t', dtype=np.float32)
            data_1_list.append(arr)
        self.data_1 = np.stack(data_1_list, axis=1)

        # Windowing: [cycles, sensors, full-length] -> [cycles*6, sensors, window]
        N_cycles = self.data_100.shape[0]
        self.N_cycles = N_cycles
        self.data_100 = self.data_100.reshape(N_cycles, 7, 6, 1000).transpose(0, 2, 1, 3).reshape(-1, 7, 1000)
        self.data_10  = self.data_10.reshape(N_cycles, 2, 6, 100).transpose(0, 2, 1, 3).reshape(-1, 2, 100)
        self.data_1   = self.data_1.reshape(N_cycles, 8, 6, 10).transpose(0, 2, 1, 3).reshape(-1, 8, 10)

        # Features (from features.pkl): (N_cycles, 6, 17, 14) -> (N_cycles*6, 17, 14)
        with open(features_path, "rb") as f:
            features_obj = pickle.load(f)
        self.features = features_obj["features"].reshape(-1, 17, 14)

        # Sanity checks
        assert self.data_100.shape[0] == self.data_10.shape[0] == self.data_1.shape[0] == self.features.shape[0]
        self.n_windows = self.features.shape[0]


        # ---- PRECOMPUTE LR-PREDICTED FEATURES FOR ALL SENSORS, ALL WINDOWS ----
        LR_MODEL_PATH = Path(raw_data_dir).parent.parent / "features" / "linear_regression_r2_0.50_mape_0.20.pkl"
        with open(LR_MODEL_PATH, "rb") as f:
            lr_model = pickle.load(f)
        coefs_all = lr_model["coefs_all"]
        parents_all = lr_model["parents_all"]

        n_windows = self.features.shape[0]
        n_sensors = self.n_sensors  # 17
        n_feats = 14

        self.lr_pred_features = np.full((n_windows, n_sensors, n_feats), np.nan, dtype=np.float32)
        num_pred = 0
        total = 0

        print("[LR PRECOMPUTE] Predicting high-level features for all sensors, all windows...")
        for idx in range(n_windows):

            present = self.features[idx].flatten()  # shape (238,)
            for sensor_id in range(n_sensors):
                for feat in range(n_feats):
                    j = sensor_id * n_feats + feat
                    coefs = coefs_all[j]
                    if coefs is None:
                        continue
                    parents = coefs["parents"]
                    pred = np.dot(present[parents], coefs["coef"]) + coefs["intercept"]
                    self.lr_pred_features[idx, sensor_id, feat] = pred
                    
                    if not np.isnan(pred):
                        num_pred += 1
                    total += 1
        print(f"[LR PRECOMPUTE] Done. {num_pred}/{total} features predicted (the rest are NaN).")
        
        #  ---verify error of feature predictions---
        
        print("\n[LR PREDICTION MSE STATISTICS]")

        # Compute MSE for every (sensor, feature) across all windows
        mse_all = []
        for sensor_idx, sensor in enumerate(self.all_sensors):
            mse_sensor = []
            print(f"Sensor {sensor:>5s} | ", end="")
            for feat_idx in range(14):
                # Extract predictions and ground truth for this feature across all windows
                pred = self.lr_pred_features[:, sensor_idx, feat_idx]
                gt   = self.features[:, sensor_idx, feat_idx]
                # Only consider non-NaN pairs
                mask = ~np.isnan(pred) & ~np.isnan(gt)
                if mask.sum() == 0:
                    mse = np.nan
                else:
                    mse = np.mean((pred[mask] - gt[mask]) ** 2)
                mse_sensor.append(mse)
                print(f"F{feat_idx}: {mse:.5f}", end=" | ")
            mse_all.extend([m for m in mse_sensor if not np.isnan(m)])
            print()
            # Optionally, print best/worst
            valid_idx = [i for i, m in enumerate(mse_sensor) if not np.isnan(m)]
            if valid_idx:
                best = min(valid_idx, key=lambda i: mse_sensor[i])
                worst = max(valid_idx, key=lambda i: mse_sensor[i])
                print(f"  [Best: F{best}={mse_sensor[best]:.5f}] [Worst: F{worst}={mse_sensor[worst]:.5f}]")

        print("\n[SUMMARY MSE STATISTICS]")
        mse_all = np.array(mse_all)
        print(f"  Mean  MSE over all sensors/features:  {np.nanmean(mse_all):.6f}")
        print(f"  Median MSE over all sensors/features: {np.nanmedian(mse_all):.6f}")
        print(f"  #Features with MSE < 1.0: {np.sum(mse_all < 1.0)} / {mse_all.size}")
        print(f"  #Features with MSE > 10.0: {np.sum(mse_all > 10.0)} / {mse_all.size}")



    def __len__(self):
        # Number of possible (prev, curr) pairs, minus 1 at the end
        return self.n_windows - 1

    def __getitem__(self, idx):
        """
        idx: index of the current (present) window (use idx-1 for past)
        missing_sensor_idx: which sensor (0-16) is 'missing'
        """

        # Accept tuple for (idx, missing_sensor_idx)
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, missing_sensor_idx = idx
        else:
            raise ValueError("You must provide both idx and missing_sensor_idx as a tuple (idx, missing_sensor_idx)")
        # --- Feature indices ---
        # Previous window
        features_past = self.features[idx-1]  # shape (17, 14)
        # Current window
        features_present = self.features[idx]  # shape (17, 14)

        # Remove missing sensor from present features for input
        features_present_input = np.delete(features_present, missing_sensor_idx, axis=0)  # shape (16, 14)
        # Target: present window features of missing sensor
        features_present_target = features_present[missing_sensor_idx]  # shape (14,)

        # --- Raw data: indices ---
        d100_past = self.data_100[idx-1]
        d10_past  = self.data_10[idx-1]
        d1_past   = self.data_1[idx-1]

        d100_present = self.data_100[idx].copy()
        d10_present  = self.data_10[idx].copy()
        d1_present   = self.data_1[idx].copy()

        # Set missing sensor's present raw data to NaN
        # Find out which group and offset
        if missing_sensor_idx < 7:  # 100 Hz group
            d100_present[missing_sensor_idx, :] = np.nan
            present_raw_target = self.data_100[idx][missing_sensor_idx]
        elif missing_sensor_idx < 9:  # 10 Hz
            ms = missing_sensor_idx - 7
            d10_present[ms, :] = np.nan
            present_raw_target = self.data_10[idx][ms]
        else:  # 1 Hz
            ms = missing_sensor_idx - 9
            d1_present[ms, :] = np.nan
            present_raw_target = self.data_1[idx][ms]

        # --- Input ---
        input_dict = {
            "missing_sensor_id": missing_sensor_idx,
            "features_past": features_past.astype(np.float32),          # (17, 14)
            "features_present_excl_missing": features_present_input.astype(np.float32),  # (16, 14)
            "raw_past": {
                "data_100": d100_past.astype(np.float32),
                "data_10": d10_past.astype(np.float32),
                "data_1": d1_past.astype(np.float32),
            },
            "raw_present_excl_missing": {
                "data_100": d100_present.astype(np.float32),
                "data_10": d10_present.astype(np.float32),
                "data_1": d1_present.astype(np.float32),
            }
        }

        # --- Target ---
        target_dict = {
            "features_present_missing": features_present_target.astype(np.float32),  # (14,)
            "raw_present_missing": present_raw_target.astype(np.float32),
        }

        return input_dict, target_dict

# Example usage:
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    raw_data_dir = BASE_DIR / "data" / "raw"
    features_path = BASE_DIR / "features" / "features.pkl"
    output_dir = BASE_DIR / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)
    dataset = SensorWindowDataset(raw_data_dir, features_path=features_path)

    freq_groups = {
        "100hz": list(range(0, 7)),
        "10hz":  list(range(7, 9)),
        "1hz":   list(range(9, 17)),
    }

    def save_pkl(lst, name):
        with open(output_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(lst, f)

    for group, sensor_idxs in freq_groups.items():
        all_samples = []
        for sensor_idx in sensor_idxs:
            for idx in range(1, len(dataset)):
                inp, tgt = dataset[idx, sensor_idx]
                all_samples.append((inp, tgt))
        # Shuffle after concatenation
        np.random.seed(42)
        np.random.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        n_test = n_total - n_train - n_val
        train_set = all_samples[:n_train]
        val_set = all_samples[n_train:n_train+n_val]
        test_set = all_samples[n_train+n_val:]

        save_pkl(train_set, f"train_{group}")
        save_pkl(val_set, f"val_{group}")
        save_pkl(test_set, f"test_{group}")

        print(f"[{group}] Total: {n_total} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    print("All splits saved.")
