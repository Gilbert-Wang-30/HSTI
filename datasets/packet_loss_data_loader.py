import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pickle
from pathlib import Path

class SensorWindowDataset(Dataset):
    def __init__(self, raw_data_dir, features_path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

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
        
        # --------- exctrac raw data predictions from npy ---------
        lstm_pred_dir = Path(raw_data_dir).parent / "lstm_predictions"
        self.lstm_pred_raw = {
            "100hz": np.load(os.path.join(lstm_pred_dir, "lstm_100.npy")),
            "10hz": np.load(os.path.join(lstm_pred_dir, "lstm_10.npy")),
            "1hz": np.load(os.path.join(lstm_pred_dir, "lstm_1.npy"))
        }
        self.lstm_pred_highlevel_features = np.load(os.path.join(lstm_pred_dir, "lstm_features.npy"))
            


    def __len__(self):
        # Number of possible (prev, curr) pairs
        return self.n_windows

    def __getitem__(self, idx):
        """
        idx: index of the current (present) window (use idx-1 for past)
        missing_sensor_idx: which sensor (0-16) is 'missing'
        """
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, missing_sensor_idx = idx
        else:
            raise ValueError("You must provide both idx and missing_sensor_idx as a tuple (idx, missing_sensor_idx)")

        
        # LR-predicted present features for missing sensor
        lr_pred = self.lr_pred_features[idx, missing_sensor_idx, :]

        # Ground-truth present raw data for missing sensor
        if missing_sensor_idx < 7:
            raw_present = self.data_100[idx, missing_sensor_idx, :]
            raw_present_all = self.data_100[idx].copy()
            raw_present_all[missing_sensor_idx, :] = np.nan  # simulate missing
            lstm_pred = self.lstm_pred_raw["100hz"][idx, missing_sensor_idx, :]

        elif missing_sensor_idx < 9:
            raw_present_all = self.data_10[idx].copy()
            ms = missing_sensor_idx - 7
            raw_present = self.data_10[idx, ms, :]
            raw_present_all[ms, :] = np.nan  # simulate missing
            lstm_pred = self.lstm_pred_raw["10hz"][idx, ms, :]
        else:
            raw_present_all = self.data_1[idx].copy()
            ms = missing_sensor_idx - 9
            raw_present = self.data_1[idx, ms, :]
            raw_present_all[ms, :] = np.nan
            lstm_pred = self.lstm_pred_raw["1hz"][idx, ms, :]


        # LSTM high-level features
        lstm_pred_hl_feat = self.lstm_pred_highlevel_features[idx, missing_sensor_idx*14:(missing_sensor_idx+1)*14]  # shape (14,)

        # Prepare dicts
        input_dict = {
            "missing_sensor_id": missing_sensor_idx,
            "lstm_pred_present": lstm_pred.astype(np.float32),
            "lstm_pred_present_highlevel": lstm_pred_hl_feat.astype(np.float32),
            "lr_pred_present": lr_pred.astype(np.float32),
            "raw_present_excl_missing": raw_present_all.astype(np.float32)
        }

        target = raw_present.astype(np.float32)

        return input_dict, target

# ------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    raw_data_dir = BASE_DIR / "data" / "raw"
    features_path = BASE_DIR / "features" / "features.pkl"
    output_dir = BASE_DIR / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)
    dataset = SensorWindowDataset(raw_data_dir, features_path=features_path)


    print("lstm_features.npy shape:", dataset.lstm_pred_highlevel_features.shape)
    print("Example [0, :14]:", dataset.lstm_pred_highlevel_features[0, :14])
    print("Example [1, 14:28]:", dataset.lstm_pred_highlevel_features[1, 14:28])

    # For demonstration, let's just print for several sensors/windows
    # indices_to_check = [ (10, 0), (100, 7), (500, 13), (1000, 16) ]
    indices_to_check = [ (0, 0), (1, 7), (2, 13) ]
    
    for idx, sensor_idx in indices_to_check:
        inp, tgt = dataset[(idx, sensor_idx)]
        print(f"\nSample idx={idx}, sensor={dataset.all_sensors[sensor_idx]}")
        print(f"  missing_sensor_id: {inp['missing_sensor_id']}")
        print(f"  lstm_pred_present shape: {inp['lstm_pred_present'].shape}  (first 5) {inp['lstm_pred_present'][:5]}")
        print(f"  lr_pred_present shape: {inp['lr_pred_present'].shape}  (first 5) {inp['lr_pred_present'][:5]}")
        print(f"  Target raw_present shape: {tgt.shape}  (first 5) {tgt[:5]}")

    # test print number of idx in dataset expect 13230
    print(f"\n[INFO] Total samples in dataset: {len(dataset)}")

    print("\n[INFO] Sample generation complete.")
    freq_groups = {
        "100hz": list(range(0, 7)),
        "10hz":  list(range(7, 9)),
        "1hz":   list(range(9, 17)),
    }

    def save_pkl(lst, name):
        with open(output_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(lst, f)

    for group, sensor_idxs in freq_groups.items():
        train_set, val_set, test_set = [], [], []
        for sensor_idx in sensor_idxs:
            samples = []
            for idx in range(1, len(dataset)):
                inp, tgt = dataset[(idx, sensor_idx)]
                samples.append((inp, tgt))
            np.random.seed(42 + sensor_idx)  # different seed per sensor for reproducibility
            np.random.shuffle(samples)
            n_total = len(samples)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            n_test = n_total - n_train - n_val
            train_set.extend(samples[:n_train])
            val_set.extend(samples[n_train:n_train + n_val])
            test_set.extend(samples[n_train + n_val:])
        # Shuffle after concatenation (for extra randomness)
        np.random.seed(42)
        np.random.shuffle(train_set)
        np.random.shuffle(val_set)
        np.random.shuffle(test_set)
        save_pkl(train_set, f"train_{group}")
        save_pkl(val_set, f"val_{group}")
        save_pkl(test_set, f"test_{group}")
        print(f"[{group}] Total: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
        print(f"Saved {group} datasets to {output_dir}")