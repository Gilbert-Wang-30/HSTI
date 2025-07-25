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


        MAX_WINDOWS = 24
        self.data_100 = self.data_100[:MAX_WINDOWS]
        self.data_10 = self.data_10[:MAX_WINDOWS]
        self.data_1 = self.data_1[:MAX_WINDOWS]
        self.features = self.features[:MAX_WINDOWS]
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
        
        # --------- predict raw data for missing sensors ---------
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add parent directory to path
        from models.LSTM import LSTMModel
        from ts_LSTM_test import predict_sensor_autoregressive

        def get_sensor_window_params(sensor_name):
            SENSOR_GROUPS = {
                "100hz": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"],
                "10hz": ["FS1", "FS2"],
                "1hz": ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
            }
            for freq, sensors in SENSOR_GROUPS.items():
                if sensor_name in sensors:
                    if freq == "100hz":
                        return 999, 1000
                    elif freq == "10hz":
                        return 99, 100
                    else:
                        return 9, 10
            raise ValueError(f"Unknown sensor: {sensor_name}")

        # LSTM predictions for each sensor
        self.lstm_pred_raw = [None] * self.n_sensors  # list of (n_windows, window_len)

        for sensor_idx, sensor in enumerate(self.all_sensors):

            print(f"[LSTM] Predicting for sensor {sensor} ({sensor_idx+1}/{self.n_sensors}) ...")

            # get window params
            win_in, win_out = get_sensor_window_params(sensor)
            model_path = Path("models") / f"{sensor.lower()}_lstm.pth"
            norm_path = Path("models") / f"{sensor.lower()}_lstm_norm_stats.npz"

            # Load model and stats as in your test file
            lstm = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
            lstm.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            lstm.to(self.device)
            lstm.eval()
            norm_stats = np.load(norm_path)
            norm_mean, norm_std = float(norm_stats["mean"]), float(norm_stats["std"])

            # select raw data
            if sensor in self.sensors_100hz:
                data = self.data_100[:, sensor_idx, :]
            elif sensor in self.sensors_10hz:
                data = self.data_10[:, sensor_idx - 7, :]
            else:
                data = self.data_1[:, sensor_idx - 9, :]

            preds = np.full((self.n_windows, win_out), np.nan, dtype=np.float32)
            for idx in range(1, self.n_windows):
                if idx % 500 == 0:
                    print(f"    [LSTM] Sensor {sensor} at window {idx}/{self.n_windows}")

                # get last window's data as init_seq
                prev_win = data[idx-1]
                pred = predict_sensor_autoregressive(
                    lstm, norm_mean, norm_std, prev_win[-win_in:], window_out=win_out, device=self.device
                )

                preds[idx] = pred
            self.lstm_pred_raw[sensor_idx] = preds
            print(f"Sensor {sensor}: LSTM prediction complete, shape={preds.shape}")
        for i, sensor in enumerate(self.all_sensors):
            preds = self.lstm_pred_raw[i]
            print(f"Sensor {sensor}: preds shape = {preds.shape}, window example = {preds[1][:5]}")



        # ---- PRECOMPUTE HIGH-LEVEL FEATURES FOR LSTM-PREDICTED RAW PRESENT ----
        from features.high_level_feature_extraction import extract_cycle_features

        def safe_slice(arr, start, end, length=14):
            sliced = arr[start:end]
            # Pad or trim to exactly 14
            if sliced.shape[0] < length:
                out = np.full((length,), np.nan, dtype=np.float32)
                out[:sliced.shape[0]] = sliced
                return out
            return sliced[:length]



        self.lstm_pred_highlevel_features = [None] * self.n_sensors  # list of (n_windows, 14)

        for sensor_idx, sensor in enumerate(self.all_sensors):
            print(f"[LSTM-HL] Extracting high-level features for {sensor} ({sensor_idx+1}/{self.n_sensors})")
            preds = self.lstm_pred_raw[sensor_idx]  # shape (n_windows, win_out)
            hl_feats = np.full((self.n_windows, 14), np.nan, dtype=np.float32)
            for idx in range(1, self.n_windows):
                pred_raw = preds[idx]  # shape (window_len,)
                print("pred_raw shape:", pred_raw.shape, "first_5 samples: ",pred_raw[:5])
                # Format as torch tensor (sensor, 1, window_len)
                if sensor in self.sensors_100hz:
                    tensor = torch.zeros((7, 1, 1000), dtype=torch.float32)
                    tensor[sensor_idx, 0, :] = torch.from_numpy(pred_raw)
                    hl, _ = extract_cycle_features(tensor, torch.zeros((2,1,100)), torch.zeros((8,1,10)), 0.0)
                    feats_238 = hl[0]
                    start, end = sensor_idx*14, (sensor_idx+1)*14
                    feats = safe_slice(feats_238, start, end, 14)
                elif sensor in self.sensors_10hz:
                    si = sensor_idx - 7
                    tensor = torch.zeros((2, 1, 100), dtype=torch.float32)
                    tensor[si, 0, :] = torch.from_numpy(pred_raw)
                    hl, _ = extract_cycle_features(torch.zeros((7,1,1000)), tensor, torch.zeros((8,1,10)), 0.0)
                    feats_238 = hl[0]
                    start, end = 7*14 + si*14, 7*14 + (si+1)*14
                    feats = safe_slice(feats_238, start, end, 14)
                else:
                    si = sensor_idx - 9
                    tensor = torch.zeros((8, 1, 10), dtype=torch.float32)
                    tensor[si, 0, :] = torch.from_numpy(pred_raw)
                    hl, _ = extract_cycle_features(torch.zeros((7,1,1000)), torch.zeros((2,1,100)), tensor, 0.0)
                    feats_238 = hl[0]
                    start, end = (7+2)*14 + si*14, (7+2)*14 + (si+1)*14
                    feats = safe_slice(feats_238, start, end, 14)
                hl_feats[idx, :] = feats

            self.lstm_pred_highlevel_features[sensor_idx] = hl_feats
            print(f"[LSTM-HL] Done for {sensor}: features shape {hl_feats.shape}, sample {hl_feats[1,:5]}")



    def __len__(self):
        # Number of possible (prev, curr) pairs, minus 1 at the end
        return self.n_windows - 1

    def __getitem__(self, idx):
        """
        idx: index of the current (present) window (use idx-1 for past)
        missing_sensor_idx: which sensor (0-16) is 'missing'
        """
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, missing_sensor_idx = idx
        else:
            raise ValueError("You must provide both idx and missing_sensor_idx as a tuple (idx, missing_sensor_idx)")

        # LSTM-predicted present window (for missing sensor)
        lstm_pred = self.lstm_pred_raw[missing_sensor_idx][idx]
        # LR-predicted present features for missing sensor
        lr_pred = self.lr_pred_features[idx, missing_sensor_idx, :]

        # Ground-truth present raw data for missing sensor
        if missing_sensor_idx < 7:
            raw_present = self.data_100[idx, missing_sensor_idx, :]
            raw_present_all = self.data_100[idx].copy()
            raw_present_all[missing_sensor_idx, :] = np.nan  # simulate missing

        elif missing_sensor_idx < 9:
            raw_present_all = self.data_10[idx].copy()
            ms = missing_sensor_idx - 7
            raw_present = self.data_10[idx, ms, :]
            raw_present_all[ms, :] = np.nan  # simulate missing
        else:
            raw_present_all = self.data_1[idx].copy()

            ms = missing_sensor_idx - 9
            raw_present = self.data_1[idx, ms, :]
            raw_present_all[ms, :] = np.nan


        # LSTM high-level features
        lstm_pred_hl_feat = self.lstm_pred_highlevel_features[missing_sensor_idx][idx]  # shape (14,)

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

    # Optionally: Save train/val/test as before, just changing how you batch/process samples as needed

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