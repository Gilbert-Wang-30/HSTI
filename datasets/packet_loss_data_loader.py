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

    dataset = SensorWindowDataset(raw_data_dir, features_path=features_path)

    idx = 100  # try an arbitrary sample
    for missing_sensor_idx in [0, 7, 13, 16]:
        input_dict, target_dict = dataset[idx, missing_sensor_idx]
        print(f"\nSample idx={idx}, missing_sensor={dataset.all_sensors[missing_sensor_idx]}")
        print(f"  Missing sensor ID: {input_dict['missing_sensor_id']}")
        print(f"  features_past shape: {input_dict['features_past'].shape}")
        print(f"  features_present_excl_missing shape: {input_dict['features_present_excl_missing'].shape}")
        print(f"  raw_past 100Hz shape: {input_dict['raw_past']['data_100'].shape}")
        print(f"  raw_present_excl_missing 1Hz shape: {input_dict['raw_present_excl_missing']['data_1'].shape}")
        print(f"  target features (missing): {target_dict['features_present_missing'].shape}")
        print(f"  target raw (missing): {target_dict['raw_present_missing'].shape}")


    for ms_idx in [0, 7, 13, 16]:
        input_dict, target_dict = dataset[(idx, ms_idx)]
        print(f"\n[{dataset.all_sensors[ms_idx]}] Is missing sensor's raw data all NaN?")
        if ms_idx < 7:
            print(np.isnan(input_dict['raw_present_excl_missing']['data_100'][ms_idx, :]).all())
        elif ms_idx < 9:
            print(np.isnan(input_dict['raw_present_excl_missing']['data_10'][ms_idx-7, :]).all())
        else:
            print(np.isnan(input_dict['raw_present_excl_missing']['data_1'][ms_idx-9, :]).all())

