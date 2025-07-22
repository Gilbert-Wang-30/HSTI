import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pickle
from pathlib import Path
import sys

class SensorWindowDataset(Dataset):
    def __init__(self, raw_data_dir, features_path=None):
        # Sensor groups by sampling rate (17 sensors total)
        self.sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
        self.sensors_10hz = ["FS1", "FS2"]
        self.sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
        
        # Load all sensor data files into numpy arrays
        data_100_list = []
        for sensor in self.sensors_100hz:
            filepath = os.path.join(raw_data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape (N_cycles, 6000)
            data_100_list.append(arr)
        # Stack into one array: shape (N_cycles, 7, 6000)
        self.data_100 = np.stack(data_100_list, axis=1)
        
        data_10_list = []
        for sensor in self.sensors_10hz:
            filepath = os.path.join(raw_data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape (N_cycles, 600)
            data_10_list.append(arr)
        # Shape (N_cycles, 2, 600)
        self.data_10 = np.stack(data_10_list, axis=1)
        
        data_1_list = []
        for sensor in self.sensors_1hz:
            filepath = os.path.join(raw_data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape (N_cycles, 60)
            data_1_list.append(arr)
        # Shape (N_cycles, 8, 60)
        self.data_1 = np.stack(data_1_list, axis=1)

        # reshape raw data

        # 100Hz sensors: (N_cycles, 7, 6000) → (N_cycles*6, 7, 1000)
        N_cycles = self.data_100.shape[0]
        self.data_100 = self.data_100.reshape(N_cycles, 7, 6, 1000)    # (N_cycles, 7, 6, 1000)
        self.data_100 = self.data_100.transpose(0, 2, 1, 3)            # (N_cycles, 6, 7, 1000)
        self.data_100 = self.data_100.reshape(-1, 7, 1000)             # (N_cycles*6, 7, 1000)

        # 10Hz sensors: (N_cycles, 2, 600) → (N_cycles*6, 2, 100)
        self.data_10 = self.data_10.reshape(N_cycles, 2, 6, 100)
        self.data_10 = self.data_10.transpose(0, 2, 1, 3)
        self.data_10 = self.data_10.reshape(-1, 2, 100)

        # 1Hz sensors: (N_cycles, 8, 60) → (N_cycles*6, 8, 10)
        self.data_1 = self.data_1.reshape(N_cycles, 8, 6, 10)
        self.data_1 = self.data_1.transpose(0, 2, 1, 3)
        self.data_1 = self.data_1.reshape(-1, 8, 10)

        assert self.data_100.shape[0] == self.data_10.shape[0] == self.data_1.shape[0], "All sensor data should have the same number of cycles"
        assert self.data_100.shape[0] == N_cycles * 6, "Data should be reshaped to 6 windows per cycle"
        assert self.data_100.shape[1] == 7 and self.data_10.shape[1] == 2 and self.data_1.shape[1] == 8, "Sensor dimensions should match expected counts"


        # load features
        with open(features_path, "rb") as f:
            self.features = pickle.load(f)
        self.features = self.features["features"] # shape (N_cycles, 6, 17, 14)

        # transpose to (N_cycles*6, 17, 14)
        self.features = self.features.reshape(-1, 17, 14)
        assert self.features.shape[0] == self.data_100.shape[0], "Features should match number of cycles"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Return all 3 bands of raw data and window-level features for this window
        data_100 = torch.from_numpy(self.data_100[idx])    # (7, 1000)
        data_10 = torch.from_numpy(self.data_10[idx])      # (2, 100)
        data_1 = torch.from_numpy(self.data_1[idx])        # (8, 10)
        features = torch.from_numpy(self.features[idx])    # (17, 14)
        # Return a tuple (raw_100, raw_10, raw_1, features)
        return data_100, data_10, data_1, features

# ------------------------
# Main logic for splitting/saving
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    raw_data_dir = BASE_DIR / "data" / "raw"
    features_path = BASE_DIR / "features" / "features.pkl"
    output_dir = BASE_DIR / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = SensorWindowDataset(raw_data_dir, features_path=features_path)
    print(f"Total windows: {len(dataset)}")

    # Example inspection
    for i in [0, 1, 100]:
        d100, d10, d1, feat = dataset[i]
        print(f"Sample {i}: d100 {d100.shape}, d10 {d10.shape}, d1 {d1.shape}, features {feat.shape}")

    # Split (80/10/10) with reproducibility
    total = len(dataset)
    train_len = int(total * 0.8)
    val_len = int(total * 0.1)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    # Save datasets as pickled torch datasets
    for split, name in zip([train_set, val_set, test_set], ["train", "val", "test"]):
        out_path = output_dir / f"{name}_loss.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(split, f)
        print(f"Saved {name} set to {out_path}")
