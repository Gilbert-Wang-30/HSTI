import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, random_split
import pickle
from pathlib import Path
import sys

# (If needed, adjust sys.path or BASE_DIR as in the original data_loader for consistent imports/paths)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

class missing_data_loader(Dataset):
    def __init__(self, data_dir, sensor_name ="TS1"):
        # Define sensor groups by sampling rate (same grouping as original data_loader)
        self.sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
        self.sensors_10hz = ["FS1", "FS2"]
        self.sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
        
        # Load sensor data from text files
        data_100_list = []
        for sensor in self.sensors_100hz:
            filepath = os.path.join(data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 6000)
            data_100_list.append(arr)
        self.data_100 = np.stack(data_100_list, axis=1)  # shape: (N_cycles, 7, 6000)
        
        data_10_list = []
        for sensor in self.sensors_10hz:
            filepath = os.path.join(data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 600)
            data_10_list.append(arr)
        self.data_10 = np.stack(data_10_list, axis=1)   # shape: (N_cycles, 2, 600)
        
        data_1_list = []
        for sensor in self.sensors_1hz:
            filepath = os.path.join(data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 60)
            data_1_list.append(arr)
        self.data_1 = np.stack(data_1_list, axis=1)     # shape: (N_cycles, 8, 60)
        
        N = self.data_100.shape[0]  # number of cycles
        # **Simulate missing sensor data**: 
        # Extract the last reading of sensor for each cycle as the target, then set it as NaN in the input data.
        if sensor_name in self.sensors_1hz:
            freq_group = '1hz'
            sensor_list = self.sensors_1hz
            ts_index = sensor_list.index(sensor_name)
            channel_data = self.data_1
            reshape_dims = (N, 8, 6, 10)
        elif sensor_name in self.sensors_10hz:
            freq_group = '10hz'
            sensor_list = self.sensors_10hz
            ts_index = sensor_list.index(sensor_name)
            channel_data = self.data_10
            reshape_dims = (N, 2, 6, 100)
        elif sensor_name in self.sensors_100hz:
            freq_group = '100hz'
            sensor_list = self.sensors_100hz
            ts_index = sensor_list.index(sensor_name)
            channel_data = self.data_100
            reshape_dims = (N, 7, 6, 1000)
        else:
            raise ValueError(f"Sensor {sensor_name} not found in known groups")
        
        # Reshape each sensor data array into 6 time-window segments (as in original):

        # After loading data_xxx with shapes (N, sensors, T_total)
        N = self.data_100.shape[0]  # number of cycles

        # 100Hz: reshape to (N*6, 7, 1000)
        self.tensor_100 = torch.from_numpy(
            self.data_100.reshape(N, 7, 6, 1000)
        ).permute(0, 2, 1, 3).reshape(-1, 7, 1000)

        # 10Hz: (N*6, 2, 100)
        self.tensor_10 = torch.from_numpy(
            self.data_10.reshape(N, 2, 6, 100)
        ).permute(0, 2, 1, 3).reshape(-1, 2, 100)

        # 1Hz: (N*6, 8, 10)
        self.tensor_1 = torch.from_numpy(
            self.data_1.reshape(N, 8, 6, 10)
        ).permute(0, 2, 1, 3).reshape(-1, 8, 10)


    def __len__(self):
        # Number of windows (minus 1 for next-window pairing)
        return self.tensor_1.shape[0] - 1

    def __getitem__(self, idx):
        # Retrieve the precomputed sensor data tensors for this cycle
        x100 = self.tensor_100[idx]   # shape: (7, 6, 1000)
        x10  = self.tensor_10[idx]    # shape: (2, 6, 100)
        x1   = self.tensor_1[idx]     # shape: (8, 6, 10)
        # Target is the true last sensor reading for this cycle
        y100 = self.tensor_100[idx + 1]  # shape: (7, 6, 1000)
        y10  = self.tensor_10[idx + 1]   # shape: (2, 6, 100)
        y1   = self.tensor_1[idx + 1]    # shape: (8, 6, 10)
        return (x100, x10, x1), (y100, y10, y1)

# If run as a script, demonstrate dataset usage and create train/dev/test splits
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", type=str, default="TS1", help="1Hz sensor to mask (e.g., TS1)")
    args = parser.parse_args()
    sensor = args.sensor.upper()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)

    dataset = missing_data_loader(data_dir, sensor)
    
    # Quick test on a few samples to verify shapes and NaN replacement
    for i in (0, 1, 220):
        (x100, x10, x1), (y100, y10, y1) = dataset[i]
        print(f"\n[Sample {i}]")
        print(f"  x100 shape: {x100.shape} (expect (7, 1000))")
        print(f"  y100 shape: {y100.shape} (expect (7, 1000))")
        print(f"  x10  shape: {x10.shape}  (expect (2, 100))")
        print(f"  y10  shape: {y10.shape}  (expect (2, 100))")
        print(f"  x1   shape: {x1.shape}   (expect (8, 10))")
        print(f"  y1   shape: {y1.shape}   (expect (8, 10))")
        print("First row of x1, first feature:", x1[0, :5].numpy())
        print("First row of y1, first feature:", y1[0, :5].numpy())


    # Compute split lengths for train (70%), dev (20%), and test (10%)
    total = len(dataset)
    train_len = int(total * 0.7)
    dev_len   = int(total * 0.2)
    test_len  = total - train_len - dev_len
    # Fixed seed for reproducibility (same split each run)
    generator = torch.Generator().manual_seed(42)
    # Split the dataset
    train_set, dev_set, test_set = random_split(dataset, [train_len, dev_len, test_len], generator=generator)
    
    # Save the splits as pickle files
    # Save the splits as pickle files using sensor name prefix
    def save_dataset(obj, name):
        path = os.path.join(output_dir, f"{sensor.lower()}_{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    print(f"\nSaving datasets to {output_dir}...")

    save_dataset(train_set, "train")
    save_dataset(dev_set, "dev")
    save_dataset(test_set, "test")

    # List the output file paths (for verification)
    output_paths = [
        os.path.join(output_dir, f"{sensor.lower()}_{suffix}.pkl")
        for suffix in ["train", "dev", "test"]
    ]
    print("Saved split datasets to:", output_paths)
