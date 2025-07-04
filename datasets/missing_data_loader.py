import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pickle
from pathlib import Path
import sys

# (If needed, adjust sys.path or BASE_DIR as in the original data_loader for consistent imports/paths)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

class missing_data_loader(Dataset):
    def __init__(self, data_dir):
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
        
        # **Simulate missing TS1 data**: 
        # Extract the last reading of TS1 for each cycle as the target, then set it as NaN in the input data.
        ts1_index = 0  # TS1 is the first sensor in the 1Hz list
        # Copy the last values of TS1 (to use as targets) 
        ts1_last_values = self.data_1[:, ts1_index, -1].astype(np.float32).copy()  # shape: (N_cycles,)
        # Set those last readings in the input data to NaN to simulate missing data
        self.data_1[:, ts1_index, -1] = np.nan
        
        # Reshape each sensor data array into 6 time-window segments (as in original):
        N = self.data_100.shape[0]  # number of cycles
        # 100 Hz sensors: reshape 6000 samples into (6 windows × 1000 samples)
        self.tensor_100 = torch.from_numpy(self.data_100.reshape(N, 7, 6, 1000))
        # 10 Hz sensors: reshape 600 samples into (6 windows × 100 samples)
        self.tensor_10  = torch.from_numpy(self.data_10.reshape(N, 2, 6, 100))
        # 1 Hz sensors: reshape 60 samples into (6 windows × 10 samples)
        self.tensor_1   = torch.from_numpy(self.data_1.reshape(N, 8, 6, 10))
        
        # Convert target TS1 last values to a torch tensor
        self.ts1_last = torch.from_numpy(ts1_last_values)  # shape: (N_cycles,)
    
    def __len__(self):
        return len(self.ts1_last)  # number of cycles
    
    def __getitem__(self, idx):
        # Retrieve the precomputed sensor data tensors for this cycle
        x100 = self.tensor_100[idx]   # shape: (7, 6, 1000)
        x10  = self.tensor_10[idx]    # shape: (2, 6, 100)
        x1   = self.tensor_1[idx]     # shape: (8, 6, 10)
        # Target is the true last TS1 reading for this cycle
        ts1_target = self.ts1_last[idx]  # shape: (), a 0-dim tensor (scalar)
        return (x100, x10, x1), ts1_target

# If run as a script, demonstrate dataset usage and create train/dev/test splits
if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the dataset
    dataset = missing_data_loader(data_dir)
    
    # Quick test on a few samples to verify shapes and NaN replacement
    for i in (0, 1, 220):
        (x100, x10, x1), ts1_val = dataset[i]
        print(f"\n[Sample {i}]")
        print(f"  x100 shape           : {x100.shape} (expect (7, 6, 1000))")
        print(f"  x10  shape           : {x10.shape}  (expect (2, 6, 100))")
        print(f"  x1   shape           : {x1.shape}   (expect (8, 6, 10))")
        # Check that the last TS1 reading in input is NaN and the target is the actual value
        ts1_input_last = x1[0, -1, -1].item()  # this is the last TS1 value in the input (should be NaN)
        print(f"  TS1 last input value : {ts1_input_last} (should be nan)")
        print(f"  TS1 target value     : {ts1_val.item()}")  # actual last reading that we aim to predict
    
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
    def save_dataset(obj, name):
        path = os.path.join(output_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    save_dataset(train_set, "ts1_train")
    save_dataset(dev_set, "ts1_dev")
    save_dataset(test_set, "ts1_test")
    
    # List the output file paths (for verification)
    output_paths = [os.path.join(output_dir, fname) for fname in ["ts1_train.pkl", "ts1_dev.pkl", "ts1_test.pkl"]]
    print("\nSaved split datasets to:", output_paths)
