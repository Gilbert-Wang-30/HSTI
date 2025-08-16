import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pickle
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from features.high_level_feature_extraction import extract_cycle_features, extract_high_level_features

class data_loader(Dataset):
    def __init__(self, data_dir):
        # Sensor groups by sampling rate (17 sensors total)
        self.sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
        self.sensors_10hz = ["FS1", "FS2"]
        self.sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
        
        # Load all sensor data files into numpy arrays
        data_100_list = []
        for sensor in self.sensors_100hz:
            filepath = os.path.join(data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape (N_cycles, 6000)
            data_100_list.append(arr)
        # Stack into one array: shape (N_cycles, 7, 6000)
        self.data_100 = np.stack(data_100_list, axis=1)
        
        data_10_list = []
        for sensor in self.sensors_10hz:
            filepath = os.path.join(data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape (N_cycles, 600)
            data_10_list.append(arr)
        # Shape (N_cycles, 2, 600)
        self.data_10 = np.stack(data_10_list, axis=1)
        
        data_1_list = []
        for sensor in self.sensors_1hz:
            filepath = os.path.join(data_dir, f"{sensor}.txt")
            arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape (N_cycles, 60)
            data_1_list.append(arr)
        # Shape (N_cycles, 8, 60)
        self.data_1 = np.stack(data_1_list, axis=1)
        
        # Load RUL labels (assuming a single-column text file)
        rul_path = os.path.join(data_dir, "rul_profile.txt")
        raw_target = np.loadtxt(rul_path, dtype=np.float32, delimiter=',')
        rul_array = raw_target[:, 0]  # Extract the first column (RUL values)
        status_array = raw_target[:, 1:5]  # Extract the status columns (not used here)
        assert rul_array.ndim == 1, "RUL array should be 1D"
        assert status_array.ndim == 2 and status_array.shape[1] == 4
        # Ensure RUL is 1D array of length N_cycles.
        if rul_array.ndim > 1:
            rul_array = rul_array.squeeze()  # flatten to 1D if needed
        self.rul = torch.from_numpy(rul_array)  # Convert to torch tensor (shape: (N_cycles,))
        self.status = torch.from_numpy(status_array)  # Convert status to tensor (shape: (N_cycles, 4))

        # Pre-compute windowed tensors for each frequency band
        N = self.data_100.shape[0]  # number of cycles
        features_matrix, _ = extract_high_level_features(data_dir, start_idx=0, end_idx=N-1)
        # `features_matrix` shape is (6 * N, 238) because the function stacks 6 windows per cycle vertically.
        # Reshape it to (N, 6, 238) where each cycle has 6 windows × 238 features:
        features_matrix = features_matrix.reshape(N, 6, 238)
        # Transpose to shape (N, 238, 6) so that each cycle's feature matrix matches (238 features × 6 windows):
        features_matrix = features_matrix.transpose(0, 2, 1)
        # Convert to torch tensor
        self.features = torch.from_numpy(features_matrix.astype(np.float32))
        
        # Pre-compute windowed tensors for raw sensor data (as before, for completeness)
        self.tensor_100 = torch.from_numpy(self.data_100.reshape(N, 7, 6, 1000))
        self.tensor_10  = torch.from_numpy(self.data_10.reshape(N, 2, 6, 100))
        self.tensor_1   = torch.from_numpy(self.data_1.reshape(N, 8, 6, 10))


    def __len__(self):
        return len(self.rul)  # number of cycles
    
    def __getitem__(self, idx):
        # Simply index into precomputed tensors and features
        tensor_100 = self.tensor_100[idx]   # shape: (7, 6, 1000)
        tensor_10  = self.tensor_10[idx]    # shape: (2, 6, 100)
        tensor_1   = self.tensor_1[idx]     # shape: (8, 6, 10)
        features   = self.features[idx]     # shape: (feature_dim, 6)
        rul_value  = self.rul[idx]          # torch scalar (0-dim tensor) for the RUL
        status_value = self.status[idx]      # shape: (4,) for the status

        # Return the tuple of sensor tensors, the features, and the RUL label
        return (tensor_100, tensor_10, tensor_1), features, rul_value, status_value
    
if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)
    dataset = data_loader(data_dir)

# Test first 3 samples
    for i in (0, 1, 220):
        (x100, x10, x1), features, rul, status = dataset[i]

        print(f"\n[Sample {i}]")
        print(f"  x100 shape    : {x100.shape} (expect (7, 6, 1000))")
        print(f"  x10  shape    : {x10.shape}  (expect (2, 6, 100))")
        print(f"  x1   shape    : {x1.shape}   (expect (8, 6, 10))")
        print(f"  features shape: {features.shape} (expect (170, 6))")
        print(f"  status shape  : {status.shape} (expect (4,))")
        print(f"  RUL value     : {rul.item()}")

    # Compute split lengths
    total = len(dataset)
    train_len = int(total * 0.7)
    val_len = int(total * 0.2)
    test_len = total - train_len - val_len

    # Set fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Split dataset
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    # Save function
    def save_dataset(obj, name):
        path = os.path.join(output_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    save_dataset(train_set, "train")
    save_dataset(val_set, "val")
    save_dataset(test_set, "test")

    output_paths = [os.path.join(output_dir, f"{name}.pkl") for name in ["train", "val", "test"]]
    output_paths