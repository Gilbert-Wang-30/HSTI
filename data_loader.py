import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pickle


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
        self.rul = np.loadtxt(rul_path, dtype=np.float32)  # shape (N_cycles,)
        # Ensure RUL is 1D array of length N_cycles.
        if self.rul.ndim > 1:
            self.rul = self.rul.squeeze()  # flatten to 1D if needed

    def __len__(self):
        return len(self.rul)  # number of cycles
    
    def __getitem__(self, idx):
        # Extract one cycle of data for each frequency band
        cycle_100 = self.data_100[idx]   # shape (7, 6000)
        cycle_10  = self.data_10[idx]    # shape (2,  600)
        cycle_1   = self.data_1[idx]     # shape (8,   60)
        
        # Split each into 6 windows of equal length along the time dimension
        windows_100 = [cycle_100[:, j*1000:(j+1)*1000] for j in range(6)]
        windows_100 = np.stack(windows_100, axis=1)   # shape (7, 6, 1000)

        windows_10  = [cycle_10[:, j*100:(j+1)*100] for j in range(6)]
        windows_10  = np.stack(windows_10, axis=1)    # shape (2, 6, 100)

        windows_1   = [cycle_1[:, j*10:(j+1)*10] for j in range(6)]
        windows_1   = np.stack(windows_1, axis=1)     # shape (8, 6, 10)

            
        # Convert to torch.FloatTensor
        tensor_100 = torch.from_numpy(windows_100)  # shape (7, 6, 1000)
        tensor_10  = torch.from_numpy(windows_10)   # shape (2, 6, 100)
        tensor_1   = torch.from_numpy(windows_1)    # shape (8, 6, 10)
        
        # print(f"Cycle {idx}: 100Hz shape {tensor_100.shape}, 10Hz shape {tensor_10.shape}, 1Hz shape {tensor_1.shape}")
        
        rul_value  = torch.tensor(self.rul[idx], dtype=torch.float32)  # scalar
        # print(f"Cycle {idx}: RUL value {rul_value.item()}")
        
            # Print the first sample of each tensor for debugging
        # print(f"Cycle {idx}: ")
        # print(f"100Hz sample {tensor_100[0, :, :5]}, ")
        # print(f"10Hz sample {tensor_10[0, :, :5]}, ")
        # print(f"1Hz sample {tensor_1[0, :, :5]}")
        # Return a tuple of the three tensors and the RUL label
        return (tensor_100, tensor_10, tensor_1), rul_value
    
if __name__ == "__main__":
    data_dir = "/home/wangyuxiao/project/gilbert_copy/HSTI/data/"
    output_dir = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    dataset = data_loader(data_dir)

    # Test first 1 samples
    for i in range(1):
        x, y = dataset[i]

    # Compute split lengths
    total = len(dataset)
    train_len = int(total * 0.7)
    val_len = int(total * 0.2)
    test_len = total - train_len - val_len

    # Split dataset
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

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