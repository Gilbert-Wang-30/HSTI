import os
import numpy as np
import torch
from scipy.stats import skew, kurtosis

def extract_high_level_features(data_dir: str, start_idx: int, end_idx: int, return_psd0_count=False):
    """
    Load raw sensor data and extract high-level statistical features for cycles in [start_idx, end_idx].
    Returns:
        features_matrix (np.ndarray): Combined feature matrix of shape (6 * n_cycles, 170).
        rul_value (float): The RUL value shared by all selected cycles.
    Raises:
        ValueError: if the selected cycles have differing RUL values or belong to different profiles/status.
    """
    # 1. Load sensor data from text files into numpy arrays.
    # Define sensor groups by sampling rate
    sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]  # 7 sensors @100Hz
    sensors_10hz = ["FS1", "FS2"]                                       # 2 sensors @10Hz
    sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]  # 8 sensors @1Hz

    psd_zero_counter = 0

    # Load all sensor files. Each file has shape (N_cycles, time_length) as per sampling rate.
    data_100_list = []
    for sensor in sensors_100hz:
        filepath = os.path.join(data_dir, f"{sensor}.txt")
        arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 6000)
        data_100_list.append(arr)
    # Stack into array of shape (N_cycles, 7, 6000)
    data_100 = np.stack(data_100_list, axis=1)

    data_10_list = []
    for sensor in sensors_10hz:
        filepath = os.path.join(data_dir, f"{sensor}.txt")
        arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 600)
        data_10_list.append(arr)
    # Stack into array of shape (N_cycles, 2, 600)
    data_10 = np.stack(data_10_list, axis=1)

    data_1_list = []
    for sensor in sensors_1hz:
        filepath = os.path.join(data_dir, f"{sensor}.txt")
        arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 60)
        data_1_list.append(arr)
    # Stack into array of shape (N_cycles, 8, 60)
    data_1 = np.stack(data_1_list, axis=1)

    # Load RUL labels (and profiles if applicable)
    rul_path = os.path.join(data_dir, "rul_profile.txt")
    rul_array = np.loadtxt(rul_path, dtype=np.float32, delimiter=',')
    if rul_array.ndim > 1:
        rul_array = rul_array.squeeze()  # ensure 1D array of length N_cycles

    # 2. Validate the requested cycle index range
    if end_idx < start_idx or start_idx < 0 or end_idx >= len(rul_array):
        raise IndexError("Invalid cycle index range specified.")
    n_cycles = end_idx - start_idx + 1

    # Check that all selected cycles have the same RUL (and same profile/status if relevant)
    selected_ruls = rul_array[start_idx:end_idx+1, 0]
    if not np.all(selected_ruls == selected_ruls[0]):
        print("[WARNING] Selected cycles have different RUL values or profiles. Proceeding with first RUL only.")
        # raise ValueError("Selected cycles have different RUL values or system profiles; cannot merge features.")
    shared_rul = float(selected_ruls[0])

    # 3. Define a helper to split a cycle’s data into 6 equal windows for each frequency group
    def split_into_windows(cycle_data, n_windows):
        """Split a 2D array (sensors × time) into `n_windows` equal segments along time axis."""
        C, T = cycle_data.shape  # number of sensors, total time length
        window_length = T // n_windows
        windows = []
        for j in range(n_windows):
            start = j * window_length
            end = (j + 1) * window_length
            windows.append(cycle_data[:, start:end])
        # Stack into shape (sensors, n_windows, window_length)
        return np.stack(windows, axis=1)

    # 4. Define a helper to compute statistical features for a given windowed data slice
    def compute_window_features(window_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        nonlocal psd_zero_counter
        # Input shape: (sensors, windows, window_length)
        C, W, T = window_data.shape
        features = []

        for sensor in range(C):
            sensor_feats = []
            for window in range(W):
                data = window_data[sensor, window, :]

                # Time domain features
                median = np.median(data)
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                skewness = skew(data)
                kurt = kurtosis(data)
                mad = np.median(np.abs(data - median))
                zero_cross_rate = np.mean(np.diff(np.sign(data - np.mean(data))) != 0)
                autocorr_1 = np.corrcoef(data[:-1], data[1:])[0, 1] if np.std(data) > 1e-8 else 0.0


                # Frequency domain features (FFT-based)
                fft_vals = np.fft.rfft(data)
                fft_freq = np.fft.rfftfreq(len(data), d=1./sampling_rate)
                psd = np.abs(fft_vals) ** 2
                total_power = np.sum(psd)
                if total_power < 1e-8:
                    psd_zero_counter += 1
                    psd_norm = np.zeros_like(psd)
                else:
                    psd_norm = psd / total_power


                # Spectral centroid
                spec_centroid = np.sum(fft_freq * psd_norm)
                # Spectral bandwidth
                spec_bandwidth = np.sqrt(np.sum(((fft_freq - spec_centroid)**2) * psd_norm))
                # Spectral entropy
                spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
                # Spectral flatness
                spec_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd + 1e-12))
                # Frequency bands energy ratios (example split: low/mid/high equally divided)
                bands = np.array_split(psd, 3)
                band_energy_ratios = [np.sum(band) / (np.sum(psd) + 1e-8) for band in bands]

                # Combine all features into one vector per window
                window_feats = [
                    median, iqr, skewness, kurt, mad, zero_cross_rate, autocorr_1,
                    spec_centroid, spec_bandwidth, spec_entropy,
                    *band_energy_ratios, spec_flatness
                ]

                sensor_feats.append(window_feats)

            features.append(sensor_feats)

        return np.array(features)  # shape: (sensors, windows, 14 features)

    # 5. Iterate over each selected cycle and compute its feature matrix
    feature_matrices = []  # to collect feature matrices of shape (6, 170) for each cycle
    for idx in range(start_idx, end_idx + 1):
        if (idx - start_idx) % 100 == 0:
            print(f"Processed {idx - start_idx} out of {n_cycles} cycles...")
        # Get the raw data for this cycle from each frequency group
        cycle_data_100 = data_100[idx]   # shape (7, 6000) for 100Hz sensors
        cycle_data_10  = data_10[idx]    # shape (2,  600) for 10Hz sensors
        cycle_data_1   = data_1[idx]     # shape (8,   60) for 1Hz sensors

        # Split each sensor group data into 6 time windows
        windows_100 = split_into_windows(cycle_data_100, n_windows=6)  # shape (7, 6, 1000)
        windows_10  = split_into_windows(cycle_data_10,  n_windows=6)  # shape (2, 6, 100)
        windows_1   = split_into_windows(cycle_data_1,   n_windows=6)  # shape (8, 6, 10)

        # Compute statistical features for each group of windows
        feats_100 = compute_window_features(windows_100, sampling_rate=100)
        feats_10  = compute_window_features(windows_10, sampling_rate=10)
        feats_1   = compute_window_features(windows_1, sampling_rate=1)


        # Combine features from all sensor groups along the sensor axis: shape (17, 6, 10)
        combined_feats = np.concatenate([feats_100, feats_10, feats_1], axis=0)

        # Reshape combined features into a (6, 170) matrix:
        #   - Transpose to shape (6, 17, 10) so that each window (6) is first dimension
        #   - Then flatten the last two dims (17 sensors × 10 features = 170 columns)
        features_matrix = combined_feats.transpose(1, 0, 2).reshape(6, 238)
        feature_matrices.append(features_matrix)

    # 6. Merge all cycles' feature matrices into one (if only one cycle, this is just itself)
    if len(feature_matrices) > 1:
        # Stack vertically: shape (6 * n_cycles, 170)
        merged_matrix = np.vstack(feature_matrices)
    else:
        merged_matrix = feature_matrices[0]

    # 7. Return the feature matrix and the shared RUL value
    if return_psd0_count:
        return merged_matrix, shared_rul, psd_zero_counter
    return merged_matrix, shared_rul


def extract_cycle_features(
    tensor_100: torch.FloatTensor,
    tensor_10: torch.FloatTensor,
    tensor_1: torch.FloatTensor,
    rul_value: float
):
    """
    Extract statistical features from a single cycle's tensor data (already windowed).
    
    Args:
        tensor_100: Tensor of shape (7, 6, 1000) for 100Hz sensors.
        tensor_10:  Tensor of shape (2, 6, 100)  for 10Hz sensors.
        tensor_1:   Tensor of shape (8, 6, 10)   for 1Hz sensors.
        rul_value:  Remaining Useful Life (float) for this cycle.

    Returns:
        features_matrix: np.ndarray of shape (6, 170)
        rul_value: same as input
    """
    def compute_window_features(window_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        # Input shape: (sensors, windows, window_length)
        C, W, T = window_data.shape
        features = []

        for sensor in range(C):
            sensor_feats = []
            for window in range(W):
                data = window_data[sensor, window, :]

                # Time domain features
                median = np.median(data)
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                skewness = skew(data)
                kurt = kurtosis(data)
                mad = np.median(np.abs(data - median))
                zero_cross_rate = np.mean(np.diff(np.sign(data - np.mean(data))) != 0)
                autocorr_1 = np.corrcoef(data[:-1], data[1:])[0, 1] if np.std(data) > 1e-8 else 0.0


                # Frequency domain features (FFT-based)
                fft_vals = np.fft.rfft(data)
                fft_freq = np.fft.rfftfreq(len(data), d=1./sampling_rate)
                psd = np.abs(fft_vals) ** 2
                psd_norm = psd / (np.sum(psd) + 1e-8)

                # Spectral centroid
                spec_centroid = np.sum(fft_freq * psd_norm)
                # Spectral bandwidth
                spec_bandwidth = np.sqrt(np.sum(((fft_freq - spec_centroid)**2) * psd_norm))
                # Spectral entropy
                spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
                # Spectral flatness
                spec_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd + 1e-12))
                # Frequency bands energy ratios (example split: low/mid/high equally divided)
                bands = np.array_split(psd, 3)
                band_energy_ratios = [np.sum(band) / (np.sum(psd) + 1e-8) for band in bands]

                # Combine all features into one vector per window
                window_feats = [
                    median, iqr, skewness, kurt, mad, zero_cross_rate, autocorr_1,
                    spec_centroid, spec_bandwidth, spec_entropy,
                    *band_energy_ratios, spec_flatness
                ]

                sensor_feats.append(window_feats)

            features.append(sensor_feats)

        return np.array(features)  # shape: (sensors, windows, 14 features)

    # Convert torch tensors to numpy arrays
    windows_100 = tensor_100.numpy()  # shape (7, 6, 1000)
    windows_10  = tensor_10.numpy()   # shape (2, 6, 100)
    windows_1   = tensor_1.numpy()    # shape (8, 6, 10)

    # Compute features
    feats_100 = compute_window_features(windows_100, sampling_rate=100)
    feats_10  = compute_window_features(windows_10, sampling_rate=10)
    feats_1   = compute_window_features(windows_1, sampling_rate=1)
    # Combine and reshape to (6, 170)
    combined = np.concatenate([feats_100, feats_10, feats_1], axis=0)
    features_matrix = combined.transpose(1, 0, 2).reshape(6, -1).T  # shape: (170, 6)

    return features_matrix, rul_value

# Example usage (not part of the module, shown for clarity):
# data_dir = "path/to/sensor_data_folder"
# X, rul_val = extract_high_level_features(data_dir, start_idx=0, end_idx=9)
# print("Feature matrix shape:", X.shape)  # Expect (60, 170) for 10 cycles
# print("Shared RUL value:", rul_val)
if __name__ == "__main__":
    import argparse
    import pickle
        # Example usage
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=2204)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    features, rul_value, psd_zero_count = extract_high_level_features(
        data_dir, args.start_idx, args.end_idx, return_psd0_count=True
    )
    print("Feature matrix shape:", features.shape)
    print("Shared RUL value:", rul_value)
    print(f"Total windows with all-zero PSD: {psd_zero_count}")

    with open(args.output, "wb") as f:
        pickle.dump({"features": features, "rul": rul_value}, f)
    print(f"Saved features to: {args.output}")