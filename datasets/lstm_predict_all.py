import os
import sys
import numpy as np
import torch
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.LSTM import LSTMModel
from features.high_level_feature_extraction import extract_cycle_features  # adjust import as needed


def compute_all_lstm_predictions(data_100, data_10, data_1, model_dir, device="cpu"):
    BATCH_SIZE_PRED = 128  # Or 256, 512, tune as you like (fit to your GPU)

    sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
    sensors_10hz = ["FS1", "FS2"]
    sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]

    n_windows = data_100.shape[0]
    lstm_100 = np.full((n_windows-1, 7, 1000), np.nan, dtype=np.float32)
    lstm_10  = np.full((n_windows-1, 2, 100), np.nan, dtype=np.float32)
    lstm_1   = np.full((n_windows-1, 8, 10), np.nan, dtype=np.float32)

    sensor_configs = [
        (sensors_100hz, data_100, lstm_100, 1000),
        (sensors_10hz,  data_10,  lstm_10,  100),
        (sensors_1hz,   data_1,   lstm_1,    10)
    ]

    print(f"[LSTM-PRED] Starting all predictions, device={device}")


    
    for sensors, data_arr, lstm_arr, win_len in sensor_configs:
        n_sensors = len(sensors)

        for si, sensor in enumerate(sensors):
            # Load model and normalization stats
            model_path = Path(model_dir) / f"{sensor.lower()}_lstm.pth"
            norm_path = Path(model_dir) / f"{sensor.lower()}_lstm_norm_stats.npz"
            lstm = LSTMModel(input_size=1, hidden_size=128, num_layers=3)
            lstm.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            lstm.to(device)
            lstm.eval()
            norm = np.load(norm_path)
            norm_mean, norm_std = float(norm["mean"]), float(norm["std"])
            print(f"  [{sensor}] Model loaded. Predicting...")

            # Batched prediction for this sensor: (n_windows-1, win_len)
            input_windows = data_arr[:-1, si, :].astype(np.float32)
            input_windows_norm = (input_windows - norm_mean) / (norm_std + 1e-8)
            num_windows = input_windows_norm.shape[0]
            preds_all = []
            preds_all.append(np.zeros((win_len), dtype=np.float32))  # Preallocate
            for i in range(0, num_windows):
                pred_norm = lstm(input_windows_norm[i])  # (win_len)
                pred_real = pred_norm.cpu().numpy().squeeze(-1) * norm_std + norm_mean  # (win_len)
                preds_all.append(pred_real)

            preds_all = np.concatenate(preds_all, axis=0)  # (num_windows, win_len)
            print(f"  [{sensor}] Predictions shape: {preds_all.shape}")
            lstm_arr[:, si, :] = preds_all
            print(f"  [{sensor}] Example window 1: {lstm_arr[1, si, :5]}")
            
        print(f"[LSTM-PRED] Finished for group: {sensors}")
    return lstm_100, lstm_10, lstm_1

def extract_highlevel_from_lstm_preds(lstm_100, lstm_10, lstm_1, debug_print=False):
    """
    Args:
        lstm_100: np.ndarray of shape (n_windows, 7, 1000)
        lstm_10:  np.ndarray of shape (n_windows, 2, 100)
        lstm_1:   np.ndarray of shape (n_windows, 8, 10)
    Returns:
        all_features: np.ndarray of shape (n_windows, 17, 14)
    """
    # First, reshape back to cycle-wise (cycle, window, sensor, freq)
    n_windows, n_100_sensors, win_1000 = lstm_100.shape
    n_cycles = n_windows // 6
    assert n_windows % 6 == 0, "n_windows must be divisible by 6"

    lstm_100_cw = lstm_100.reshape(n_cycles, 6, n_100_sensors, win_1000).transpose(0,2,1,3)  # (n_cycles, 7, 6, 1000)
    lstm_10_cw  = lstm_10.reshape(n_cycles, 6, 2, 100).transpose(0,2,1,3)   # (n_cycles, 2, 6, 100)
    lstm_1_cw   = lstm_1.reshape(n_cycles, 6, 8, 10).transpose(0,2,1,3)     # (n_cycles, 8, 6, 10)

    all_feats = []
    for cyc in range(n_cycles):
        if debug_print and cyc % 100 == 0:
            print(f"[DEBUG] Processing cycle {cyc}/{n_cycles}...")
        tensor_100 = torch.from_numpy(lstm_100_cw[cyc])
        tensor_10  = torch.from_numpy(lstm_10_cw[cyc])
        tensor_1   = torch.from_numpy(lstm_1_cw[cyc])
        feats, _ = extract_cycle_features(tensor_100, tensor_10, tensor_1, rul_value=0.0)  # shape (6, 238)
        if debug_print and cyc < 2:
            print(f"[DEBUG] Cycle {cyc}: features shape={feats.shape}, sample={feats[:, :5]}")
        all_feats.append(feats)  # shape (238, 6)

    # Stack to (n_cycles, 238, 6) -> reshape to (n_cycles*6, 238)
    all_feats = np.stack(all_feats, axis=0)  # (n_cycles, 238, 6)
    all_feats = all_feats.transpose(0, 2, 1)  # (n_cycles, 6, 238)
    all_feats = all_feats.reshape(-1, all_feats.shape[2])  # (n_windows, 6)

    return all_feats

if __name__ == "__main__":
    # --- Setup ---
    import os
    import sys
    
    ROOT = Path(__file__).resolve().parent.parent  # or .parent if your script is in the root, adjust as needed

    raw_data_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "data" / "lstm_predictions"
    model_dir = ROOT / "models"
    features_path = ROOT / "features" / "features.pkl"
    READ_FROM_DISK = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(out_dir, exist_ok=True)
    sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
    sensors_10hz = ["FS1", "FS2"]
    sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]

    print(f"[INFO] Sensors: 100Hz={sensors_100hz}, 10Hz={sensors_10hz}, 1Hz={sensors_1hz}")
    # --- Load and window data ---
    # 100 Hz
    data_100 = [np.loadtxt(os.path.join(raw_data_dir, f"{s}.txt"), delimiter='\t', dtype=np.float32) for s in sensors_100hz]
    data_100 = np.stack(data_100, axis=1)
    N_cycles = data_100.shape[0]
    data_100 = data_100.reshape(N_cycles, 7, 6, 1000).transpose(0,2,1,3).reshape(-1, 7, 1000)
    # 10 Hz
    data_10 = [np.loadtxt(os.path.join(raw_data_dir, f"{s}.txt"), delimiter='\t', dtype=np.float32) for s in sensors_10hz]
    data_10 = np.stack(data_10, axis=1)
    data_10 = data_10.reshape(N_cycles, 2, 6, 100).transpose(0,2,1,3).reshape(-1, 2, 100)
    # 1 Hz
    data_1 = [np.loadtxt(os.path.join(raw_data_dir, f"{s}.txt"), delimiter='\t', dtype=np.float32) for s in sensors_1hz]
    data_1 = np.stack(data_1, axis=1)
    data_1 = data_1.reshape(N_cycles, 8, 6, 10).transpose(0,2,1,3).reshape(-1, 8, 10)

    n_windows = data_100.shape[0]
    print(f"[INFO] Windowed data shapes: 100Hz: {data_100.shape}, 10Hz: {data_10.shape}, 1Hz: {data_1.shape}")

    # --- Run LSTM predictions ---
    lstm_100_path = os.path.join(out_dir, "lstm_100.npy")
    lstm_10_path  = os.path.join(out_dir, "lstm_10.npy")
    lstm_1_path   = os.path.join(out_dir, "lstm_1.npy")
    lstm_feat_path = os.path.join(out_dir, "lstm_features.npy")


    if READ_FROM_DISK and os.path.exists(lstm_100_path) and os.path.exists(lstm_10_path) and os.path.exists(lstm_1_path):
        print("[INFO] LSTM predictions found on disk. Loading from file.")
        lstm_100 = np.load(lstm_100_path)
        lstm_10  = np.load(lstm_10_path)
        lstm_1   = np.load(lstm_1_path)
    else:
        print("[INFO] LSTM predictions not found. Running LSTM inference...")
        lstm_100, lstm_10, lstm_1 = compute_all_lstm_predictions(data_100, data_10, data_1, model_dir, device=device)
        np.save(lstm_100_path, lstm_100)
        np.save(lstm_10_path, lstm_10)
        np.save(lstm_1_path,  lstm_1)
        print(f"[INFO] Saved LSTM predictions to {out_dir}")

    if READ_FROM_DISK and os.path.exists(lstm_feat_path):
        print("[INFO] High-level features already computed. Loading from file.")
        features_matrix = np.load(lstm_feat_path)
    else:
        # Assuming lstm_100, lstm_10, lstm_1 are loaded and in (n_windows, n_sensors, freq) shape
        features_matrix = extract_highlevel_from_lstm_preds(lstm_100, lstm_10, lstm_1, debug_print=True)
        print("Extracted feature matrix shape:", features_matrix.shape)  # (n_windows, 238)
        np.save(lstm_feat_path, features_matrix)
        print(f"[INFO] Saved high-level features to {lstm_feat_path}")

    # --- Compare LSTM predictions to ground truth ---
    print("\n--- LSTM Prediction vs Ground Truth MSE (per sensor) ---")
    for si, sensor in enumerate(sensors_100hz):
        mse = np.mean((data_100[1:, si, :] - lstm_100[1:, si, :]) ** 2, axis=1)
        print(f"{sensor:>5} MSE: min={np.min(mse):.4f} | max={np.max(mse):.4f} | median={np.median(mse):.4f} | mean={np.mean(mse):.4f}")
    for si, sensor in enumerate(sensors_10hz):
        mse = np.mean((data_10[1:, si, :] - lstm_10[1:, si, :]) ** 2, axis=1)
        print(f"{sensor:>5} MSE: min={np.min(mse):.4f} | max={np.max(mse):.4f} | median={np.median(mse):.4f} | mean={np.mean(mse):.4f}")
    for si, sensor in enumerate(sensors_1hz):
        mse = np.mean((data_1[1:, si, :] - lstm_1[1:, si, :]) ** 2, axis=1)
        print(f"{sensor:>5} MSE: min={np.min(mse):.4f} | max={np.max(mse):.4f} | median={np.median(mse):.4f} | mean={np.mean(mse):.4f}")

    # --- Debug for high-level features ---
    print(f"Extracted features shape: {features_matrix.shape}")  # Should be (n_windows, 238)
    print("Any NaN in features?:", np.isnan(features_matrix).any())
    print("Number of NaNs:", np.isnan(features_matrix).sum())
    print("Feature-wise NaN counts:", np.isnan(features_matrix).sum(axis=0))  # Per feature

    print("Feature matrix (first 3 rows):\n", features_matrix[:3])

    # Basic statistics
    print("Mean per feature (first 10):", np.mean(features_matrix, axis=0)[:10])
    print("Std per feature (first 10):", np.std(features_matrix, axis=0)[:10])
    print("Min/Max overall:", np.min(features_matrix), np.max(features_matrix))



    # idx: the flat window index (0 <= idx < n_windows-1)
    def debug_compare_one(idx, raw_data_100, raw_data_10, raw_data_1, feats_matrix, extract_fn):
        n_sensors_100, n_sensors_10, n_sensors_1 = 7, 2, 8

        # Figure out which cycle and window
        cycle_idx = idx // 6
        window_idx = idx % 6

        # Reconstruct the full cycle for this index (shape: (n_sensors, 6, freq))
        d100 = raw_data_100[cycle_idx*6:(cycle_idx+1)*6].transpose(1,0,2)  # (7,6,1000)
        d10  = raw_data_10[cycle_idx*6:(cycle_idx+1)*6].transpose(1,0,2)   # (2,6,100)
        d1   = raw_data_1[cycle_idx*6:(cycle_idx+1)*6].transpose(1,0,2)    # (8,6,10)

        # Compute features from raw data
        feats, _ = extract_fn(
            torch.from_numpy(d100),
            torch.from_numpy(d10),
            torch.from_numpy(d1),
            0.0
        )  # feats: (238, 6)

        print(f"\n[Window idx={idx} | Cycle {cycle_idx}, Window {window_idx}]")
        print("Extracted features from saved matrix:", feats_matrix[idx, :10])
        print("Freshly computed features:", feats[:, window_idx][:10])
        diff = np.abs(feats_matrix[idx] - feats[:, window_idx])
        print("Max diff:", np.max(diff))
        print("Mean diff:", np.mean(diff))

    # Example usage (after all_feats computed)
    debug_compare_one(100, data_100, data_10, data_1, features_matrix, extract_cycle_features)
