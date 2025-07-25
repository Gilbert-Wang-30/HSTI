import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from models.LSTM import LSTMModel

def extract_sensor_sequences(data_dir, sensor, window_in=59, window_out=61):
    """
    Extract sequences for a given sensor with variable window size.

    Args:
        data_dir: folder with raw txt files
        sensor: name, e.g. 'TS1', 'EPS1'
        window_in: number of steps for input
        window_out: number of steps for output (auto-reg window)

    Returns:
        inputs: shape (N/2, window_in)
        labels: shape (N/2, window_out)
    """
    filepath = os.path.join(data_dir, f"{sensor}.txt")
    arr = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, ...)
    if arr.ndim == 2:
        # Use the first cycle dimension, flatten the rest
        arr = arr.reshape(arr.shape[0], -1)
    N = arr.shape[0]
    # Make even for pairing
    if N % 2 != 0:
        arr = arr[:-1]
    num_samples = N // 2
    inputs = []
    labels = []
    for i in range(0, N - 1, 2):
        x = arr[i, 1:1+window_in]  # skip index 0, get next window_in
        y = np.concatenate([arr[i, -1:], arr[i+1, :window_out-1]], axis=0)
        inputs.append(x)
        labels.append(y)
    inputs = np.stack(inputs)
    labels = np.stack(labels)
    return inputs, labels

def predict_sensor_autoregressive(model, norm_mean, norm_std, init_seq, window_out=61, device="cpu"):
    """
    Efficient autoregressive inference using LSTM hidden state.
    """
    preds = []
    # Normalize and convert to tensor
    current = (init_seq - norm_mean) / norm_std
    current = current.astype(np.float32)
    inp = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, window_in, 1)
    with torch.no_grad():
        # Initial sequence: get hidden and cell state after full window
        lstm_out, (h, c) = model.lstm(inp)
        last_out = lstm_out[:, -1, :]  # (1, hidden_size)
        pred_norm = model.fc(last_out) # (1, 1)
        pred_val_norm = pred_norm.item()
        pred_val_real = pred_val_norm * norm_std + norm_mean
        preds.append(pred_val_real)
        # Now, autoregressively generate using hidden state, feeding only the last predicted value
        for _ in range(window_out-1):
            in_val = torch.tensor([[[pred_val_norm]]], dtype=torch.float32).to(device) # (1, 1, 1)
            lstm_out, (h, c) = model.lstm(in_val, (h, c)) # (1, 1, hidden_size)
            pred_norm = model.fc(lstm_out[:, -1, :]) # (1, 1)
            pred_val_norm = pred_norm.item()
            pred_val_real = pred_val_norm * norm_std + norm_mean
            preds.append(pred_val_real)
    return np.array(preds)

def log_columnwise_mse_to_tensorboard(y_pred, y_true, run_dir="runs/long_test", tag="MSE_per_step"):
    """
    Logs column-wise MSE (per timestep) to TensorBoard.

    Args:
        y_pred (np.ndarray): shape (N, window_out)
        y_true (np.ndarray): shape (N, window_out)
        run_dir (str): tensorboard log directory
    """
    assert y_pred.shape == y_true.shape
    mse_per_timestep = ((y_pred - y_true) ** 2).mean(axis=0)  # shape: (window_out,)

    writer = SummaryWriter(log_dir=run_dir)
    for t in range(y_pred.shape[1]):
        writer.add_scalar(tag, mse_per_timestep[t], t)
    writer.flush()
    writer.close()
    print(f"Logged per-step MSE to TensorBoard at {run_dir}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from datetime import datetime

    # Define per-frequency window sizes
    SENSOR_FREQ_WINDOWS = {
        "100hz": (5999, 6001),
        "10hz":  (599, 601),
        "1hz":   (59, 61)
    }

    SENSOR_GROUPS = {
        "100hz": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"],
        "10hz":  ["FS1", "FS2"],
        "1hz":   ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
    }

    def sensor_freq(sensor_name):
        for freq, group in SENSOR_GROUPS.items():
            if sensor_name.upper() in group:
                return freq
        raise ValueError(f"Sensor {sensor_name} not found in known groups")


    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", type=str, required=True, help="Sensor name, e.g., TS1 or EPS1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sensor = args.sensor.upper()

    # Get window sizes based on sensor frequency
    freq = sensor_freq(sensor)
    window_in, window_out = SENSOR_FREQ_WINDOWS[freq]
    print(f"Sensor {sensor} frequency group: {freq}, window_in={window_in}, window_out={window_out}")

    # Prepare paths
    base_dir = Path(__file__).resolve().parent / "data" / "raw"
    model_path = Path(f"models/{sensor.lower()}_lstm.pth")
    norm_path = Path(f"models/{sensor.lower()}_lstm_norm_stats.npz")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{sensor.lower()}_long_test/{run_id}"

    # Load normalization stats
    norm = np.load(norm_path)
    norm_mean, norm_std = float(norm["mean"]), float(norm["std"])

    # Load model ONCE, pass into autoregressive predictor
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    model.to(args.device)
    model.eval()

    # Get data for this sensor
    X, Y = extract_sensor_sequences(str(base_dir), sensor, window_in=window_in, window_out=window_out)
    print("Inputs shape:", X.shape)  # (N, window_in)
    print("Labels shape:", Y.shape)  # (N, window_out)

    preds = []
    for i in range(X.shape[0]):
        pred_seq = predict_sensor_autoregressive(model, norm_mean, norm_std, X[i], window_out=window_out, device=args.device)
        if i % 100 == 0 or i == X.shape[0] - 1:
            # Print every 100th prediction and the last one
            print(f"Predicted sequence {i+1}/{X.shape[0]}: {pred_seq[:5]}...")
        preds.append(pred_seq)

    preds = np.stack(preds)  # (N, window_out)
    print("Predicted ~y shape:", preds.shape)

    log_columnwise_mse_to_tensorboard(preds, Y, run_dir=run_dir, tag=f"{sensor}_MSE_per_step")
