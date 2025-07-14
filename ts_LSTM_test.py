import os
import numpy as np
import torch
from models.LSTM import LSTMModel
from torch.utils.tensorboard import SummaryWriter

def extract_temperature_sequences(data_dir):
    """
    Extract sequences from raw data to generate:
      - inputs: (N/2, 4, 59)  → first 59 TS1-TS4 readings from every 2 cycles
      - labels: (N/2, 4, 61) → full 60+1 TS1-TS4 readings from the next 2 cycles
    """
    temp_sensors = ["TS1", "TS2", "TS3", "TS4"]
    sensor_data = []

    for sensor in temp_sensors:
        filepath = os.path.join(data_dir, f"{sensor}.txt")
        data = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)  # shape: (N_cycles, 60)
        sensor_data.append(data)  # list of (N_cycles, 60)

    # Stack to (N_cycles, 4, 60)
    stacked = np.stack(sensor_data, axis=1)

    N = stacked.shape[0]
    if N % 2 != 0:
        stacked = stacked[:-1]  # trim if odd

    num_samples = N // 2
    inputs = []
    labels = []

    for i in range(0, N - 1, 2):
        x = stacked[i, :, 1:]     # (4, 59) from cycle i (skip index 0)
        y = np.concatenate([stacked[i, :, -1:], stacked[i+1, :, :]], axis=1)  # (4, 1+60) = (4, 61)
        inputs.append(x)
        labels.append(y)

    inputs = np.stack(inputs)  # (N/2, 4, 59)
    labels = np.stack(labels)  # (N/2, 4, 61)
    return inputs, labels

def predict_ts1_autoregressive(model_path, init_seq, device="cpu"):
    """
    Args:
        model_path (str): path to trained ts1_lstm.pth model
        init_seq (np.ndarray): array of shape (59,) containing initial values
        device (str): 'cuda' or 'cpu'

    Returns:
        np.ndarray: predicted sequence of shape (61,)
    """
    assert init_seq.shape == (59,)
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    preds = []
    current = init_seq.copy()

    for _ in range(61):
        # Prepare input tensor
        inp = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # shape: (1, 59, 1)
        with torch.no_grad():
            pred = model(inp)  # shape: (1, 1)
        pred_val = pred.item()
        preds.append(pred_val)
        # Slide window
        current = np.concatenate([current[1:], [pred_val]])

    return np.array(preds)



def log_ts1_columnwise_mse_to_tensorboard(y_pred, y_true, run_dir="runs/ts1_long_test"):
    """
    Logs column-wise MSE (per timestep) to TensorBoard.

    Args:
        y_pred (np.ndarray): shape (N, 61) predicted TS1 values
        y_true (np.ndarray): shape (N, 61) ground truth TS1 values
        run_dir (str): tensorboard log directory
    """
    assert y_pred.shape == y_true.shape
    mse_per_timestep = ((y_pred - y_true) ** 2).mean(axis=0)  # shape: (61,)

    writer = SummaryWriter(log_dir=run_dir)
    for t in range(61):
        writer.add_scalar("MSE_per_step", mse_per_timestep[t], t)
    writer.flush()
    writer.close()
    print(f"Logged per-step MSE to TensorBoard at {run_dir}")

# Example usage (disabled unless called explicitly)
# base_dir = "/path/to/raw"
# X, Y = extract_temperature_sequences(base_dir)
# print(X.shape, Y.shape)  # Expect (N/2, 4, 59) and (N/2, 4, 61)
if __name__ == "__main__":
    # Example usage
    base_dir = "data/raw"  # Adjust to your actual data directory
    X, Y = extract_temperature_sequences(base_dir)
    print("Inputs shape:", X.shape)  # Expect (N/2, 4, 59)
    print("Labels shape:", Y.shape)  # Expect (N/2, 4, 61)

    from pathlib import Path
    base_dir = Path(__file__).resolve().parent / "data" / "raw"
    model_path = Path("models/ts1_lstm.pth")

    X, Y = extract_temperature_sequences(base_dir)
    ts1_preds = []
    for i in range(X.shape[0]):
        x_ts1 = X[i, 0]  # shape: (59,)
        pred_seq = predict_ts1_autoregressive(model_path, x_ts1, device="gpu" if torch.cuda.is_available() else "cpu")
        ts1_preds.append(pred_seq)

    ts1_preds = np.stack(ts1_preds)  # shape: (N/2, 61)
    print("Predicted ~y shape:", ts1_preds.shape)

    log_ts1_columnwise_mse_to_tensorboard(ts1_preds, Y[:, 0, :], run_dir="runs/ts1_long_test")


