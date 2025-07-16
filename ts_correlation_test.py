import os
import numpy as np
import torch
from models.LSTM import LSTMModel
from torch.utils.tensorboard import SummaryWriter
from ts_LSTM_test import extract_temperature_sequences, predict_ts_autoregressive, log_ts_columnwise_mse_to_tensorboard

def predict_ts(model_path, init_seq, device="cpu"):
    """
    Args:
        model_path (str): path to trained ts1_lstm.pth model
        init_seq (np.ndarray): array of shape (59,) containing initial values
        device (str): 'cuda' or 'cpu'

    Returns:
        np.ndarray: predicted val
    """
    assert init_seq.shape == (59,)
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    inp = torch.tensor(init_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # shape: (1, 59, 1)
    with torch.no_grad():
        preds = model(inp)
    return preds.item()



def predict_ts_from_model(model, init_seq, device="cpu"):
    assert init_seq.shape == (59,)
    model.eval()
    inp = torch.tensor(init_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        preds = model(inp)
    return preds.item()

def known_ts_autoregressive(known_seq, ts_idx, models, device="cpu"):
    """
    Args:
        known_seq (np.ndarray): array of shape (3, 120) containing values for known sensors,(excepting the one to predict)
        ts_idx (int): index of the temperature sensor to predict (0-3 for TS1-TS4)

    Returns:
        np.ndarray: array of shape (3, 61) defference between actual value and predicted value for known temperature sensors
    """
    print("  [known_ts_autoregressive] Computing deltas...")
    assert known_seq.shape == (3, 120)
    delta = np.zeros((3, 61))  # shape: (3, 61)

    model_indices = [i for i in range(4) if i != ts_idx]
    for i in range(61):
        for model_pos, actual_idx in enumerate(model_indices):
            init_seq = known_seq[model_pos, i:i +59]  # shape: (59,)
            preds = predict_ts_from_model(models[actual_idx], init_seq, device=device)
            target = known_seq[model_pos, i+59]
            delta[model_pos, i] = target - preds  # actual value - predicted value

    print("  [known_ts_autoregressive] Delta computation done.")
    return delta

def autoregresive_correlation_fix(base_dir, model_dir, ts_idx, models, device="cpu"):
    print("[INFO] Extracting temperature sequences...")
    X, Y = extract_temperature_sequences(base_dir)
    print(f"[INFO] Loaded {X.shape[0]} samples")
    model_path = os.path.join(model_dir, f"ts{ts_idx + 1}_lstm.pth")
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    fixed_seqs = []
    for i in range(X.shape[0]):
        if i % 100 == 0:
            print(f"[PROGRESS] Processing sample {i}/{X.shape[0]}")

        x_ts = X[i]  # shape: (4, 59)
        y_ts = Y[i]  # shape: (4, 61)
        known_seq = np.zeros((4, 120))
        known_seq[:, :59] = x_ts
        known_seq[:, 59:] = y_ts

        # Remove the row for the sensor to predict
        known_seq_trimmed = known_seq[[i for i in range(4) if i != ts_idx]]
        delta = known_ts_autoregressive(known_seq_trimmed, ts_idx, models, device)
        
        current_seq = x_ts[ts_idx]  # shape: (59,) 
        fixed_seq = np.zeros(61)  # shape: (61,)
        for t in range(61):
            if t == 0:
                print(f"    Starting autoregressive prediction for sample {i}")

            pred = predict_ts(model_path, current_seq, device = device)
            correction = delta[:, t].mean() * 0.8  # Adjust the correction factor as needed
            fixed_value = pred + correction
            fixed_seq[t] = fixed_value
            current_seq = np.append(current_seq[1:], fixed_value)
        
        fixed_seqs.append(fixed_seq)

    print("[INFO] All sequences processed.")
    return np.stack(fixed_seqs)

# Example usage (disabled unless called explicitly)

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=int, default=1, choices=[1, 2, 3, 4], help="Which TS sensor to fix (1-4)")
    args = parser.parse_args()


    model_dir = Path(__file__).resolve().parent / "models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[MAIN] Preloading models into memory...")
    models = []
    for i in range(4):
        path = model_dir / f"ts{i+1}_lstm.pth"
        m = LSTMModel(input_size=1, hidden_size=64, num_layers=1)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.to(device)
        models.append(m)

    ts_idx = args.ts - 1  # convert to 0-based index
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define base path
    base_dir = Path(__file__).resolve().parent / "data" / "raw"
    model_dir = Path(__file__).resolve().parent / "models"

    # Run autoregressive correction
    fixed_preds = autoregresive_correlation_fix(base_dir, model_dir, ts_idx=ts_idx, models=models, device=device)

    # Get true labels
    _, labels = extract_temperature_sequences(base_dir)
    true_y = labels[:, ts_idx, :]  # shape: (N, 61)

    # Setup TensorBoard log path
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/ts{args.ts}_corr_fix/{run_id}"

    # Log MSE per timestep
    log_ts_columnwise_mse_to_tensorboard(fixed_preds, true_y, run_dir=log_dir)


