import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

if __name__ == "__main__":
    # ---- Paths ----
    BASE_DIR = Path(__file__).resolve().parent
    raw_data_dir = BASE_DIR / "data" / "raw"
    lstm_pred_dir = BASE_DIR / "data" / "lstm_predictions"
    out_dir = BASE_DIR / "runs" / "lstm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sensor Names ----
    sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
    sensors_10hz = ["FS1", "FS2"]
    sensors_1hz  = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]

    # ---- Load Predictions ----
    lstm_100 = np.load(lstm_pred_dir / "lstm_100.npy")  # (n_windows, 7, 1000)
    lstm_10  = np.load(lstm_pred_dir / "lstm_10.npy")   # (n_windows, 2, 100)
    lstm_1   = np.load(lstm_pred_dir / "lstm_1.npy")    # (n_windows, 8, 10)

    # ---- Load Raw Data (windowed) ----
    data_100 = [np.loadtxt(raw_data_dir / f"{s}.txt", delimiter='\t', dtype=np.float32) for s in sensors_100hz]
    data_100 = np.stack(data_100, axis=1)
    N_cycles = data_100.shape[0]
    data_100 = data_100.reshape(N_cycles, 7, 6, 1000).transpose(0, 2, 1, 3).reshape(-1, 7, 1000)

    data_10 = [np.loadtxt(raw_data_dir / f"{s}.txt", delimiter='\t', dtype=np.float32) for s in sensors_10hz]
    data_10 = np.stack(data_10, axis=1)
    data_10 = data_10.reshape(N_cycles, 2, 6, 100).transpose(0, 2, 1, 3).reshape(-1, 2, 100)

    data_1 = [np.loadtxt(raw_data_dir / f"{s}.txt", delimiter='\t', dtype=np.float32) for s in sensors_1hz]
    data_1 = np.stack(data_1, axis=1)
    data_1 = data_1.reshape(N_cycles, 8, 6, 10).transpose(0, 2, 1, 3).reshape(-1, 8, 10)

    print(f"[INFO] Loaded predictions and GT: {lstm_100.shape=}, {lstm_10.shape=}, {lstm_1.shape=}")

    writer = SummaryWriter(log_dir=out_dir)

    # ---- Evaluate 100Hz sensors ----
    for si, sensor in enumerate(sensors_100hz):
        pred = lstm_100[:, si, :]   # (n_windows, 1000)
        true = data_100[:, si, :]   # (n_windows, 1000)
        mse = (pred - true) ** 2
        per_step_mse = np.nanmean(mse, axis=0)  # (1000,)
        print(f"[100Hz] {sensor}: mean={np.mean(per_step_mse):.6f}, median={np.median(per_step_mse):.6f}")
        for t in range(per_step_mse.shape[0]):
            writer.add_scalar(f"{sensor}_100hz/mse_per_step", per_step_mse[t], t)

    # ---- Evaluate 10Hz sensors ----
    for si, sensor in enumerate(sensors_10hz):
        pred = lstm_10[:, si, :]   # (n_windows, 100)
        true = data_10[:, si, :]   # (n_windows, 100)
        mse = (pred - true) ** 2
        per_step_mse = np.nanmean(mse, axis=0)  # (100,)
        print(f"[10Hz] {sensor}: mean={np.mean(per_step_mse):.6f}, median={np.median(per_step_mse):.6f}")
        for t in range(per_step_mse.shape[0]):
            writer.add_scalar(f"{sensor}_10hz/mse_per_step", per_step_mse[t], t)

    # ---- Evaluate 1Hz sensors ----
    for si, sensor in enumerate(sensors_1hz):
        pred = lstm_1[:, si, :]   # (n_windows, 10)
        true = data_1[:, si, :]   # (n_windows, 10)
        mse = (pred - true) ** 2
        per_step_mse = np.nanmean(mse, axis=0)  # (10,)
        print(f"[1Hz] {sensor}: mean={np.mean(per_step_mse):.6f}, median={np.median(per_step_mse):.6f}")
        for t in range(per_step_mse.shape[0]):
            writer.add_scalar(f"{sensor}_1hz/mse_per_step", per_step_mse[t], t)

    writer.flush()
    writer.close()
    print(f"[INFO] Stepwise MSE logged to TensorBoard dir: {out_dir}")
