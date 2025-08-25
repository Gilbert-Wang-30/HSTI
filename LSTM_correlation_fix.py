# LSTM_correlation_fix.py
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.missing_data_loader import missing_data_loader
# -----------------------
# Sensor groups & ordering
# -----------------------
SENSORS_100 = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
SENSORS_10  = ["FS1", "FS2"]
SENSORS_1   = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]

ALL_SENSORS = SENSORS_100 + SENSORS_10 + SENSORS_1
FREQ_MAP: Dict[str, str] = {}
FREQ_MAP.update({s: "100hz" for s in SENSORS_100})
FREQ_MAP.update({s: "10hz"  for s in SENSORS_10})
FREQ_MAP.update({s: "1hz"   for s in SENSORS_1})
IDX_MAP = {s: i for i, s in enumerate(ALL_SENSORS)}

# Window length by frequency
WINLEN = {"1hz": 10, "10hz": 100, "100hz": 1000}

BASE_DIR = Path(__file__).resolve().parent

# -----------------------
# Model import
# -----------------------
from models.LSTM import LSTMModel  # seq2seq: window -> window

# -----------------------
# IO helpers
# -----------------------
def _find_model_paths(sensor_name: str) -> Tuple[Path, Optional[Path]]:
    """
    Return (weights_path, stats_path or None).
    Tries checkpoints/LSTM/<sensor>/<sensor>_lstm.pth then models/<sensor>_lstm.pth.
    Stats are expected at models/<sensor>_lstm_norm_stats.npz (fallback to ckpt dir if present).
    """
    ckpt_dir = BASE_DIR / "checkpoints" / "LSTM" / sensor_name.lower()
    models_dir = BASE_DIR / "models"

    w_ckpt   = ckpt_dir   / f"{sensor_name.lower()}_lstm.pth"
    w_models = models_dir / f"{sensor_name.lower()}_lstm.pth"

    if w_ckpt.exists():
        weights = w_ckpt
        stats = models_dir / f"{sensor_name.lower()}_lstm_norm_stats.npz"
        if not stats.exists():
            alt = ckpt_dir / f"{sensor_name.lower()}_lstm_norm_stats.npz"
            stats = alt if alt.exists() else None
    elif w_models.exists():
        weights = w_models
        stats = models_dir / f"{sensor_name.lower()}_lstm_norm_stats.npz"
        stats = stats if stats.exists() else None
    else:
        raise FileNotFoundError(f"No LSTM weights for {sensor_name} at {w_ckpt} or {w_models}")

    return weights, stats

def _load_norm_stats(sensor_name: str) -> Tuple[float, float]:
    """
    Load (mean, std) for a sensor from saved npz; if missing, fallback to (0,1) with a warning.
    """
    _, stats_path = _find_model_paths(sensor_name)
    if stats_path is None or not stats_path.exists():
        print(f"[WARN] No norm stats for {sensor_name} at {stats_path}; using mean=0, std=1")
        return 0.0, 1.0
    d = np.load(stats_path)
    return float(d["mean"]), float(d["std"])

def _load_lstm(sensor_name: str, device: torch.device) -> LSTMModel:
    """
    Instantiate and load the LSTM for a sensor using the training hyperparams (hidden=128, layers=3).
    """
    weights_path, _ = _find_model_paths(sensor_name)
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3)
    # weights_only is PyTorch >=2.4; fall back if not available
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------
# Prediction utils
# -----------------------
@torch.no_grad()
def predict_window(model: nn.Module, x_raw: np.ndarray, mean: float, std: float, device: torch.device) -> np.ndarray:
    """
    model: window->window LSTM
    x_raw: (L,) previous window in RAW units
    Returns yhat_raw: (L,) next window prediction in RAW units
    """
    x_norm = (torch.tensor(x_raw, dtype=torch.float32, device=device).view(1, -1, 1) - mean) / (std + 1e-8)
    yhat_norm = model(x_norm)                # (1,L,1)
    yhat_raw = yhat_norm.squeeze(0).squeeze(-1) * (std + 1e-8) + mean
    return yhat_raw.detach().cpu().numpy()

def slice_window_pair(freq: str, batched_tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    From batched ((x100,x10,x1),(y100,y10,y1)) where each is (B, sensors, L), return
    (X_mat, Y_mat) for B=1 squeezed to (sensors, L) numpy arrays.
    """
    (x100_b, x10_b, x1_b), (y100_b, y10_b, y1_b) = batched_tuple
    if freq == "100hz":
        X = x100_b.squeeze(0).cpu().numpy()  # (7,1000)
        Y = y100_b.squeeze(0).cpu().numpy()
    elif freq == "10hz":
        X = x10_b.squeeze(0).cpu().numpy()   # (2,100)
        Y = y10_b.squeeze(0).cpu().numpy()
    elif freq == "1hz":
        X = x1_b.squeeze(0).cpu().numpy()    # (8,10)
        Y = y1_b.squeeze(0).cpu().numpy()
    else:
        raise ValueError(f"Unknown frequency {freq}")
    return X, Y

# -----------------------
# Main correlation-fix executor
# -----------------------
def run_fix_for_sensor(sensor: str, alpha: float, device: torch.device):
    sensor = sensor.upper()
    if sensor not in ALL_SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Must be one of: {ALL_SENSORS}")

    freq = FREQ_MAP[sensor]
    L = WINLEN[freq]
    print(f"[INFO] Target sensor: {sensor}  freq={freq}  L={L}")

    # Load correlation matrix
    corr_path = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"
    with open(corr_path, "rb") as f:
        co = pickle.load(f)
    co = np.asarray(co, dtype=np.float32)
    if co.shape != (17, 17):
        raise ValueError(f"Correlation matrix must be (17,17); got {co.shape}")

    # Correlated same-frequency sensors (exclude self)
    s_idx = IDX_MAP[sensor]
    correlated_idxs = [j for j in range(17) if co[s_idx, j] == 1.0]
    same_freq_correlated: List[str] = []
    for j in correlated_idxs:
        name_j = ALL_SENSORS[j]
        if FREQ_MAP[name_j] == freq and name_j != sensor:
            same_freq_correlated.append(name_j)

    if not same_freq_correlated:
        print("[INFO] No same-frequency correlated sensors. Terminating.")
        return

    print(f"[INFO] Same-frequency correlated sensors ({len(same_freq_correlated)}): {same_freq_correlated}")

    # Load target model/stats
    target_model = _load_lstm(sensor, device)
    t_mean, t_std = _load_norm_stats(sensor)

    # Load correlated models/stats
    corr_models: Dict[str, LSTMModel] = {}
    corr_stats: Dict[str, Tuple[float, float]] = {}
    for name in same_freq_correlated:
        try:
            corr_models[name] = _load_lstm(name, device)
            corr_stats[name]  = _load_norm_stats(name)
        except FileNotFoundError:
            print(f"[WARN] Missing LSTM for correlated sensor {name}; skipping.")
    if not corr_models:
        print("[INFO] No correlated LSTMs available; terminating.")
        return

    # dataset splits for the target sensor (train/dev/test prepared by your packer)
    def _load_split(suffix: str):
        p = BASE_DIR / "data" / "processed" / f"{sensor.lower()}_{suffix}.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    splits = []
    for suf in ["train", "dev", "test"]:
        ds = _load_split(suf)
        if ds is None:
            print(f"[WARN] {sensor.lower()}_{suf}.pkl not found; skipping this split.")
        else:
            splits.append(ds)

    if not splits:
        print("[ERR] No dataset splits found. Nothing to evaluate.")
        return

    # choose local index function in the frequency group
    if freq == "1hz":
        s_local_idx = SENSORS_1.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_1.index(name)
    elif freq == "10hz":
        s_local_idx = SENSORS_10.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_10.index(name)
    else:
        s_local_idx = SENSORS_100.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_100.index(name)

    # Iterate over all samples; compute MSE for pure LSTM and corrected predictions
    total_windows = 0
    sum_mse_pure = 0.0
    sum_mse_fixed = 0.0

    for split in splits:
        loader = DataLoader(split, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            # batch is ((x100_b, x10_b, x1_b), (y100_b, y10_b, y1_b)) with leading B=1
            X_mat, Y_mat = slice_window_pair(freq, batch)

            # Target prev/current windows (raw)
            x_t = X_mat[s_local_idx]  # (L,)
            y_t = Y_mat[s_local_idx]  # (L,)

            # Predict target yhat (raw)
            yhat_t = predict_window(target_model, x_t, t_mean, t_std, device)  # (L,)

            # Build mean residual from correlated sensors (raw)
            deltas = []
            for name, mdl in corr_models.items():
                c_idx = local_idx(name)
                c_mean, c_std = corr_stats[name]
                x_c = X_mat[c_idx]
                y_c = Y_mat[c_idx]
                yhat_c = predict_window(mdl, x_c, c_mean, c_std, device)
                deltas.append(y_c - yhat_c)  # (L,)

            if deltas:
                delta_mean = np.mean(np.stack(deltas, axis=0), axis=0)  # (L,)
            else:
                delta_mean = np.zeros_like(y_t)

            # Corrected prediction
            yfixed = yhat_t + alpha * delta_mean  # (L,)

            # MSEs (raw units)
            mse_pure  = float(np.mean((y_t - yhat_t) ** 2))
            mse_fixed = float(np.mean((y_t - yfixed) ** 2))

            sum_mse_pure  += mse_pure
            sum_mse_fixed += mse_fixed
            total_windows += 1

    if total_windows == 0:
        print("[ERR] No windows processed (empty datasets?).")
        return

    avg_mse_pure  = sum_mse_pure  / total_windows
    avg_mse_fixed = sum_mse_fixed / total_windows

    print("\n=== Correlation Fix Summary ===")
    print(f"Target sensor: {sensor}  freq={freq}  L={L}")
    print(f"Correlated same-frequency sensors used: {list(corr_models.keys())}")
    print(f"Alpha (scale): {alpha:.3f}")
    print(f"Windows processed: {total_windows}")
    print(f"Average MSE (pure LSTM) : {avg_mse_pure:.6f}")
    print(f"Average MSE (after fix) : {avg_mse_fixed:.6f}")
    if avg_mse_pure > 1e-12:
        imp = 100.0 * (avg_mse_pure - avg_mse_fixed) / avg_mse_pure
        print(f"Relative improvement  : {imp:+.2f}%")

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=str, required=True,
                    help="Target sensor to fix (one of: " + ",".join(ALL_SENSORS) + ")")
    ap.add_argument("--alpha", type=float, default=1.0, help="Scale for residual correction (default: 1.0)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_fix_for_sensor(args.sensor, alpha=args.alpha, device=device)

if __name__ == "__main__":
    main()
