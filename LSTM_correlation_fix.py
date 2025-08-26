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
import torch.nn.functional as F  # <-- add
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

from models.LSTM import LSTMModel  # seq2seq: window -> window

# -----------------------
# IO helpers (unchanged)
# -----------------------
def _find_model_paths(sensor_name: str) -> Tuple[Path, Optional[Path]]:
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
    _, stats_path = _find_model_paths(sensor_name)
    if stats_path is None or not stats_path.exists():
        print(f"[WARN] No norm stats for {sensor_name} at {stats_path}; using mean=0, std=1")
        return 0.0, 1.0
    d = np.load(stats_path)
    return float(d["mean"]), float(d["std"])

def _load_lstm(sensor_name: str, device: torch.device) -> LSTMModel:
    weights_path, _ = _find_model_paths(sensor_name)
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# -----------------------
# Prediction utils (unchanged)
# -----------------------
@torch.no_grad()
def predict_window(model: nn.Module, x_raw: np.ndarray, mean: float, std: float, device: torch.device) -> np.ndarray:
    x_norm = (torch.tensor(x_raw, dtype=torch.float32, device=device).view(1, -1, 1) - mean) / (std + 1e-8)
    yhat_norm = model(x_norm)                # (1,L,1)
    yhat_raw = yhat_norm.squeeze(0).squeeze(-1) * (std + 1e-8) + mean
    return yhat_raw.detach().cpu().numpy()

def slice_window_pair(freq: str, batched_tuple) -> Tuple[np.ndarray, np.ndarray]:
    (x100_b, x10_b, x1_b), (y100_b, y10_b, y1_b) = batched_tuple
    if freq == "100hz":
        X = x100_b.squeeze(0).cpu().numpy()
        Y = y100_b.squeeze(0).cpu().numpy()
    elif freq == "10hz":
        X = x10_b.squeeze(0).cpu().numpy()
        Y = y10_b.squeeze(0).cpu().numpy()
    elif freq == "1hz":
        X = x1_b.squeeze(0).cpu().numpy()
        Y = y1_b.squeeze(0).cpu().numpy()
    else:
        raise ValueError(f"Unknown frequency {freq}")
    return X, Y

# -----------------------
# Main correlation-fix executor (UPDATED)
# -----------------------
def run_fix_for_sensor(sensor: str, alpha: float, device: torch.device):
    sensor = sensor.upper()
    if sensor not in ALL_SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Must be one of: {ALL_SENSORS}")

    freq = FREQ_MAP[sensor]
    L = WINLEN[freq]
    print(f"[INFO] Target sensor: {sensor}  freq={freq}  L={L}")

    # Load binary correlation (to filter same-frequency partners)
    corr_path = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"
    with open(corr_path, "rb") as f:
        co = pickle.load(f)
    co = np.asarray(co, dtype=np.float32)
    if co.shape != (17, 17):
        raise ValueError(f"Correlation matrix must be (17,17); got {co.shape}")

    # Build pooled, magnitude-based A_caus (17x17) from PCMCI (238x238)
    caus_path = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    with open(caus_path, "rb") as f:
        C = pickle.load(f)
    C_t = torch.as_tensor(C, dtype=torch.float32)
    if C_t.shape != (17*14, 17*14):
        raise ValueError(f"PCMCI matrix must be (238,238); got {tuple(C_t.shape)}")
    # |C| -> avg pool 14x14 (stride 14) -> (17,17)
    A_caus = F.avg_pool2d(C_t.abs().unsqueeze(0).unsqueeze(0), kernel_size=(14,14), stride=(14,14))
    A_caus = A_caus.squeeze(0).squeeze(0).cpu().numpy()  # (17,17)

    # Same-frequency correlated sensors (exclude self)
    s_idx = IDX_MAP[sensor]
    same_freq_correlated = []
    for j in range(17):
        if j == s_idx:
            continue
        name_j = ALL_SENSORS[j]
        if FREQ_MAP[name_j] == freq and co[s_idx, j] == 1.0:
            same_freq_correlated.append(name_j)

    if not same_freq_correlated:
        print("[INFO] No same-frequency correlated sensors. Terminating.")
        return

    print(f"[INFO] Same-frequency correlated sensors ({len(same_freq_correlated)}): {same_freq_correlated}")

    # Load models/stats
    target_model = _load_lstm(sensor, device)
    t_mean, t_std = _load_norm_stats(sensor)

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

    # Local index function for this frequency group
    if freq == "1hz":
        s_local_idx = SENSORS_1.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_1.index(name)
    elif freq == "10hz":
        s_local_idx = SENSORS_10.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_10.index(name)
    else:
        s_local_idx = SENSORS_100.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_100.index(name)

    # Iterate all windows and compute weighted fix
    total_windows = 0
    sum_mse_pure = 0.0
    sum_mse_fixed = 0.0

    # Load splits
    def _load_split(suffix: str):
        p = BASE_DIR / "data" / "processed" / f"{sensor.lower()}_{suffix}.pkl"
        if not p.exists(): return None
        with open(p, "rb") as f: return pickle.load(f)

    splits = [ds for suf in ["train", "dev", "test"] if (ds := _load_split(suf)) is not None]
    if not splits:
        print("[ERR] No dataset splits found. Nothing to evaluate.")
        return

    for split in splits:
        loader = DataLoader(split, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            X_mat, Y_mat = slice_window_pair(freq, batch)
            x_t = X_mat[s_local_idx]  # (L,)
            y_t = Y_mat[s_local_idx]  # (L,)

            # Target prediction (raw)
            yhat_t = predict_window(target_model, x_t, t_mean, t_std, device)  # (L,)

            # Collect (weight, delta) per correlated sensor
            weights: List[float] = []
            deltas:  List[np.ndarray] = []

            for name, mdl in corr_models.items():
                k_global = IDX_MAP[name]              # global [0..16]
                w = float(A_caus[k_global, s_idx])    # use directed weight k -> target
                # Optional: symmetrize if you prefer undirected
                # w = float(0.5 * (A_caus[k_global, s_idx] + A_caus[s_idx, k_global]))

                c_idx = local_idx(name)
                c_mean, c_std = corr_stats[name]
                x_c = X_mat[c_idx]
                y_c = Y_mat[c_idx]
                yhat_c = predict_window(mdl, x_c, c_mean, c_std, device)
                delta_c = y_c - yhat_c  # (L,)

                weights.append(max(w, 0.0))
                deltas.append(delta_c)

            if deltas:
                W = np.array(weights, dtype=np.float32)
                D = np.stack(deltas, axis=0)  # (K,L)
                wsum = float(W.sum())
                if wsum > 1e-8:
                    delta_mean = (W[:, None] * D).sum(axis=0) / wsum
                else:
                    # all weights ~0 -> fallback to unweighted mean
                    delta_mean = D.mean(axis=0)
            else:
                delta_mean = np.zeros_like(y_t)

            # Corrected prediction
            yfixed = yhat_t + alpha * delta_mean

            # MSE (raw)
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
    print(f"Correlated sensors used: {list(corr_models.keys())}")
    print(f"Alpha (scale): {alpha:.3f}")
    print(f"Windows processed: {total_windows}")
    print(f"Average MSE (pure LSTM) : {avg_mse_pure:.6f}")
    print(f"Average MSE (after fix) : {avg_mse_fixed:.6f}")
    if avg_mse_pure > 1e-12:
        imp = 100.0 * (avg_mse_pure - avg_mse_fixed) / avg_mse_pure
        print(f"Relative improvement  : {imp:+.7f}%")

# -----------------------
# CLI (unchanged)
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
