# LSTM_correlation_fix.py
# -*- coding: utf-8 -*-

"""
Correlation-based residual correction for per-sensor LSTM window predictions (no manual alpha).

Core idea (one window):
  ŷ_s = LSTM_s(x_s)                                   # target sensor prediction
  λ_c  = y_c - ŷ_c                                    # correlated sensor residual
  β_c  = pooled-PCMCI(s <- c)                          # directional weight from C (k->s), non-negative
  α_c  = clip(E_s / (E_c + eps), min_alpha, max_alpha) # adaptive scale from sensor sensitivities E
  Δ    = (∑_c β_c * α_c * λ_c) / (∑_c β_c + eps)       # weighted correction
  ŷ_s^fixed = ŷ_s + Δ

Where sensor sensitivity E_k is computed (first pass) as:
  E_k = median_over_windows( median_over_time(|y_k - ŷ_k|) )
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# First pass: compute sensitivities E_k (median of window-wise median |residual|)
# -----------------------
def compute_sensitivities(sensor: str,
                          same_freq_corr: List[str],
                          device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Returns: (E_s, E_c_dict) where E_s is target sensor sensitivity,
             and E_c_dict maps correlated sensor name -> E_c.
    """
    sensor = sensor.upper()
    freq = FREQ_MAP[sensor]

    # Load models and stats
    t_model = _load_lstm(sensor, device)
    t_mean, t_std = _load_norm_stats(sensor)

    corr_models: Dict[str, LSTMModel] = {}
    corr_stats: Dict[str, Tuple[float, float]] = {}
    for name in same_freq_corr:
        try:
            corr_models[name] = _load_lstm(name, device)
            corr_stats[name]  = _load_norm_stats(name)
        except FileNotFoundError:
            print(f"[WARN] Missing LSTM for correlated sensor {name}; skipping sensitivity.")
    # Prepare splits
    def _load_split(suffix: str):
        p = BASE_DIR / "data" / "processed" / f"{sensor.lower()}_{suffix}.pkl"
        if not p.exists(): return None
        with open(p, "rb") as f: return pickle.load(f)
    splits = [ds for suf in ["train", "dev", "test"] if (ds := _load_split(suf)) is not None]
    if not splits:
        print("[ERR] No dataset splits for sensitivity; returning defaults.")
        return 1.0, {k: 1.0 for k in same_freq_corr}

    # Local index helpers
    if freq == "1hz":
        s_local_idx = SENSORS_1.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_1.index(name)
    elif freq == "10hz":
        s_local_idx = SENSORS_10.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_10.index(name)
    else:
        s_local_idx = SENSORS_100.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_100.index(name)

    # Collect window-wise residual medians for target and correlated
    target_meds: List[float] = []
    corr_meds: Dict[str, List[float]] = {k: [] for k in corr_models.keys()}

    for split in splits:
        loader = DataLoader(split, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            X_mat, Y_mat = slice_window_pair(freq, batch)
            # target residual median
            x_t = X_mat[s_local_idx]
            y_t = Y_mat[s_local_idx]
            yhat_t = predict_window(t_model, x_t, t_mean, t_std, device)
            med_t = float(np.median(np.abs(y_t - yhat_t)))
            target_meds.append(med_t)
            # correlated residual medians
            for name, mdl in corr_models.items():
                c_idx = local_idx(name)
                c_mean, c_std = corr_stats[name]
                x_c = X_mat[c_idx]
                y_c = Y_mat[c_idx]
                yhat_c = predict_window(mdl, x_c, c_mean, c_std, device)
                med_c = float(np.median(np.abs(y_c - yhat_c)))
                corr_meds[name].append(med_c)

    # Aggregate by median across windows
    E_s = float(np.median(target_meds)) if target_meds else 1.0
    E_c = {name: (float(np.median(vals)) if len(vals) > 0 else 1.0)
           for name, vals in corr_meds.items()}
    return E_s, E_c


# -----------------------
# Main correlation-fix executor (two-pass; no manual alpha)
# -----------------------
def run_fix_for_sensor(sensor: str,
                       device: torch.device,
                       min_alpha: float,
                       max_alpha: float):
    sensor = sensor.upper()
    if sensor not in ALL_SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Must be one of: {ALL_SENSORS}")

    freq = FREQ_MAP[sensor]
    L = WINLEN[freq]
    print(f"[INFO] Target sensor: {sensor}  freq={freq}  L={L}")

    # Load binary correlation (to find partners at same frequency)
    corr_path = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"
    with open(corr_path, "rb") as f:
        co = pickle.load(f)
    co = np.asarray(co, dtype=np.float32)
    if co.shape != (17, 17):
        raise ValueError(f"Correlation matrix must be (17,17); got {co.shape}")

    # Pooled, magnitude-based A_caus (17x17) from PCMCI (238x238)
    caus_path = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    with open(caus_path, "rb") as f:
        C = pickle.load(f)
    C_t = torch.as_tensor(C, dtype=torch.float32)
    if C_t.shape != (17*14, 17*14):
        raise ValueError(f"PCMCI matrix must be (238,238); got {tuple(C_t.shape)}")
    A_caus = F.avg_pool2d(C_t.abs().unsqueeze(0).unsqueeze(0), kernel_size=(14,14), stride=(14,14))
    A_caus = A_caus.squeeze(0).squeeze(0).cpu().numpy()  # (17,17)

    # Same-frequency correlated sensors (exclude self)
    s_idx = IDX_MAP[sensor]
    same_freq_correlated = []
    for j in range(17):
        if j == s_idx: continue
        name_j = ALL_SENSORS[j]
        if FREQ_MAP[name_j] == freq and co[s_idx, j] == 1.0:
            same_freq_correlated.append(name_j)

    if not same_freq_correlated:
        print("[INFO] No same-frequency correlated sensors. Terminating.")
        return

    print(f"[INFO] Same-frequency correlated sensors ({len(same_freq_correlated)}): {same_freq_correlated}")

    # -------- First pass: sensitivities E_s and E_c --------
    print("[PASS 1] Computing per-sensor sensitivities (robust residual medians)...")
    E_s, E_c = compute_sensitivities(sensor, same_freq_correlated, device)
    print(f"[PASS 1] E_s (target): {E_s:.6f}")
    for name in same_freq_correlated:
        print(f"[PASS 1] E_{name}: {E_c.get(name, float('nan')):.6f}")

    # Fixed α_c per correlated sensor (clipped)
    eps = 1e-8
    alpha_c: Dict[str, float] = {}
    for name in same_freq_correlated:
        Ec = E_c.get(name, 1.0)
        a = E_s / (Ec + eps)
        a = max(min_alpha, min(max_alpha, a))
        alpha_c[name] = float(a)
    print("[PASS 1] α_c (clipped) per correlated sensor:")
    for name in same_freq_correlated:
        print(f"  alpha[{name}] = {alpha_c[name]:.6f}")

    # Load models/stats for pass 2
    target_model = _load_lstm(sensor, device)
    t_mean, t_std = _load_norm_stats(sensor)

    corr_models: Dict[str, LSTMModel] = {}
    corr_stats: Dict[str, Tuple[float, float]] = {}
    for name in same_freq_correlated:
        try:
            corr_models[name] = _load_lstm(name, device)
            corr_stats[name]  = _load_norm_stats(name)
        except FileNotFoundError:
            print(f"[WARN] Missing LSTM for correlated sensor {name}; skipping in pass 2.")
    if not corr_models:
        print("[INFO] No correlated LSTMs available; terminating.")
        return

    # Load splits
    def _load_split(suffix: str):
        p = BASE_DIR / "data" / "processed" / f"{sensor.lower()}_{suffix}.pkl"
        if not p.exists(): return None
        with open(p, "rb") as f: return pickle.load(f)
    splits = [ds for suf in ["train", "dev", "test"] if (ds := _load_split(suf)) is not None]
    if not splits:
        print("[ERR] No dataset splits found. Nothing to evaluate.")
        return

    # Local index function in the frequency group
    if freq == "1hz":
        s_local_idx = SENSORS_1.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_1.index(name)
    elif freq == "10hz":
        s_local_idx = SENSORS_10.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_10.index(name)
    else:
        s_local_idx = SENSORS_100.index(sensor)
        def local_idx(name: str) -> int: return SENSORS_100.index(name)

    # -------- Second pass: do weighted correction with β_c * α_c --------
    print("[PASS 2] Evaluating correction with learned α_c and A_caus weights...")
    total_windows = 0
    sum_mse_pure = 0.0
    sum_mse_fixed = 0.0

    for split in splits:
        loader = DataLoader(split, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            X_mat, Y_mat = slice_window_pair(freq, batch)
            x_t = X_mat[s_local_idx]
            y_t = Y_mat[s_local_idx]
            yhat_t = predict_window(target_model, x_t, t_mean, t_std, device)

            weights: List[float] = []
            scaled_deltas: List[np.ndarray] = []

            for name, mdl in corr_models.items():
                k_global = IDX_MAP[name]
                beta = float(A_caus[k_global, s_idx])     # directional weight k -> s
                a    = alpha_c.get(name, 1.0)            # learned α_c
                c_idx = local_idx(name)
                c_mean, c_std = corr_stats[name]
                x_c = X_mat[c_idx]
                y_c = Y_mat[c_idx]
                yhat_c = predict_window(mdl, x_c, c_mean, c_std, device)
                delta_c = y_c - yhat_c
                weights.append(max(beta, 0.0))
                scaled_deltas.append(a * delta_c)

            if scaled_deltas:
                W = np.array(weights, dtype=np.float32)
                D = np.stack(scaled_deltas, axis=0)  # (K,L) already scaled by α_c
                wsum = float(W.sum())
                if wsum > 1e-8:
                    delta_mean = (W[:, None] * D).sum(axis=0) / wsum
                else:
                    delta_mean = D.mean(axis=0)
            else:
                delta_mean = np.zeros_like(y_t)

            yfixed = yhat_t + delta_mean

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
    rel_imp = 100.0 * (avg_mse_pure - avg_mse_fixed) / max(avg_mse_pure, 1e-12)

    print("\n=== Correlation Fix Summary (learned α_c, no manual alpha) ===")
    print(f"Target sensor: {sensor}  freq={freq}  L={L}")
    print(f"Correlated sensors used: {list(corr_models.keys())}")
    print(f"Windows processed: {total_windows}")
    print(f"Average MSE (pure LSTM) : {avg_mse_pure:.6f}")
    print(f"Average MSE (after fix) : {avg_mse_fixed:.6f}")
    print(f"Relative improvement     : {rel_imp:+.4f}%")

    # Show a few weights/scales used
    print("\nTop correlation weights β_c (k->s) and α_c scales:")
    pairs = []
    for name in corr_models.keys():
        k_global = IDX_MAP[name]
        beta = float(A_caus[k_global, s_idx])
        pairs.append((name, beta, alpha_c.get(name, 1.0)))
    pairs.sort(key=lambda t: t[1], reverse=True)
    for name, beta, a in pairs[:10]:
        print(f"  {name:>4s}: beta={beta:.4f}  alpha={a:.4f}")


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=str, required=True,
                    help="Target sensor to fix (one of: " + ",".join(ALL_SENSORS) + ")")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-alpha", type=float, default=0.0001,
                    help="Lower clamp for adaptive scale α_c (default 0.0001)")
    ap.add_argument("--max-alpha", type=float, default=4.0,
                    help="Upper clamp for adaptive scale α_c (default 4.0)")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_fix_for_sensor(args.sensor,
                       device=device,
                       min_alpha=float(args.min_alpha),
                       max_alpha=float(args.max_alpha))


if __name__ == "__main__":
    main()
