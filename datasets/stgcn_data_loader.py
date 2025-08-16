# stgcn_data_loader.py
# -*- coding: utf-8 -*-

"""
ST-GCN Dataset
--------------
Prepares inputs for ST-GCN as:
    x : (C, T, V) where C = 10 (raw, downsampled) + 14 (high-level), T = 6, V = 17
    rul: scalar RUL
    status: (4,) integer labels

Raw downsampling:
  - 100 Hz (7, 6, 1000) → mean every 100 → (7, 6, 10)
  - 10  Hz (2, 6,  100) → mean every 10  → (2, 6, 10)
  - 1   Hz (8, 6,   10) → unchanged      → (8, 6, 10)
Stack sensors → (17, 6, 10) and permute → (10, 6, 17)

High-level features:
  extract_high_level_features(...) must return (N, 17, 6, 14)
  We permute to (N, 14, 6, 17) and concat with raw → (N, 24, 6, 17).
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from features.high_level_feature_extraction import (
    extract_high_level_features,
    extract_cycle_features,  # kept for compatibility if needed elsewhere
)


class STGCNDataset(Dataset):
    def __init__(self, data_dir: Union[str, Path]):
        data_dir = str(data_dir)

        # Fixed node order: 7@100Hz + 2@10Hz + 8@1Hz = 17
        self.sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]  # type: List[str]
        self.sensors_10hz: List[str] = ["FS1", "FS2"]
        self.sensors_1hz: List[str] = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]

        # ---------- Load raw sensor files ----------
        def load_block(names: List[str], expected_len: int) -> np.ndarray:
            mats = []
            for s in names:
                fp = os.path.join(data_dir, f"{s}.txt")
                arr = np.loadtxt(fp, delimiter="\t", dtype=np.float32)  # (N_cycles, expected_len)
                if arr.ndim != 2 or arr.shape[1] != expected_len:
                    raise ValueError(f"{s}.txt has shape {arr.shape}, expected (N_cycles, {expected_len})")
                mats.append(arr)
            return np.stack(mats, axis=1)  # (N, n_sensors, expected_len)

        self.data_100 = load_block(self.sensors_100hz, 6000)  # (N, 7, 6000)
        self.data_10  = load_block(self.sensors_10hz,   600)  # (N, 2,  600)
        self.data_1   = load_block(self.sensors_1hz,     60)  # (N, 8,   60)

        N = self.data_100.shape[0]
        if self.data_10.shape[0] != N or self.data_1.shape[0] != N:
            raise ValueError("Mismatch in number of cycles among frequency groups.")

        # ---------- Load labels ----------
        rul_path = os.path.join(data_dir, "rul_profile.txt")
        raw_target = np.loadtxt(rul_path, dtype=np.float32, delimiter=",")
        if raw_target.ndim != 2 or raw_target.shape[1] < 5:
            raise ValueError(f"rul_profile.txt must have ≥5 columns (RUL + 4 statuses), got {raw_target.shape}")

        rul_array = raw_target[:, 0]       # (N,)
        status_array = raw_target[:, 1:5]  # (N, 4)
        if rul_array.shape[0] != N or status_array.shape[0] != N:
            raise ValueError("Label counts do not match number of cycles.")

        self.rul = torch.from_numpy(rul_array.astype(np.float32))     # (N,)
        self.status = torch.from_numpy(status_array.astype(np.int64)) # (N, 4)

        # ---------- Window to 6 segments per cycle ----------
        x100 = self.data_100.reshape(N, len(self.sensors_100hz), 6, 1000)  # (N,7,6,1000)
        x10  = self.data_10.reshape( N, len(self.sensors_10hz),  6, 100)   # (N,2,6,100)
        x1   = self.data_1.reshape(  N, len(self.sensors_1hz),   6, 10)    # (N,8,6,10)

        # ---------- Downsample to length 10 within each window ----------
        x100 = x100.reshape(N, len(self.sensors_100hz), 6, 10, 100).mean(axis=-1)  # (N,7,6,10)
        x10  = x10.reshape( N, len(self.sensors_10hz),  6, 10, 10 ).mean(axis=-1)  # (N,2,6,10)
        # x1 unchanged                                                    # (N,8,6,10)

        # ---------- Raw stack: (N, 17, 6, 10) → (N, 10, 6, 17) ----------
        raw_v_t_u10 = np.concatenate([x100, x10, x1], axis=1)  # (N,17,6,10)
        tensor_raw = torch.from_numpy(raw_v_t_u10.astype(np.float32)).permute(0, 3, 2, 1).contiguous()

        # ---------- High-level features: accept (6*N, 238) or (N, 17, 6, 14) ----------
        hl_mat, _ = extract_high_level_features(data_dir, start_idx=0, end_idx=N - 1)
        if isinstance(hl_mat, torch.Tensor):
            hl_mat = hl_mat.detach().cpu().numpy()

        if hl_mat.ndim == 2:
            # Expect legacy flat features: (6 * N, 238) where 238 = 17 * 14
            if hl_mat.shape[0] != 6 * N:
                raise ValueError(f"Expected hl_mat with shape (6*N, F) and N={N}, got {hl_mat.shape}")
            F = hl_mat.shape[1]
            if F not in (238, 17 * 14):
                raise ValueError(f"Expected feature dim 238 (=17*14), got {F}")
            # Unflatten: (6*N, 238) -> (N, 6, 17, 14)
            hl_mat = hl_mat.reshape(N, 6, 17, 14)
        elif hl_mat.ndim == 4:
            # Allow either (N, 17, 6, 14) or (N, 6, 17, 14)
            if hl_mat.shape == (N, 17, 6, 14):
                hl_mat = np.transpose(hl_mat, (0, 2, 1, 3))  # -> (N, 6, 17, 14)
            elif hl_mat.shape != (N, 6, 17, 14):
                raise ValueError(f"Unexpected hl_mat 4D shape {hl_mat.shape}. "
                                 f"Expected (N,17,6,14) or (N,6,17,14).")
        else:
            raise ValueError(f"Unexpected hl_mat ndim={hl_mat.ndim}. Expected 2D or 4D.")

        # To ST-GCN layout: (N, 6, 17, 14) -> (N, 14, 6, 17)
        tensor_high = torch.from_numpy(hl_mat.astype(np.float32)).permute(0, 3, 1, 2).contiguous()

        # ---------- Final ST-GCN input: (N, 24, 6, 17) ----------
        self.X = torch.cat([tensor_raw, tensor_high], dim=1)

        # Cache for quick checks
        self.N = N
        self.C = self.X.shape[1]
        self.T = self.X.shape[2]
        self.V = self.X.shape[3]
        if (self.C, self.T, self.V) != (24, 6, 17):
            raise RuntimeError(f"Final x has shape (N, {self.C}, {self.T}, {self.V}), expected (N, 24, 6, 17).")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (C, T, V) = (24, 6, 17)
            rul: scalar float tensor
            status: (4,), dtype long
        """
        x = self.X[idx]                  # (24, 6, 17)
        rul = self.rul[idx]              # ()
        status = self.status[idx].long() # (4,)
        return x, rul, status


if __name__ == "__main__":
    data_dir = BASE_DIR / "data" / "raw"
    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = STGCNDataset(data_dir)

    print(f"Dataset size: {len(dataset)} cycles")
    for i in [0, 1, min(220, len(dataset) - 1)]:
        x, rul, status = dataset[i]
        print(f"\n[Sample {i}]")
        print(f"  x shape     : {x.shape}   (expect (24, 6, 17))")
        print(f"  rul         : {float(rul):.4f}")
        print(f"  status shape: {tuple(status.shape)} (expect (4,))  dtype={status.dtype}")

    # 70/20/10 split
    total = len(dataset)
    train_len = int(total * 0.7)
    val_len   = int(total * 0.2)
    test_len  = total - train_len - val_len

    gen = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=gen)

    def save_split(obj, name: str) -> None:
        path = out_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved {name} -> {path}")

    save_split(train_set, "train_stgcn")
    save_split(val_set,   "val_stgcn")
    save_split(test_set,  "test_stgcn")
    print("Dataset splits saved successfully.")