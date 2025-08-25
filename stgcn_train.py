# stgcn_train.py
# -*- coding: utf-8 -*-

"""
Training script for ST-GCN on HSTI sensors.

Inputs (from stgcn_data_loader.py pickles):
  x: (C, T, V) with C=24 (10 raw + 14 hl), T=6, V=17
  rul: scalar in [0, 1] (recommended; if not, consider scaling)
  status: (4,) integer labels for 4 classification heads

This mirrors a typical train.py structure but targets the ST-GCN model:
- YAML config support (optional). Defaults kick in if the YAML is missing.
- TensorBoard logging:
    * default (3 base partitions): runs/<model>_experiment/<timestamp>
    * with causality (3+1 partitions): runs/<model>_with_causality_experiment/<timestamp>
- Weighted multi-task loss: MSE for RUL + CrossEntropy for 4 status heads
- Checkpointing best model by validation loss

To enable the 1 causality layers in A at runtime:
    python3 stgcn_train.py --add-causality
(Without the flag, only the 3 base layers are used.)
"""

import os
import math
import time
import yaml
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.stgcn_data_loader import STGCNDataset  


# -----------------------
# Utilities
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

def load_yaml_config(default_cfg: Dict[str, Any], yaml_path: Path) -> Dict[str, Any]:
    """Merge defaults with YAML if it exists."""
    cfg = default_cfg.copy()
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # fixed-shape convs benefit from cudnn benchmark
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_partitioned_ring_adjacency(V: int) -> torch.Tensor:
    """
    Build a simple K=3 partitioned adjacency:
      - A_self      : identity
      - A_forward   : ring edges v -> (v+1) % V
      - A_backward  : transpose of forward
    Shape: (K=3, V, V)
    Replace with your true partitions if you have them.
    """
    A_self = torch.eye(V)
    A_fwd = torch.zeros(V, V)
    for v in range(V):
        A_fwd[v, (v + 1) % V] = 1.0
    A_bwd = A_fwd.t()
    A = torch.stack([A_self, A_fwd, A_bwd], dim=0)  # (3, V, V)
    return A

def print_adj_binary(A: torch.Tensor, names=None, sep=""):
    """
    Print each adjacency in A (K,V,V) as 0/1 grid.
    sep="" gives '1001...' lines. Use sep=" " for spaced rows.
    """
    K, V, _ = A.shape
    A_bin = (A > 0).to(torch.int)
    for k in range(K):
        title = names[k] if (names and k < len(names)) else f"A[{k}]"
        print(f"\n{title}:")
        for i in range(V):
            row = sep.join(str(int(x)) for x in A_bin[k, i].tolist())
            print(row)


def macro_precision_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Macro-precision for multi-class classification using argmax predictions.
    No sklearn dependency. Returns python float.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)  # (N,)
        precs: List[float] = []
        eps = 1e-8
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            prec = tp / (tp + fp + eps)
            precs.append(prec)
        return float(sum(precs) / len(precs))

def make_partitioned_adj_from_specs(
    V: int,
    corr_pkl_path: Path,
    add_causality: bool
) -> torch.Tensor:
    """
    K adjacency (K=3 or 4):
      A[0] = I_V  (self)
      A[1] = frequency-group edges (same group, no self)
      A[2] = correlation edges (loaded from pickle, symmetrized, no self)
      If add_causality:
        A[3] = ONE causality layer (17x17), built from the 238x238 PCMCI matrix by:
               abs -> avg-pool (14x14, stride 14) -> threshold (>0.5 -> 1) -> diag=0.
               (Assumes sensor-major ordering: feature index = sensor*14 + feat_id.)
    Node order MUST match the dataset:
      [PS1..PS6, EPS1] (7) + [FS1, FS2] (2) + [TS1..TS4, VS1, SE, CE, CP] (8)  => V=17
    """
    # --- group definitions (must match STGCNDataset ordering) ---
    sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]  # 7
    sensors_10hz  = ["FS1", "FS2"]                                      # 2
    sensors_1hz   = ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]  # 8
    assert V == (len(sensors_100hz) + len(sensors_10hz) + len(sensors_1hz)), "V must be 17."

    # Index ranges in the concatenated node order used by the dataset
    idx_100 = list(range(0, len(sensors_100hz)))  # 0..6
    idx_10  = list(range(idx_100[-1] + 1, idx_100[-1] + 1 + len(sensors_10hz)))  # 7..8
    idx_1   = list(range(idx_10[-1] + 1, idx_10[-1] + 1 + len(sensors_1hz)))     # 9..16

    # --- A0: identity (self connections) ---
    A_self = torch.eye(V, dtype=torch.float32)

    # --- A1: frequency-group adjacency (fully connected within each group, no self) ---
    A_freq = torch.zeros(V, V, dtype=torch.float32)
    for group in [idx_100, idx_10, idx_1]:
        g = torch.tensor(group, dtype=torch.long)
        A_freq[g[:, None], g[None, :]] = 1.0
        A_freq[g, g] = 0.0

    # --- A2: correlation adjacency (binary), force symmetric, drop self loops ---
    import pickle as _pkl
    with open(corr_pkl_path, "rb") as f:
        co = _pkl.load(f)  # expected shape (V, V), dtype numeric (0/1)
    A_corr = co if isinstance(co, torch.Tensor) else torch.tensor(co, dtype=torch.float32)
    if A_corr.shape != (V, V):
        raise ValueError(f"Correlation matrix shape {A_corr.shape} != ({V}, {V})")
    A_corr = 0.5 * (A_corr + A_corr.t())
    A_corr.fill_diagonal_(0.0)

    A_list = [A_self, A_freq, A_corr]

    # --- optionally add ONE 17x17 causality layer ---
    if add_causality:
        caus_path = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
        with open(caus_path, "rb") as f:
            C = _pkl.load(f)  # expected (238, 238) = (17*14, 17*14)
        C = C if isinstance(C, torch.Tensor) else torch.tensor(C, dtype=torch.float32)
        if C.shape != (V*14, V*14):
            raise ValueError(f"Causality matrix must be ({V*14},{V*14}); got {tuple(C.shape)}")

        # abs, then average pooling (kernel=14, stride=14) to collapse features->sensors
        # reshape to NCHW for pooling: (1,1,238,238)
        C_abs = C.abs().unsqueeze(0).unsqueeze(0)  # (1,1,238,238)
        A_caus = F.avg_pool2d(C_abs, kernel_size=(14,14), stride=(14,14))  # (1,1,17,17)
        A_caus = A_caus.squeeze(0).squeeze(0)  # (17,17)

        # threshold > 0.5 -> 1 else 0; drop self-loops
        A_caus = (A_caus > 0.5).to(torch.float32)
        A_caus.fill_diagonal_(0.0)

        A_list.append(A_caus)

    A = torch.stack(A_list, dim=0)  # K=3 (no causality) or K=4 (with causality)
    return A

# -----------------------
# Model factory (added)
# -----------------------
def _model_needs_graph(name: str) -> bool:
    name = name.lower()
    return name in ("stgcn", "stgcn_no_attention", "stgcn_residual_attention","dcrnn")

def build_model(cfg: Dict[str, Any], device: torch.device, A: torch.Tensor = None) -> nn.Module:
    """
    Factory: returns a model with the common interface:
      forward(x) -> {"rul": (N,1), "status_logits": [ ... ]}
    """
    name = cfg.get("model", "stgcn").lower()
    status_classes = cfg["status_classes"]
    in_channels = cfg["in_channels"]
    V = 17
    T = cfg.get("T", 6)

    if name == "stgcn":
        from models.stgcn import STGCN
        model = STGCN(
            A=A,
            status_classes=status_classes,
            in_channels=in_channels,
            channels=tuple(cfg["channels"]),
            temporal_kernel=tuple(cfg["temporal_kernel"]),
            dropout=cfg["dropout"],
            edge_importance=cfg["edge_importance"],
        )

    elif name == "stgcn_residual_attention":
        from models.stgcn_residual_attention import STGCN_residual_attention
        model = STGCN_residual_attention(
            A=A,
            status_classes=status_classes,
            in_channels=in_channels,
            channels=tuple(cfg["channels"]),
            temporal_kernel=tuple(cfg["temporal_kernel"]),
            dropout=cfg["dropout"],
            edge_importance=cfg["edge_importance"],
        )

    elif name == "stgcn_no_attention":
        from models.stgcn_no_attention import STGCN_NoAttention
        model = STGCN_NoAttention(
            A=A,
            status_classes=status_classes,
            in_channels=in_channels,
            channels=tuple(cfg["channels"]),
            temporal_kernel=tuple(cfg["temporal_kernel"]),
            dropout=cfg["dropout"],
            edge_importance=cfg["edge_importance"],
        )

    elif name == "tcn":
        from models.tcn import TCNBackbone
        tcn_channels = tuple(cfg.get("tcn_channels", cfg["channels"]))
        dilations    = tuple(cfg.get("tcn_dilations", [1, 2, 4]))
        kernel_size  = int(cfg.get("tcn_kernel", 3))
        dropout      = float(cfg.get("tcn_dropout", 0.1))
        causal       = bool(cfg.get("tcn_causal", False))
        model = TCNBackbone(
            status_classes=status_classes,
            in_channels=in_channels,
            V=V,
            tcn_channels=tcn_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            causal=causal,
            head_hidden=256,
            head_dropout=0.2,
            pool="mean",
        )

    elif name == "dcrnn":
        from models.dcrnn import DCRNNBackbone
        model = DCRNNBackbone(
            A=A,
            in_channels=in_channels,
            T=T, V=V,
            status_classes=status_classes,
            hidden=cfg.get("dcrnn_hidden", 64),
            num_layers=cfg.get("dcrnn_layers", 2),
            k=cfg.get("dcrnn_k", 1),
            head_hidden=256, head_dropout=0.2,
        )

    elif name == "inception_time":
        from models.inception_time import InceptionTimeBackbone
        model = InceptionTimeBackbone(
            in_channels=in_channels,
            V=V, T=T,
            status_classes=status_classes,
            nb_filters=cfg.get("it_filters", 32),
            depth=cfg.get("it_depth", 6),
            bottleneck_channels=cfg.get("it_bottleneck", 32),
            kernel_sizes=tuple(cfg.get("it_kernels", [3, 5, 7])),
            head_hidden=256, head_dropout=0.2,
            pool="mean",
        )

    else:
        raise ValueError(f"Unknown model '{name}'. Expected one of "
                         "stgcn | stgcn_no_attention | stgcn_residual_attention | tcn | dcrnn | inception_time")

    return model.to(device)


# -----------------------
# Training / Eval loops
# -----------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    status_classes: List[int],
    rul_weight: float,
    status_weight: float,
    scaler: torch.cuda.amp.GradScaler = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    grad_clip: float = 0.0,
) -> Tuple[float, float, List[float]]:
    """
    Returns:
      avg_total_loss, avg_mse_loss (RUL), avg_macro_precisions (per status head)
    """
    model.train()
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_sums = [0.0 for _ in status_classes]  # for status losses
    total_loss = 0.0
    total_mse = 0.0
    prec_sums = [0.0 for _ in status_classes]
    num_batches = 0

    for x, rul, status in loader:
        # x: (N, C, T, V); rul: (N,), status: (N, 4)
        x = x.to(device, non_blocking=True)
        rul = rul.to(device, non_blocking=True).float().view(-1, 1)  # (N,1)
        status = status.to(device, non_blocking=True).long()         # (N,4)


        optimizer.zero_grad(set_to_none=True)

        _check_finite("x", x)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                out = model(x)  # {"rul": (N,1), "status_logits": [ (N,Ci), ... ]}
                rul_pred = out["rul"]            # (N,1), model already has Sigmoid
                logits_list = out["status_logits"]
                _check_finite("rul_pred", rul_pred)
                for i, logits in enumerate(logits_list):
                    _check_finite(f"status_logits[{i}]", logits)
                loss_rul = mse_loss_fn(rul_pred, rul)

                loss_status = 0.0
                for i, num_cls in enumerate(status_classes):
                    logits = logits_list[i]       # (N, num_cls)
                    targets_i = status[:, i]      # (N,)
                    ce = ce_loss_fn(logits, targets_i)
                    loss_status = loss_status + ce
                    ce_sums[i] += float(ce.item())

                loss = rul_weight * loss_rul + status_weight * loss_status
        else:
            out = model(x)
            rul_pred = out["rul"]
            logits_list = out["status_logits"]
            loss_rul = mse_loss_fn(rul_pred, rul)
            loss_status = 0.0
            for i, num_cls in enumerate(status_classes):
                logits = logits_list[i]       # (N, num_cls)
                targets_i = status[:, i]      # (N,)
                ce = ce_loss_fn(logits, targets_i)
                loss_status = loss_status + ce
                ce_sums[i] += float(ce.item())

            loss = rul_weight * loss_rul + status_weight * loss_status

        # backward + step
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # metrics (detach to avoid holding graph)
        total_loss += float(loss.item())
        total_mse  += float(loss_rul.item())

        for i, num_cls in enumerate(status_classes):
            prec = macro_precision_from_logits(logits_list[i].detach(), status[:, i], num_cls)
            prec_sums[i] += prec

        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_mse  = total_mse  / max(1, num_batches)
    avg_precs = [p / max(1, num_batches) for p in prec_sums]
    avg_ces = [s / max(1, num_batches) for s in ce_sums]
    return avg_loss, avg_mse, avg_precs, avg_ces


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    status_classes: List[int],
    rul_weight: float,
    status_weight: float,
) -> Tuple[float, float, List[float], float]:
    """
    Returns:
      avg_total_loss, avg_mse_loss (RUL), avg_macro_precisions (per status head), RMSE(RUL)
    """
    model.eval()
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_sums = [0.0 for _ in status_classes]  # for status losses

    total_loss = 0.0
    total_mse  = 0.0
    prec_sums = [0.0 for _ in status_classes]
    num_batches = 0

    for x, rul, status in loader:
        x = x.to(device, non_blocking=True)
        rul = rul.to(device, non_blocking=True).float().view(-1, 1)
        status = status.to(device, non_blocking=True).long()

        out = model(x)
        rul_pred = out["rul"]
        logits_list = out["status_logits"]

        loss_rul = mse_loss_fn(rul_pred, rul)
        loss_status = 0.0
        for i, num_cls in enumerate(status_classes):
            logits = logits_list[i]
            targets_i = status[:, i]
            ce = ce_loss_fn(logits, targets_i)
            loss_status = loss_status + ce
            ce_sums[i] += float(ce.item())

        loss = rul_weight * loss_rul + status_weight * loss_status

        total_loss += float(loss.item())
        total_mse  += float(loss_rul.item())
        for i, num_cls in enumerate(status_classes):
            prec = macro_precision_from_logits(logits_list[i], status[:, i], num_cls)
            prec_sums[i] += prec
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_mse  = total_mse  / max(1, num_batches)
    avg_precs = [p / max(1, num_batches) for p in prec_sums]
    avg_ces = [s / max(1, num_batches) for s in ce_sums]
    rmse_val = math.sqrt(avg_mse) if avg_mse >= 0.0 else float('nan')
    return avg_loss, avg_mse, avg_precs, rmse_val, avg_ces

def _check_finite(tag: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
        print(f"[NaN/Inf] in {tag} at indices: {bad[:5].tolist()} ...")
        raise RuntimeError(f"{tag} contains NaN/Inf")

# -----------------------
# Main
# -----------------------
def main():
    # CLI flag: --add-causality (default False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-causality", action="store_true",
                        help="Append 1 causality layers to adjacency A; "
                             "logs under <model>_with_causality_experiment/.")
    args, _ = parser.parse_known_args()

    # Defaults (overridden by configs/stgcn.yaml if present)
    default_cfg = {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "epochs": 5000,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "use_amp": False,
        "val_every": 1,
        "status_classes": [3, 4, 3, 4],    
        "in_channels": 24,
        "channels": [64, 32, 10],
        "temporal_kernel": [3, 3, 2],
        "dropout": 0.0,
        "edge_importance": False,
        "rul_weight": 1.0,
        "status_weight": 1.0,
        "num_workers": 4,
        "pin_memory": True,
        # paths
        "train_pkl": str(BASE_DIR / "data" / "processed" / "train_stgcn.pkl"),
        "val_pkl":   str(BASE_DIR / "data" / "processed" / "val_stgcn.pkl"),
        "test_pkl":  str(BASE_DIR / "data" / "processed" / "test_stgcn.pkl"),
        "run_root":  str(BASE_DIR / "runs"),
        "ckpt_dir":  str(BASE_DIR / "checkpoints" / "with_1_causality"),
        # added
        "model": "stgcn",
        "T": 6,
        # alt-model params
        "tcn_channels": [64, 32, 10],
        "tcn_dilations": [1, 2, 4],
        "tcn_kernel": 3,
        "tcn_dropout": 0.1,
        "tcn_causal": False,
        "dcrnn_hidden": 64,
        "dcrnn_layers": 2,
        "dcrnn_k": 1,
        "it_filters": 32,
        "it_depth": 6,
        "it_bottleneck": 32,
        "it_kernels": [3, 5, 7],
        # causality toggle (default false; overridden by CLI)
        "add_causality": False,
    }

    cfg_path = BASE_DIR / "configs" / "stgcn.yaml"
    cfg = load_yaml_config(default_cfg, cfg_path)
    # CLI override
    cfg["add_causality"] = bool(args.add_causality)

    seed_everything(cfg["seed"])
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg["device"])

    # honor env override for model
    cfg["model"] = os.getenv("MODEL", cfg.get("model", "stgcn"))
    print(f"Using model: {cfg['model']}; add_causality={cfg['add_causality']}")

    # --- Logging dirs ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (f"{cfg['model']}_with_causality_experiment"
                if cfg["add_causality"] else f"{cfg['model']}_experiment")
    log_dir = Path(cfg["run_root"]) / exp_name / run_id
    ckpt_dir = Path(cfg["ckpt_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print("TensorBoard:", writer.log_dir)

    # --- Data ---
    with open(cfg["train_pkl"], "rb") as f:
        train_dataset = pickle.load(f)
    with open(cfg["val_pkl"], "rb") as f:
        val_dataset = pickle.load(f)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    # --- Model ---
    V = 17  # number of nodes
    A = None
    if _model_needs_graph(cfg["model"]):
        corr_pkl = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"
        A = make_partitioned_adj_from_specs(V, corr_pkl, cfg["add_causality"])
        names = ["A_self (identity)", "A_freq (within-group)", "A_corr (binary, sym)"]
        print(f"A shape: {tuple(A.shape)}  (K, V, V)")
        print_adj_binary(A, names, sep="")
        for k, name in enumerate(names):
            deg = A[k].sum(dim=1)
            print(f"{name}: nnz={int(A[k].count_nonzero())}, degree range=({deg.min().item():.0f},{deg.max().item():.0f})")

    status_classes: List[int] = cfg["status_classes"]
    model = build_model(cfg, device=device, A=A)

    # --- Optimizer & (optional) scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["use_amp"] and device.type == "cuda"))

    # --- Train ---
    best_val_loss = float("inf")
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss, train_mse, train_precs, train_ce = train_one_epoch(
            model, train_loader, optimizer, device,
            status_classes=status_classes,
            rul_weight=cfg["rul_weight"],
            status_weight=cfg["status_weight"],
            scaler=scaler,
            scheduler=scheduler,
            grad_clip=cfg["grad_clip"],
        )
        val_loss, val_mse, val_precs, val_rmse, val_ce = evaluate(
            model, val_loader, device,
            status_classes=status_classes,
            rul_weight=cfg["rul_weight"],
            status_weight=cfg["status_weight"],
        )
        # --- Logging ---
        writer.add_scalar("total_loss/train_total", train_loss, epoch)
        writer.add_scalar("total_loss/val_total",   val_loss, epoch)
        writer.add_scalar("rul_loss/train",    train_mse, epoch)
        writer.add_scalar("rul_loss/val",      val_mse, epoch)
        writer.add_scalar("rul_loss/rmse_val", val_rmse, epoch)
        for i, p in enumerate(train_precs):
            writer.add_scalar(f"status{i}_precision/train", p, epoch)
            writer.add_scalar(f"status{i}_ce/train", train_ce[i], epoch)
        for i, p in enumerate(val_precs):
            writer.add_scalar(f"status{i}_precision/val", p, epoch)
            writer.add_scalar(f"status{i}_ce/val",   val_ce[i], epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("status_ce/val_sum", sum(val_ce), epoch)
        writer.add_scalar("status_ce/train_sum", sum(train_ce), epoch)

        dt = time.time() - t0
        prec_str = " ".join([f"S{i}_P:{val_precs[i]*100:.1f}%" for i in range(len(status_classes))])
        print(f"Epoch {epoch+1:03d}/{cfg['epochs']}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_RMSE={val_rmse:.4f} {prec_str}  ({dt:.1f}s)")

        # --- Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / f"{cfg['model']}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "cfg": cfg,
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
            print(f"  ↳ saved best checkpoint to {best_path}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
