# stgcn_ablation_train.py
# -*- coding: utf-8 -*-

"""
Ablation training for ST-GCN (NO attention) on HSTI sensors.

Trains three variants with *different K and actual partitions* (no zeroing):
  1) self                    -> K=1
  2) self + corr             -> K=2
  3) self + causality(thr)   -> K=2  (ONE pooled PCMCI layer with threshold)

Causality layer is built from the 238x238 PCMCI matrix by:
    |C| -> avg_pool2d(14,14,stride=14) -> (17x17) -> >thr -> 1/0 -> diag=0

Defaults mirror stgcn_train.py:
- frequency partition is NOT used
- same LR/scheduler/label smoothing etc.
- logs to runs/stgcn_ablation_simple/<variant>_experiment/<timestamp>
- saves to checkpoints/stgcn_ablation_simple/<variant>_best.pt

Usage:
  python3 stgcn_ablation_train.py --device cuda
  python3 stgcn_ablation_train.py --causality-thresh 0.75 --epochs 2000 --device cuda

Env override to try a different backbone (optional):
  MODEL=stgcn python3 stgcn_ablation_train.py ...
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
from datasets.stgcn_data_loader import STGCNDataset  # noqa: F401 (kept for parity)


BASE_DIR = Path(__file__).resolve().parent

# -----------------------
# Small utils
# -----------------------
def seed_everything(seed: int) -> None:
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def macro_precision_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    precs, eps = [], 1e-8
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        precs.append(tp / (tp + fp + eps))
    return float(sum(precs) / len(precs))

def print_adj_binary(A: torch.Tensor, names=None, sep=""):
    K, V, _ = A.shape
    A_bin = (A > 0).to(torch.int)
    for k in range(K):
        title = names[k] if (names and k < len(names)) else f"A[{k}]"
        print(f"\n{title}:")
        for i in range(V):
            row = sep.join(str(int(x)) for x in A_bin[k, i].tolist())
            print(row)

# -----------------------
# Adjacency builders (no frequency)
# -----------------------
def make_A_self(V: int) -> torch.Tensor:
    return torch.eye(V, dtype=torch.float32).unsqueeze(0)  # (1,V,V)

def make_A_corr(V: int, corr_pkl_path: Path) -> torch.Tensor:
    import pickle as _pkl
    with open(corr_pkl_path, "rb") as f:
        co = _pkl.load(f)
    A_corr = co if isinstance(co, torch.Tensor) else torch.tensor(co, dtype=torch.float32)
    if A_corr.shape != (V, V):
        raise ValueError(f"Correlation matrix {A_corr.shape} != ({V},{V})")
    A_corr = 0.5 * (A_corr + A_corr.t())
    A_corr.fill_diagonal_(0.0)
    return A_corr.unsqueeze(0)  # (1,V,V)

def make_A_causality(V: int, thresh: float) -> torch.Tensor:
    """ ONE pooled causality layer from PCMCI 238x238. """
    caus_pkl = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    if not caus_pkl.exists():
        raise FileNotFoundError(f"Missing PCMCI file: {caus_pkl}")
    import pickle as _pkl
    with open(caus_pkl, "rb") as f:
        C = _pkl.load(f)
    C = C if isinstance(C, torch.Tensor) else torch.tensor(C, dtype=torch.float32)
    if C.shape != (V*14, V*14):
        raise ValueError(f"PCMCI matrix must be ({V*14},{V*14}); got {tuple(C.shape)}")
    C_abs = C.abs().unsqueeze(0).unsqueeze(0)
    A_caus = F.avg_pool2d(C_abs, kernel_size=(14,14), stride=(14,14)).squeeze(0).squeeze(0)  # (17,17)
    A_caus = (A_caus > float(thresh)).to(torch.float32)
    A_caus.fill_diagonal_(0.0)
    return A_caus.unsqueeze(0)  # (1,V,V)

# -----------------------
# Model factory (no-attention default)
# -----------------------
def build_model(name: str, cfg: Dict[str, Any], device: torch.device, A: torch.Tensor) -> nn.Module:
    name = name.lower()
    status_classes = cfg["status_classes"]; in_channels = cfg["in_channels"]
    if name == "stgcn_no_attention":
        from models.stgcn_no_attention import STGCN_NoAttention
        model = STGCN_NoAttention(
            A=A, status_classes=status_classes, in_channels=in_channels,
            channels=tuple(cfg["channels"]), temporal_kernel=tuple(cfg["temporal_kernel"]),
            dropout=cfg["dropout"], edge_importance=cfg["edge_importance"],
        )
    elif name == "stgcn":
        from models.stgcn import STGCN
        model = STGCN(
            A=A, status_classes=status_classes, in_channels=in_channels,
            channels=tuple(cfg["channels"]), temporal_kernel=tuple(cfg["temporal_kernel"]),
            dropout=cfg["dropout"], edge_importance=cfg["edge_importance"],
        )
    else:
        raise ValueError(f"Unsupported MODEL={name} for this ablation script. Use stgcn_no_attention (default) or stgcn.")
    return model.to(device)

# -----------------------
# Train / Eval loops (same as your trainer)
# -----------------------
def train_one_epoch(
    model, loader, optimizer, device, status_classes, rul_weight, status_weight,
    scaler=None, scheduler=None, grad_clip=0.0,
):
    model.train()
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss = total_mse = 0.0
    prec_sums = [0.0 for _ in status_classes]
    num_batches = 0
    for x, rul, status in loader:
        x = x.to(device, non_blocking=True)
        rul = rul.to(device, non_blocking=True).float().view(-1,1)
        status = status.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                out = model(x)
                lr = mse_loss(out["rul"], rul)
                ls = 0.0
                for i,_ in enumerate(status_classes):
                    ls = ls + ce_loss(out["status_logits"][i], status[:,i])
                loss = rul_weight*lr + status_weight*ls
        else:
            out = model(x)
            lr = mse_loss(out["rul"], rul)
            ls = 0.0
            for i,_ in enumerate(status_classes):
                ls = ls + ce_loss(out["status_logits"][i], status[:,i])
            loss = rul_weight*lr + status_weight*ls
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None: scheduler.step()
        total_loss += float(loss.item())
        total_mse  += float(lr.item())
        for i,nc in enumerate(status_classes):
            prec_sums[i] += macro_precision_from_logits(out["status_logits"][i].detach(), status[:,i], nc)
        num_batches += 1
    avg_loss = total_loss / max(1,num_batches)
    avg_mse  = total_mse  / max(1,num_batches)
    avg_prec = [p/max(1,num_batches) for p in prec_sums]
    return avg_loss, avg_mse, avg_prec

@torch.no_grad()
def evaluate(model, loader, device, status_classes, rul_weight, status_weight):
    model.eval()
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss = total_mse = 0.0
    prec_sums  = [0.0 for _ in status_classes]
    num_batches = 0
    for x, rul, status in loader:
        x = x.to(device, non_blocking=True)
        rul = rul.to(device, non_blocking=True).float().view(-1,1)
        status = status.to(device, non_blocking=True).long()
        out = model(x)
        lr = mse_loss(out["rul"], rul)
        ls = 0.0
        for i,_ in enumerate(status_classes):
            ls = ls + ce_loss(out["status_logits"][i], status[:,i])
        loss = rul_weight*lr + status_weight*ls
        total_loss += float(loss.item())
        total_mse  += float(lr.item())
        for i,nc in enumerate(status_classes):
            prec_sums[i] += macro_precision_from_logits(out["status_logits"][i], status[:,i], nc)
        num_batches += 1
    avg_loss = total_loss / max(1,num_batches)
    avg_mse  = total_mse  / max(1,num_batches)
    avg_prec = [p/max(1,num_batches) for p in prec_sums]
    rmse     = math.sqrt(avg_mse) if avg_mse>=0 else float("nan")
    return avg_loss, avg_mse, avg_prec, rmse

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--causality-thresh", type=float, default=0.75)

    # data paths like stgcn_train.py
    ap.add_argument("--train-pkl", type=str, default=str(BASE_DIR / "data" / "processed" / "train_stgcn.pkl"))
    ap.add_argument("--val-pkl",   type=str, default=str(BASE_DIR / "data" / "processed" / "val_stgcn.pkl"))
    ap.add_argument("--run-root",  type=str, default=str(BASE_DIR / "runs" / "stgcn_ablation"))
    ap.add_argument("--ckpt-dir",  type=str, default=str(BASE_DIR / "checkpoints" / "stgcn_ablation"))
    args = ap.parse_args()

    # cfg aligned with your stgcn_train.py
    cfg: Dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "use_amp": args.use_amp,
        "status_classes": [3,4,3,4],
        "in_channels": 24,
        "channels": [64,32,10],
        "temporal_kernel": [3,3,2],
        "dropout": 0.0,
        "edge_importance": False,
        "rul_weight": 1.0,
        "status_weight": 1.0,
        "num_workers": 4,
        "pin_memory": True,
    }

    seed_everything(cfg["seed"])
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg["device"])

    # choose backbone: your best is stgcn_no_attention
    model_name = os.getenv("MODEL", "stgcn_no_attention")
    print(f"Backbone: {model_name} | causality_thresh={args.causality_thresh}")

    # data
    with open(args.train_pkl, "rb") as f: train_dataset = pickle.load(f)
    with open(args.val_pkl,   "rb") as f: val_dataset   = pickle.load(f)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    V = 17
    corr_pkl = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"

    # Prepare the three A variants
    A_self   = make_A_self(V)                                 # K=1
    A_corr   = torch.cat([make_A_self(V), make_A_corr(V, corr_pkl)], dim=0)  # K=2
    A_caus   = torch.cat([make_A_self(V), make_A_causality(V, args.causality_thresh)], dim=0)  # K=2

    variants = [
        ("self",          A_self, "self_best.pt"),
        ("self_corr",     A_corr, "self_corr_best.pt"),
        (f"self_caus_{args.causality_thresh}", A_caus, f"self_caus_{args.causality_thresh}_best.pt"),
    ]

    results = []
    for tag, A, ckpt_name in variants:
        # logging dirs
        run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir  = Path(args.run_root) / f"{tag}_experiment" / run_id
        ckpt_dir = Path(args.ckpt_dir);    log_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer   = SummaryWriter(log_dir=str(log_dir))
        print(f"\n===== Training variant: {tag} (K={A.size(0)}) =====")
        # pretty print adjacency
        names = ["A_self"] if A.size(0)==1 else (["A_self","A_corr"] if tag.startswith("self_corr") else ["A_self",f"A_caus(thr>{args.causality_thresh})"])
        print_adj_binary(A, names, sep="")

        # build model
        model = build_model(model_name, cfg, device, A=A.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]*max(1,steps_per_epoch))
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg["use_amp"] and device.type=="cuda"))

        best_val = float("inf")
        for epoch in range(cfg["epochs"]):
            t0 = time.time()
            tr_loss, tr_mse, tr_prec = train_one_epoch(
                model, train_loader, optimizer, device,
                status_classes=cfg["status_classes"],
                rul_weight=cfg["rul_weight"], status_weight=cfg["status_weight"],
                scaler=scaler, scheduler=scheduler, grad_clip=cfg["grad_clip"]
            )
            va_loss, va_mse, va_prec, va_rmse = evaluate(
                model, val_loader, device,
                status_classes=cfg["status_classes"],
                rul_weight=cfg["rul_weight"], status_weight=cfg["status_weight"]
            )
            # logging
            writer.add_scalar("total_loss/train_total", tr_loss, epoch)
            writer.add_scalar("total_loss/val_total",   va_loss, epoch)
            writer.add_scalar("rul_loss/train",         tr_mse,  epoch)
            writer.add_scalar("rul_loss/val",           va_mse,  epoch)
            writer.add_scalar("rul_loss/rmse_val",      va_rmse, epoch)
            for i,p in enumerate(tr_prec): writer.add_scalar(f"status{i}_precision/train", p, epoch)
            for i,p in enumerate(va_prec): writer.add_scalar(f"status{i}_precision/val",   p, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            dt = time.time() - t0
            prec_str = " ".join([f"S{i}_P:{va_prec[i]*100:.1f}%" for i in range(len(cfg["status_classes"]))])
            print(f"[{tag}] Epoch {epoch+1:04d}/{cfg['epochs']}: "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_RMSE={va_rmse:.4f} {prec_str} ({dt:.1f}s)")

            if va_loss < best_val:
                best_val = va_loss
                best_path = ckpt_dir / ckpt_name
                torch.save(
                    {"epoch":epoch, "model_state":model.state_dict(),
                     "optimizer_state":optimizer.state_dict(),
                     "scheduler_state":scheduler.state_dict(),
                     "cfg":cfg, "best_val_loss":best_val},
                    best_path
                )
                print(f"  ↳ saved best checkpoint to {best_path}")

        writer.flush(); writer.close()
        results.append((tag, best_val))

    print("\n=== Simple ablation summary (lower val loss is better) ===")
    for tag, val in sorted(results, key=lambda x: x[1]):
        print(f"{tag:>18s}: best_val_loss = {val:.6f}")

if __name__ == "__main__":
    main()
