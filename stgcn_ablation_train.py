# stgcn_ablation_train.py
# -*- coding: utf-8 -*-

"""
Ablation training for ST-GCN (residual-attention variant) on HSTI.

This version COMPLETELY REMOVES the frequency partition.
Base adjacency now has only TWO layers:
  [0] A_self (identity)
  [1] A_corr (binary correlation, sym, no self)
We then add ONE pooled causality layer:
  [2] A_caus (ONE pooled causality layer built from 238x238 PCMCI)

We train THREE ablation models, each with ONE of these partitions zeroed:
  "no_self"        -> A[0] = 0
  "no_corr"        -> A[1] = 0
  "no_causality"   -> A[2] = 0

Why zero (not drop) a layer?
- K stays 3 for all ablations => identical parameter counts and architecture.
- Only the information content of that partition is removed.

Logging / Checkpoints:
  runs/stgcn_ablation/<variant>_experiment/<timestamp>
  checkpoints/stgcn_ablation/<variant>_best.pt

Usage (typical):
  python3 stgcn_ablation_train.py --device cuda
  python3 stgcn_ablation_train.py --epochs 2000 --batch-size 64 --device cuda

By default, this script:
- Uses the same pickles as stgcn_train.py (train_stgcn.pkl / val_stgcn.pkl)
- Uses STGCN_residual_attention backbone only
- Builds K=3 adjacency (self + corr + pooled-causality), then zeros one partition per run
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
# Adjacency builders (NO FREQUENCY PARTITION)
# -----------------------
def make_partitioned_base2(V: int, corr_pkl_path: Path) -> torch.Tensor:
    """Return K=2 base partitions: identity (self), correlation (sym, no self)."""
    A_self = torch.eye(V, dtype=torch.float32)

    with open(corr_pkl_path, "rb") as f:
        co = pickle.load(f)
    A_corr = co if isinstance(co, torch.Tensor) else torch.tensor(co, dtype=torch.float32)
    if A_corr.shape != (V, V):
        raise ValueError(f"Correlation matrix shape {A_corr.shape} != ({V}, {V})")
    A_corr = 0.5 * (A_corr + A_corr.t())
    A_corr.fill_diagonal_(0.0)

    return torch.stack([A_self, A_corr], dim=0)  # (2,V,V)

def add_one_causality_layer(A_base2: torch.Tensor, V: int, caus_path: Path) -> torch.Tensor:
    """
    Append ONE pooled causality layer to base A (K=2 -> K=3).
    |C| (238x238) -> avg_pool2d(kernel=14,stride=14) -> (17x17) -> >0.5 -> 1/0 -> diag=0.
    Assumes sensor-major feature ordering in 238x238 PCMCI matrix.
    """
    if not caus_path.exists():
        print(f"[WARN] causality file not found at {caus_path}, keeping K=2")
        return A_base2
    with open(caus_path, "rb") as f:
        C = pickle.load(f)
    C = C if isinstance(C, torch.Tensor) else torch.tensor(C, dtype=torch.float32)
    if C.shape != (V*14, V*14):
        raise ValueError(f"Causality matrix must be {(V*14, V*14)}; got {tuple(C.shape)}")

    C_abs = C.abs().unsqueeze(0).unsqueeze(0)  # (1,1,238,238)
    A_caus = F.avg_pool2d(C_abs, kernel_size=(14,14), stride=(14,14)).squeeze(0).squeeze(0)  # (17,17)
    A_caus = (A_caus > 0.5).to(torch.float32)
    A_caus.fill_diagonal_(0.0)
    return torch.cat([A_base2, A_caus.unsqueeze(0)], dim=0)  # (3,V,V)

def zero_one_partition(A3: torch.Tensor, idx: int) -> torch.Tensor:
    """Return a copy of A3 (K=3) with partition `idx` zeroed out (keeps K=3, params constant)."""
    A = A3.clone()
    A[idx].zero_()
    return A

# -----------------------
# Model factory
# -----------------------
def build_residual_attention_stgcn(cfg: Dict[str, Any], device: torch.device, A: torch.Tensor) -> nn.Module:
    """
    Builds STGCN_residual_attention with the provided adjacency (K must match A.shape[0]).
    """
    from models.stgcn_residual_attention import STGCN_residual_attention
    model = STGCN_residual_attention(
        A=A,
        status_classes=cfg["status_classes"],
        in_channels=cfg["in_channels"],
        channels=tuple(cfg["channels"]),
        temporal_kernel=tuple(cfg["temporal_kernel"]),
        dropout=cfg["dropout"],
        edge_importance=cfg["edge_importance"],
    )
    return model.to(device)

# -----------------------
# Training / Eval loops
# -----------------------
def _check_finite(tag: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
        print(f"[NaN/Inf] in {tag} at indices: {bad[:5].tolist()} ...")
        raise RuntimeError(f"{tag} contains NaN/Inf")

def macro_precision_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    precs, eps = [], 1e-8
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        precs.append(tp / (tp + fp + eps))
    return float(sum(precs) / len(precs))

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
) -> Tuple[float, float, List[float], List[float]]:
    model.train()
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    ce_sums = [0.0 for _ in status_classes]
    total_loss = total_mse = 0.0
    prec_sums = [0.0 for _ in status_classes]
    num_batches = 0

    for x, rul, status in loader:
        x = x.to(device, non_blocking=True)                   # (N,C,T,V)
        rul = rul.to(device, non_blocking=True).float().view(-1, 1)
        status = status.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        _check_finite("x", x)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                out = model(x)
                rul_pred = out["rul"]
                logits_list = out["status_logits"]
                _check_finite("rul_pred", rul_pred)
                for i, logits in enumerate(logits_list):
                    _check_finite(f"status_logits[{i}]", logits)

                loss_rul = mse_loss_fn(rul_pred, rul)
                loss_status = 0.0
                for i, num_cls in enumerate(status_classes):
                    ce = nn.functional.cross_entropy(logits_list[i], status[:, i], label_smoothing=0.05)
                    ce_sums[i] += float(ce.item())
                    loss_status = loss_status + ce
                loss = rul_weight * loss_rul + status_weight * loss_status
        else:
            out = model(x)
            rul_pred = out["rul"]
            logits_list = out["status_logits"]
            loss_rul = mse_loss_fn(rul_pred, rul)
            loss_status = 0.0
            for i, num_cls in enumerate(status_classes):
                ce = ce_loss_fn(logits_list[i], status[:, i])
                ce_sums[i] += float(ce.item())
                loss_status = loss_status + ce
            loss = rul_weight * loss_rul + status_weight * loss_status

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_mse  += float(loss_rul.item())
        for i, num_cls in enumerate(status_classes):
            prec_sums[i] += macro_precision_from_logits(logits_list[i].detach(), status[:, i], num_cls)
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_mse  = total_mse  / max(1, num_batches)
    avg_precs = [p / max(1, num_batches) for p in prec_sums]
    avg_ces   = [c / max(1, num_batches) for c in ce_sums]
    return avg_loss, avg_mse, avg_precs, avg_ces

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    status_classes: List[int],
    rul_weight: float,
    status_weight: float,
) -> Tuple[float, float, List[float], float, List[float]]:
    model.eval()
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn  = nn.CrossEntropyLoss(label_smoothing=0.05)

    ce_sums = [0.0 for _ in status_classes]
    total_loss = total_mse = 0.0
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
            ce = ce_loss_fn(logits_list[i], status[:, i])
            ce_sums[i] += float(ce.item())
            loss_status = loss_status + ce

        loss = rul_weight * loss_rul + status_weight * loss_status

        total_loss += float(loss.item())
        total_mse  += float(loss_rul.item())
        for i, num_cls in enumerate(status_classes):
            prec_sums[i] += macro_precision_from_logits(logits_list[i], status[:, i], num_cls)
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_mse  = total_mse  / max(1, num_batches)
    avg_precs = [p / max(1, num_batches) for p in prec_sums]
    avg_ces   = [c / max(1, num_batches) for c in ce_sums]
    rmse_val = math.sqrt(avg_mse) if avg_mse >= 0.0 else float('nan')
    return avg_loss, avg_mse, avg_precs, rmse_val, avg_ces

# -----------------------
# Main (ablations loop)
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
    # data paths (same as stgcn_train)
    ap.add_argument("--train-pkl", type=str, default=str(BASE_DIR / "data" / "processed" / "train_stgcn.pkl"))
    ap.add_argument("--val-pkl",   type=str, default=str(BASE_DIR / "data" / "processed" / "val_stgcn.pkl"))
    ap.add_argument("--run-root",  type=str, default=str(BASE_DIR / "runs" / "stgcn_ablation"))
    ap.add_argument("--ckpt-dir",  type=str, default=str(BASE_DIR / "checkpoints" / "stgcn_ablation"))
    args = ap.parse_args()

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

    def seed_everything(seed: int) -> None:
        import random, numpy as np
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    seed_everything(cfg["seed"])
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg["device"])

    # data
    with open(args.train_pkl, "rb") as f: train_dataset = pickle.load(f)
    with open(args.val_pkl,   "rb") as f: val_dataset   = pickle.load(f)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]
    )

    # Build K=3 adjacency once (base2 + 1 pooled causality)
    V = 17
    corr_pkl = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"
    caus_pkl = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    A2 = make_partitioned_base2(V, corr_pkl)         # (2,17,17): self + corr
    A3 = add_one_causality_layer(A2, V, caus_pkl).to(device)  # (3,17,17)

    # Ablation variants: zero one partition each time (0=self, 1=corr, 2=causality)
    ablations = [
        ("no_self",        0),
        ("no_corr",        1),
        ("no_causality",   2),
    ]

    results_summary = []

    for tag, idx in ablations:
        # logging dirs for this ablation
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir  = Path(args.run_root) / f"{tag}_experiment" / run_id
        ckpt_dir = Path(args.ckpt_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"\n===== Ablation: {tag} (zero partition index {idx}) =====")
        print("TensorBoard:", writer.log_dir)

        # zero the chosen partition (keep K=3)
        A_abl = zero_one_partition(A3, idx)

        # build model
        model = build_residual_attention_stgcn(cfg, device, A=A_abl)

        # optimizer & scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps_per_epoch = len(train_loader)
        total_steps = cfg["epochs"] * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg["use_amp"] and device.type == "cuda"))

        # train loop
        best_val_loss = float("inf")
        for epoch in range(cfg["epochs"]):
            t0 = time.time()
            tr_loss, tr_mse, tr_precs, tr_ce = train_one_epoch(
                model, train_loader, optimizer, device,
                status_classes=cfg["status_classes"],
                rul_weight=cfg["rul_weight"], status_weight=cfg["status_weight"],
                scaler=scaler, scheduler=scheduler, grad_clip=cfg["grad_clip"]
            )
            va_loss, va_mse, va_precs, va_rmse, va_ce = evaluate(
                model, val_loader, device,
                status_classes=cfg["status_classes"],
                rul_weight=cfg["rul_weight"], status_weight=cfg["status_weight"]
            )

            # logging (same tags as main trainer for easy overlay)
            writer.add_scalar("total_loss/train_total", tr_loss, epoch)
            writer.add_scalar("total_loss/val_total",   va_loss, epoch)
            writer.add_scalar("rul_loss/train",         tr_mse,  epoch)
            writer.add_scalar("rul_loss/val",           va_mse,  epoch)
            writer.add_scalar("rul_loss/rmse_val",      va_rmse, epoch)
            for i, p in enumerate(tr_precs):
                writer.add_scalar(f"status{i}_precision/train", p, epoch)
                writer.add_scalar(f"status{i}_ce/train", tr_ce[i], epoch)
            for i, p in enumerate(va_precs):
                writer.add_scalar(f"status{i}_precision/val", p, epoch)
                writer.add_scalar(f"status{i}_ce/val",   va_ce[i], epoch)
            writer.add_scalar("status_ce/train_sum", sum(tr_ce), epoch)
            writer.add_scalar("status_ce/val_sum",   sum(va_ce), epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            dt = time.time() - t0
            prec_str = " ".join([f"S{i}_P:{va_precs[i]*100:.1f}%" for i in range(len(cfg["status_classes"]))])
            print(f"[{tag}] Epoch {epoch+1:04d}/{cfg['epochs']}: "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                  f"val_RMSE={va_rmse:.4f} {prec_str} ({dt:.1f}s)")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_path = ckpt_dir / f"{tag}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "cfg": cfg,
                        "best_val_loss": best_val_loss,
                        "ablated_partition": idx,
                    },
                    best_path,
                )
                print(f"  ↳ saved best checkpoint to {best_path}")

        writer.flush(); writer.close()
        results_summary.append((tag, best_val_loss))

    # print quick summary
    print("\n=== Ablation summary (lower is better) ===")
    for tag, loss in sorted(results_summary, key=lambda x: x[1]):
        print(f"{tag:>15s}  best_val_loss={loss:.6f}")

if __name__ == "__main__":
    main()
