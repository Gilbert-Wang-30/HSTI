# train.py
# -*- coding: utf-8 -*-

import os, math, time, yaml, pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.stgcn_data_loader import STGCNDataset  

# dataset pickles are the same ones used by stgcn_train.py
BASE_DIR = Path(__file__).resolve().parent

from models.ll import MultiTaskLL  # <— the LL baseline

# -----------------------
# Utilities
# -----------------------
def load_yaml_config(default_cfg: Dict[str, Any], yaml_path: Path) -> Dict[str, Any]:
    cfg = default_cfg.copy()
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg

def seed_everything(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def macro_precision_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    precs, eps = [], 1e-8
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        precs.append(tp / (tp + fp + eps))
    return float(sum(precs) / len(precs))

def _check_finite(tag: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
        print(f"[NaN/Inf] in {tag} at indices: {bad[:5].tolist()} ...")
        raise RuntimeError(f"{tag} contains NaN/Inf")

# -----------------------
# Training / Eval
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
                    ce = ce_loss_fn(logits_list[i], status[:, i])
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
# Main
# -----------------------
def main():
    default_cfg = {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "epochs": 5000,                 # match your ST-GCN long run
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "use_amp": False,
        "val_every": 1,
        "status_classes": [3, 4, 3, 4],
        "in_channels": 24,
        "T": 6,
        "V": 17,
        "dropout": 0.2,                 # same as heads in ST-GCN
        "rul_weight": 1.0,
        "status_weight": 1.0,
        "num_workers": 4,
        "pin_memory": True,
        # paths (same pickles as stgcn_train.py)
        "train_pkl": str(BASE_DIR / "data" / "processed" / "train_stgcn.pkl"),
        "val_pkl":   str(BASE_DIR / "data" / "processed" / "val_stgcn.pkl"),
        "test_pkl":  str(BASE_DIR / "data" / "processed" / "test_stgcn.pkl"),
        "run_root":  str(BASE_DIR / "runs" / "ll_experiment"),  # tags are same; dir is different
        "ckpt_dir":  str(BASE_DIR / "checkpoints" / "ll"),
    }

    cfg_path = BASE_DIR / "configs" / "ll.yaml"
    cfg = load_yaml_config(default_cfg, cfg_path)

    seed_everything(cfg["seed"])
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg["device"])

    # logging dirs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg["run_root"]) / run_id
    ckpt_dir = Path(cfg["ckpt_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print("TensorBoard:", writer.log_dir)

    # data
    with open(cfg["train_pkl"], "rb") as f: train_dataset = pickle.load(f)
    with open(cfg["val_pkl"],   "rb") as f: val_dataset   = pickle.load(f)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"]
    )

    # model
    status_classes: List[int] = cfg["status_classes"]
    model = MultiTaskLL(
        in_channels=cfg["in_channels"], T=cfg["T"], V=cfg["V"],
        status_classes=status_classes, dropout=cfg["dropout"]
    ).to(device)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["use_amp"] and device.type == "cuda"))

    # train
    best_val_loss = float("inf")
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        tr_loss, tr_mse, tr_precs, tr_ce = train_one_epoch(
            model, train_loader, optimizer, device,
            status_classes=status_classes,
            rul_weight=cfg["rul_weight"], status_weight=cfg["status_weight"],
            scaler=scaler, scheduler=scheduler, grad_clip=cfg["grad_clip"]
        )
        va_loss, va_mse, va_precs, va_rmse, va_ce = evaluate(
            model, val_loader, device,
            status_classes=status_classes,
            rul_weight=cfg["rul_weight"], status_weight=cfg["status_weight"]
        )

        # TensorBoard — SAME TAGS as stgcn_train.py so curves overlay
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
        prec_str = " ".join([f"S{i}_P:{va_precs[i]*100:.1f}%" for i in range(len(status_classes))])
        print(f"Epoch {epoch+1:04d}/{cfg['epochs']}: "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_RMSE={va_rmse:.4f} {prec_str} ({dt:.1f}s)")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_path = ckpt_dir / "ll_best.pt"
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

    writer.flush(); writer.close()

if __name__ == "__main__":
    main()
