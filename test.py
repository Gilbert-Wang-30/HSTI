
# test.py
# -*- coding: utf-8 -*-
"""
Evaluate best checkpoints of all backbones on the TEST set and print comparisons.
- Loads checkpoints: checkpoints/stgcn/<model>_best.pt
- Uses pickles from configs YAML (expects test_stgcn.pkl)
- Metrics over the entire test set:
    total loss (MSE + sum CE), RUL MSE/RMSE,
    per-head micro accuracy, per-head macro precision
- Also reports: parameter count, model size (MB), forward speed (avg ms / throughput)
- Prints per-model metrics and a ranked summary.

Usage:
  python3 test.py --config configs/stgcn.yaml
  python3 test.py --models stgcn,tcn,dcrnn,stgcn_no_attention,inception_time --device cuda
"""

import os
import math
import time
import yaml
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.stgcn_data_loader import STGCNDataset  # noqa: F401 (kept for clarity / parity)

BASE_DIR = Path(__file__).resolve().parent

# ---------- Config ----------
def load_yaml_config(default_cfg: Dict[str, Any], yaml_path: Path) -> Dict[str, Any]:
    cfg = default_cfg.copy()
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    print("Using:", cfg["train_pkl"], cfg["val_pkl"], cfg.get("test_pkl", "<no test_pkl in YAML>"))
    return cfg

def seed_everything(seed: int) -> None:
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ---------- Metrics ----------
def micro_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())

def macro_precision(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    precs = []
    eps = 1e-8
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        precs.append(tp / (tp + fp + eps))
    return float(sum(precs) / len(precs))

def check_finite(tag: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
        print(f"[NaN/Inf] in {tag} at indices: {bad[:5].tolist()} ...")
        raise RuntimeError(f"{tag} contains NaN/Inf")

# ---------- Adjacency (matches train) ----------
def make_partitioned_adj_from_specs(V: int, corr_pkl_path: Path) -> torch.Tensor:
    sensors_100hz = ["PS1","PS2","PS3","PS4","PS5","PS6","EPS1"]
    sensors_10hz  = ["FS1","FS2"]
    sensors_1hz   = ["TS1","TS2","TS3","TS4","VS1","SE","CE","CP"]
    assert V == (len(sensors_100hz) + len(sensors_10hz) + len(sensors_1hz)), "V must be 17."
    idx_100 = list(range(0, len(sensors_100hz)))
    idx_10  = list(range(idx_100[-1] + 1, idx_100[-1] + 1 + len(sensors_10hz)))
    idx_1   = list(range(idx_10[-1] + 1, idx_10[-1] + 1 + len(sensors_1hz)))

    A_self = torch.eye(V, dtype=torch.float32)
    A_freq = torch.zeros(V, V, dtype=torch.float32)
    for group in [idx_100, idx_10, idx_1]:
        g = torch.tensor(group, dtype=torch.long)
        A_freq[g[:, None], g[None, :]] = 1.0
        A_freq[g, g] = 0.0

    with open(corr_pkl_path, "rb") as f:
        co = pickle.load(f)
    A_corr = co if isinstance(co, torch.Tensor) else torch.tensor(co, dtype=torch.float32)
    A_corr = 0.5 * (A_corr + A_corr.t()); A_corr.fill_diagonal_(0.0)

    return torch.stack([A_self, A_freq, A_corr], dim=0)

def model_needs_graph(name: str) -> bool:
    n = name.lower()
    return n in ("stgcn", "stgcn_no_attention", "dcrnn")

# ---------- Model factory (matches your train factory) ----------
def build_model(name: str, cfg: Dict[str, Any], device: torch.device, A: torch.Tensor = None) -> nn.Module:
    name = name.lower()
    status_classes = cfg["status_classes"]; in_channels = cfg["in_channels"]
    V = 17; T = cfg.get("T", 6)

    if name == "stgcn":
        from models.stgcn import STGCN
        model = STGCN(A=A, status_classes=status_classes, in_channels=in_channels,
                      channels=tuple(cfg["channels"]), temporal_kernel=tuple(cfg["temporal_kernel"]),
                      dropout=cfg["dropout"], edge_importance=cfg["edge_importance"])
    elif name == "stgcn_no_attention":
        from models.stgcn_no_attention import STGCN_NoAttention
        model = STGCN_NoAttention(A=A, status_classes=status_classes, in_channels=in_channels,
                                  channels=tuple(cfg["channels"]), temporal_kernel=tuple(cfg["temporal_kernel"]),
                                  dropout=cfg["dropout"], edge_importance=cfg["edge_importance"])
    elif name == "tcn":
        from models.tcn import TCNBackbone
        model = TCNBackbone(status_classes=status_classes, in_channels=in_channels, V=V,
                            tcn_channels=tuple(cfg.get("tcn_channels", cfg["channels"])),
                            kernel_size=int(cfg.get("tcn_kernel", 3)),
                            dilations=tuple(cfg.get("tcn_dilations", [1,2,4])),
                            dropout=float(cfg.get("tcn_dropout", 0.1)),
                            causal=bool(cfg.get("tcn_causal", False)),
                            head_hidden=256, head_dropout=0.2, pool="mean")
    elif name == "dcrnn":
        from models.dcrnn import DCRNNBackbone
        model = DCRNNBackbone(A=A, in_channels=in_channels, T=T, V=V,
                              status_classes=status_classes,
                              hidden=cfg.get("dcrnn_hidden", 64),
                              num_layers=cfg.get("dcrnn_layers", 2),
                              k=cfg.get("dcrnn_k", 1), dropout=0.1,
                              head_hidden=256, head_dropout=0.2)
    elif name == "inception_time":
        from models.inception_time import InceptionTimeBackbone
        model = InceptionTimeBackbone(in_channels=in_channels, V=V, T=T, status_classes=status_classes,
                                      nb_filters=cfg.get("it_filters", 32),
                                      depth=cfg.get("it_depth", 6),
                                      bottleneck_channels=cfg.get("it_bottleneck", 32),
                                      kernel_sizes=tuple(cfg.get("it_kernels", [3,5,7])),
                                      dropout=0.1, head_hidden=256, head_dropout=0.2, pool="mean")
    else:
        raise ValueError(f"Unknown model {name}")
    return model.to(device)

# ---------- Param & speed ----------
def count_params_and_size_mb(m: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # bytes = sum(p.numel()*p.element_size() for p in m.parameters())
    # Use element_size to be dtype-aware:
    bytes_total = 0
    for p in m.parameters():
        bytes_total += p.numel() * p.element_size()
    size_mb = bytes_total / (1024**2)
    return total, trainable, size_mb

def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def benchmark_forward(model: nn.Module,
                      x: torch.Tensor,
                      device: torch.device,
                      warmup: int = 5,
                      iters: int = 20) -> Tuple[float, float]:
    """Return (avg_ms_per_forward, throughput_samples_per_s) on a single batch."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    sync_if_cuda(device)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / max(1, iters)
    thr = (x.size(0) * iters) / ((avg_ms * iters) / 1000.0)
    return avg_ms, thr

# ---------- Evaluation ----------
@torch.no_grad()
def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   status_classes: List[int]) -> Dict[str, Any]:
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn  = nn.CrossEntropyLoss(label_smoothing=0.05)

    total_samples = 0
    sum_total_loss = 0.0
    sum_mse = 0.0

    # per-head CE weighted by samples
    sum_ce = [0.0]*len(status_classes)

    # per-head micro accuracy counts
    correct = [0]*len(status_classes)

    # per-head macro precision components (TP/FP per class)
    tp = [torch.zeros(n, dtype=torch.long) for n in status_classes]
    fp = [torch.zeros(n, dtype=torch.long) for n in status_classes]

    for batch in loader:
        if not (isinstance(batch, (tuple, list)) and len(batch) == 3):
            raise RuntimeError(f"Expected (x,rul,status); got {type(batch).__name__} len={len(batch) if hasattr(batch,'__len__') else 'NA'}")
        x, rul, status = batch
        x = x.to(device).float()
        rul = rul.to(device).float().view(-1,1)
        status = status.to(device).long()

        out = model(x)
        rul_pred = out["rul"]
        logits = out["status_logits"]

        loss_rul = mse_loss_fn(rul_pred, rul)
        loss_status = 0.0

        bs = x.size(0)
        total_samples += bs
        sum_mse += float(loss_rul.item()) * bs

        for j, ncls in enumerate(status_classes):
            ce = ce_loss_fn(logits[j], status[:, j])
            sum_ce[j] += float(ce.item()) * bs
            loss_status += ce

            preds = logits[j].argmax(dim=1)
            correct[j] += int((preds == status[:, j]).sum().item())

            for c in range(ncls):
                tp[j][c] += int(((preds == c) & (status[:, j] == c)).sum().item())
                fp[j][c] += int(((preds == c) & (status[:, j] != c)).sum().item())

        total = float((loss_rul + loss_status).item()) * bs
        sum_total_loss += total

    avg_total_loss = sum_total_loss / max(1, total_samples)
    avg_mse = sum_mse / max(1, total_samples)
    rmse = math.sqrt(avg_mse)

    ce_avg = [c / max(1, total_samples) for c in sum_ce]
    micro_acc = [correct[j] / max(1, total_samples) for j in range(len(status_classes))]

    macro_prec = []
    eps = 1e-8
    for j, ncls in enumerate(status_classes):
        precs = []
        for c in range(ncls):
            precs.append(tp[j][c].item() / (tp[j][c].item() + fp[j][c].item() + eps))
        macro_prec.append(sum(precs) / len(precs))

    return {
        "avg_total_loss": avg_total_loss,
        "rul_mse": avg_mse,
        "rul_rmse": rmse,
        "status_ce": ce_avg,
        "status_micro_acc": micro_acc,
        "status_macro_prec": macro_prec,
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(BASE_DIR / "configs" / "stgcn.yaml"))
    ap.add_argument("--models", type=str, default="stgcn,stgcn_no_attention,tcn,dcrnn,inception_time")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    args = ap.parse_args()

    default_cfg = {
        "seed": 42,
        "device": args.device,
        "batch_size": args.batch_size,
        "status_classes": [3,4,3,4],
        "in_channels": 24,
        "T": 6,
        # paths
        "train_pkl": str(BASE_DIR / "data" / "processed" / "train_stgcn.pkl"),
        "val_pkl":   str(BASE_DIR / "data" / "processed" / "val_stgcn.pkl"),
        "test_pkl":  str(BASE_DIR / "data" / "processed" / "test_stgcn.pkl"),
        # backbone defaults (used if YAML missing)
        "channels": [64,32,10],
        "temporal_kernel": [3,3,2],
        "dropout": 0.0,
        "edge_importance": False,
        "tcn_channels": [64,32,10], "tcn_dilations": [1,2,4], "tcn_kernel": 3, "tcn_dropout": 0.1, "tcn_causal": False,
        "dcrnn_hidden": 64, "dcrnn_layers": 2, "dcrnn_k": 1,
        "it_filters": 32, "it_depth": 6, "it_bottleneck": 32, "it_kernels": [3,5,7],
    }
    cfg = load_yaml_config(default_cfg, Path(args.config))
    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"])

    # Load TEST set
    with open(cfg["test_pkl"], "rb") as f:
        test_dataset = pickle.load(f)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Prepare one batch for speed benchmark (first batch of test set)
    try:
        bench_batch = next(iter(test_loader))
        if isinstance(bench_batch, (tuple, list)) and len(bench_batch) == 3:
            x_bench = bench_batch[0].to(device).float()
        else:
            raise RuntimeError("Unexpected batch structure for speed benchmark.")
    except StopIteration:
        x_bench = None

    # Build adjacency if any requested model needs it
    names = [m.strip() for m in args.models.split(",")]
    need_graph = any(model_needs_graph(n) for n in names)
    A = None
    if need_graph:
        corr_pkl = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"
        if corr_pkl.exists():
            A = make_partitioned_adj_from_specs(17, corr_pkl).to(device)
        else:
            print("[WARN] correlation not found; using identity-only adjacency")
            A = torch.stack([torch.eye(17, device=device)]*3, dim=0)

    # Evaluate each model with its best checkpoint
    results = []
    for name in names:
        print("\n" + "="*90)
        print(f"[MODEL] {name} — loading best checkpoint and evaluating on TEST")
        print("="*90)

        model = build_model(name, cfg, device, A=A if model_needs_graph(name) else None)

        ckpt_path = BASE_DIR / "checkpoints" / "stgcn" / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"[LOAD] No checkpoint found at {ckpt_path}. Skipping.")
            continue
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"], strict=False)
            print(f"[LOAD] Loaded {ckpt_path}")
        except Exception as e:
            print(f"[LOAD] Failed to load {ckpt_path}: {e}")
            continue

        # Params and size
        total_params, trainable_params, size_mb = count_params_and_size_mb(model)
        print(f"Params: total={total_params:,}  trainable={trainable_params:,}  (~{size_mb:.2f} MB)")

        # Speed benchmark (single batch)
        if x_bench is not None:
            fwd_ms, fwd_thr = benchmark_forward(model, x_bench, device,
                                                warmup=args.bench_warmup, iters=args.bench_iters)
            print(f"Forward speed: {fwd_ms:.3f} ms/batch  |  throughput: {fwd_thr:.1f} samples/s")
        else:
            fwd_ms, fwd_thr = float('nan'), float('nan')
            print("Forward speed: N/A (empty test loader)")

        # Complete TEST evaluation
        metrics = evaluate_model(model, test_loader, device, cfg["status_classes"])

        # Print per-model metrics
        print(f"Total loss (test): {metrics['avg_total_loss']:.4f}")
        print(f"RUL MSE / RMSE   : {metrics['rul_mse']:.6f} / {metrics['rul_rmse']:.6f}")
        for j, (acc, mp) in enumerate(zip(metrics["status_micro_acc"], metrics["status_macro_prec"])):
            print(f"status{j}: micro_acc={acc*100:.2f}%  macro_prec={mp*100:.2f}%  CE={metrics['status_ce'][j]:.4f}")

        results.append({
            "name": name,
            "params": total_params,
            "size_mb": size_mb,
            "fwd_ms": fwd_ms,
            "fwd_thr": fwd_thr,
            **metrics
        })

    # Ranked summary
    print("\n" + "="*90)
    print("SUMMARY (TEST) — lower loss/RMSE better; higher accuracy/precision/speed better")
    print("="*90)

    if not results:
        print("No models evaluated. (Missing checkpoints?)")
        return

    # total loss
    print("\nTotal loss (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["avg_total_loss"])):
        print(f"{i+1}. {r['name']} {r['avg_total_loss']:.4f}")

    # RMSE
    print("\nRUL RMSE (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["rul_rmse"])):
        print(f"{i+1}. {r['name']} {r['rul_rmse']:.6f}")

    # per-head micro acc (desc)
    H = len(cfg["status_classes"])
    for j in range(H):
        print(f"\nstatus{j} micro accuracy (desc):")
        for i, r in enumerate(sorted(results, key=lambda r: r["status_micro_acc"][j], reverse=True)):
            print(f"{i+1}. {r['name']} {r['status_micro_acc'][j]*100:.2f}%")

    # per-head macro precision (desc)
    for j in range(H):
        print(f"\nstatus{j} macro precision (desc):")
        for i, r in enumerate(sorted(results, key=lambda r: r["status_macro_prec"][j], reverse=True)):
            print(f"{i+1}. {r['name']} {r['status_macro_prec'][j]*100:.2f}%")

    # Parameters and size
    print("\nParameter count (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["params"])):
        print(f"{i+1}. {r['name']} {r['params']:,} params (~{r['size_mb']:.2f} MB)")

    print("\nModel size MB (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["size_mb"])):
        print(f"{i+1}. {r['name']} ~{r['size_mb']:.2f} MB")

    # Speed
    print("\nForward avg ms/batch (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: (float('inf') if math.isnan(r['fwd_ms']) else r['fwd_ms']))):
        print(f"{i+1}. {r['name']} {r['fwd_ms']:.3f} ms")

    print("\nForward throughput (desc):")
    for i, r in enumerate(sorted(results, key=lambda r: (-1.0 if math.isnan(r['fwd_thr']) else r['fwd_thr']), reverse=True)):
        print(f"{i+1}. {r['name']} {r['fwd_thr']:.1f} samples/s")

if __name__ == "__main__":
    main()
