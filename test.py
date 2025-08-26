# test.py
# -*- coding: utf-8 -*-
"""
Evaluate best checkpoints of all backbones on the TEST set and print comparisons.
- Loads checkpoints from one or more roots (e.g., checkpoints/stgcn, checkpoints/with_causalities,
  checkpoints/with_1_causality, checkpoints/stgcn_ablation, checkpoints/causality_test)
- Supports suffixed filenames like: <model>_<suffix>_best.pt  (e.g., thresholds 0.1, 0.75, etc.)
- Infers K (=2/3/17) from checkpoint and builds a matching test adjacency.
- Reports: total loss, RUL MSE/RMSE, per-head micro acc & macro precision, params, size, speed.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.stgcn_data_loader import STGCNDataset  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent

# ---------------- Config / utils ----------------
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

def micro_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())

def macro_precision(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    precs, eps = [], 1e-8
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

# ---------------- Adjacency builders ----------------
def make_partitioned_base3(V: int, corr_pkl_path: Path) -> torch.Tensor:
    """Back-compat base3 (self + freq + corr)."""
    # identity
    A_self = torch.eye(V, dtype=torch.float32)
    # freq blocks
    idx_100 = list(range(0, 7))
    idx_10  = list(range(7, 9))
    idx_1   = list(range(9, 17))
    A_freq = torch.zeros(V, V, dtype=torch.float32)
    for group in [idx_100, idx_10, idx_1]:
        g = torch.tensor(group, dtype=torch.long)
        A_freq[g[:, None], g[None, :]] = 1.0
        A_freq[g, g] = 0.0
    # correlation
    with open(corr_pkl_path, "rb") as f:
        co = pickle.load(f)
    A_corr = co if isinstance(co, torch.Tensor) else torch.tensor(co, dtype=torch.float32)
    if A_corr.shape != (V, V):
        raise ValueError(f"Correlation matrix shape {A_corr.shape} != ({V}, {V})")
    A_corr = 0.5 * (A_corr + A_corr.t())
    A_corr.fill_diagonal_(0.0)
    return torch.stack([A_self, A_freq, A_corr], dim=0)

def make_partitioned_base2(V: int, corr_pkl_path: Path) -> torch.Tensor:
    """Base2 (self + corr), no frequency partition."""
    A_self = torch.eye(V, dtype=torch.float32)
    with open(corr_pkl_path, "rb") as f:
        co = pickle.load(f)
    A_corr = co if isinstance(co, torch.Tensor) else torch.tensor(co, dtype=torch.float32)
    if A_corr.shape != (V, V):
        raise ValueError(f"Correlation matrix shape {A_corr.shape} != ({V}, {V})")
    A_corr = 0.5 * (A_corr + A_corr.t())
    A_corr.fill_diagonal_(0.0)
    return torch.stack([A_self, A_corr], dim=0)

def add_causality_layers(A_base: torch.Tensor, V: int) -> torch.Tensor:
    """Append 14 causality layers (K += 14)."""
    caus_path = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    if not caus_path.exists():
        print(f"[WARN] causality file not found at {caus_path}, skipping 14-layer causality")
        return A_base
    with open(caus_path, "rb") as f:
        C = pickle.load(f)
    C = torch.as_tensor(C, dtype=torch.float32)
    if C.shape != (V*14, V*14):
        raise ValueError(f"Causality matrix must be {(V*14, V*14)}; got {tuple(C.shape)}")
    layers = []
    for feat in range(14):
        idx = torch.tensor([s*14 + feat for s in range(V)], dtype=torch.long)
        sub = C.index_select(0, idx).index_select(1, idx)
        A_feat = ((sub > 0.5) | (sub < -0.5)).to(torch.float32)
        A_feat.fill_diagonal_(0.0)
        layers.append(A_feat)
    return torch.cat([A_base, torch.stack(layers, dim=0)], dim=0)

def add_one_causality_layer(A_base: torch.Tensor, V: int, thresh: float = 0.5) -> torch.Tensor:
    """Append ONE pooled causality layer (K += 1)."""
    caus_path = BASE_DIR / "data" / "causality" / "pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl"
    if not caus_path.exists():
        print(f"[WARN] causality file not found at {caus_path}, skipping one-layer causality")
        return A_base
    with open(caus_path, "rb") as f:
        C = pickle.load(f)
    C = torch.as_tensor(C, dtype=torch.float32)
    if C.shape != (V*14, V*14):
        raise ValueError(f"Causality matrix must be {(V*14, V*14)}; got {tuple(C.shape)}")
    C_abs = C.abs().unsqueeze(0).unsqueeze(0)              # (1,1,238,238)
    A_caus = F.avg_pool2d(C_abs, kernel_size=(14,14), stride=(14,14)).squeeze(0).squeeze(0)  # (17,17)
    A_caus = (A_caus > thresh).to(torch.float32)
    A_caus.fill_diagonal_(0.0)
    return torch.cat([A_base, A_caus.unsqueeze(0)], dim=0)

def model_needs_graph(name: str) -> bool:
    n = name.lower()
    return n in ("stgcn", "stgcn_no_attention", "stgcn_residual_attention", "dcrnn")

# ---------------- Model factory ----------------
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
    elif name == "stgcn_residual_attention":
        from models.stgcn_residual_attention import STGCN_residual_attention
        model = STGCN_residual_attention(A=A, status_classes=status_classes, in_channels=in_channels,
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

# ---------------- Params / speed ----------------
def count_params_and_size_mb(m: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    bytes_total = sum(p.numel() * p.element_size() for p in m.parameters())
    size_mb = bytes_total / (1024**2)
    return total, trainable, size_mb

def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def benchmark_forward(model: nn.Module, x: torch.Tensor, device: torch.device,
                      warmup: int = 5, iters: int = 20) -> Tuple[float, float]:
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

# ---------------- Infer K from checkpoint ----------------
def infer_k_from_checkpoint(state: dict) -> int:
    ms = state.get("model_state", {})
    for key, tensor in ms.items():
        if key.endswith(".A") and hasattr(tensor, "dim") and tensor.dim() == 3:
            return int(tensor.size(0))
    return -1

# ---------------- Eval loop ----------------
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                   status_classes: List[int]) -> Dict[str, Any]:
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn  = nn.CrossEntropyLoss(label_smoothing=0.05)

    total_samples = 0
    sum_total_loss = 0.0
    sum_mse = 0.0
    sum_ce = [0.0]*len(status_classes)
    correct = [0]*len(status_classes)
    tp = [torch.zeros(n, dtype=torch.long) for n in status_classes]
    fp = [torch.zeros(n, dtype=torch.long) for n in status_classes]

    for batch in loader:
        if not (isinstance(batch, (tuple, list)) and len(batch) == 3):
            raise RuntimeError("Expected (x,rul,status); got something else")
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

        sum_total_loss += float((loss_rul + loss_status).item()) * bs

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

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(BASE_DIR / "configs" / "stgcn.yaml"))
    ap.add_argument("--models", type=str,
                    default="stgcn,stgcn_no_attention,stgcn_residual_attention,tcn,dcrnn,inception_time")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--ckpt-roots", type=str,
                    default="checkpoints/stgcn,checkpoints/with_causalities,checkpoints/with_1_causality,checkpoints/stgcn_ablation,checkpoints/causality_test",
                    help="comma-separated checkpoint roots to scan")
    args = ap.parse_args()

    default_cfg = {
        "seed": 42,
        "device": args.device,
        "batch_size": args.batch_size,
        "status_classes": [3,4,3,4],
        "in_channels": 24,
        "T": 6,
        "train_pkl": str(BASE_DIR / "data" / "processed" / "train_stgcn.pkl"),
        "val_pkl":   str(BASE_DIR / "data" / "processed" / "val_stgcn.pkl"),
        "test_pkl":  str(BASE_DIR / "data" / "processed" / "test_stgcn.pkl"),
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

    # Data
    with open(cfg["test_pkl"], "rb") as f:
        test_dataset = pickle.load(f)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Bench batch
    try:
        x_bench = next(iter(test_loader))[0].to(device).float()
    except StopIteration:
        x_bench = None

    ckpt_roots = [BASE_DIR / p.strip() for p in args.ckpt_roots.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]

    results = []
    corr_pkl = BASE_DIR / "data" / "correlation" / "binary_co_matrix.pkl"

    for root in ckpt_roots:
        # Collect candidates: exact + suffixed (e.g., _0.75_best.pt)
        def gather_candidates(model_name: str):
            candidates = []
            exact = root / f"{model_name}_best.pt"
            if exact.exists():
                candidates.append(("", exact))
            for f in root.glob(f"{model_name}_*_best.pt"):
                # suffix between model_ and _best.pt
                s = f.name[len(model_name) + 1 : -8]
                if s:  # avoid double-adding exact
                    candidates.append((s, f))
            return candidates

        for model_name in model_names:
            cand = gather_candidates(model_name)
            if not cand:
                print(f"\n== {model_name}@{root.name}: no matching checkpoints ==")
                continue

            for suffix, ckpt_path in cand:
                label = f"{model_name}[{suffix}]" if suffix else model_name
                print("\n" + "="*90)
                print(f"[MODEL] {label}@{root.name} — loading best checkpoint and evaluating on TEST")
                print("="*90)

                # Load checkpoint first to infer K
                try:
                    state = torch.load(ckpt_path, map_location=device)
                except Exception as e:
                    print(f"[LOAD] Failed to load {ckpt_path}: {e}")
                    continue

                # Build adjacency to MATCH K in checkpoint
                K_expected = infer_k_from_checkpoint(state)  # 2 / 3 / 17
                if not corr_pkl.exists():
                    print("[WARN] correlation not found; using identity-only fallback")
                    A_use = torch.stack([torch.eye(17)]* (K_expected if K_expected>0 else 2), dim=0).to(device)
                else:
                    if K_expected == 2:
                        # self + corr
                        A_use = make_partitioned_base2(17, corr_pkl_path=corr_pkl).to(device)
                    elif K_expected == 3:
                        # self + corr + ONE causality (threshold used here doesn't matter;
                        # checkpoint's buffers overwrite values; we only need matching SHAPE K=3)
                        A_use = add_one_causality_layer(make_partitioned_base2(17, corr_pkl), V=17, thresh=0.5).to(device)
                    elif K_expected == 17:
                        # back-compat older K=17 checkpoints
                        A_use = add_causality_layers(make_partitioned_base3(17, corr_pkl), V=17).to(device)
                    else:
                        # unknown -> default to base3 (back-compat)
                        print(f"[WARN] Could not infer K (got {K_expected}); defaulting to base3.")
                        A_use = make_partitioned_base3(17, corr_pkl).to(device)

                # Build model, then load weights
                model = build_model(model_name, cfg, device, A=A_use if model_needs_graph(model_name) else None)
                try:
                    model.load_state_dict(state["model_state"], strict=False)
                    print(f"[LOAD] Loaded {ckpt_path}")
                except Exception as e:
                    print(f"[LOAD] Failed to load weights for {label}: {e}")
                    continue

                # Params & speed
                total_params, trainable_params, size_mb = count_params_and_size_mb(model)
                print(f"Params: total={total_params:,}  trainable={trainable_params:,}  (~{size_mb:.2f} MB)")
                if x_bench is not None:
                    fwd_ms, fwd_thr = benchmark_forward(model, x_bench, device, warmup=5, iters=20)
                    print(f"Forward speed: {fwd_ms:.3f} ms/batch  |  throughput: {fwd_thr:.1f} samples/s")
                else:
                    fwd_ms, fwd_thr = float('nan'), float('nan')
                    print("Forward speed: N/A (empty test loader)")

                # Metrics
                metrics = evaluate_model(model, test_loader, device, cfg["status_classes"])
                print(f"Total loss (test): {metrics['avg_total_loss']:.4f}")
                print(f"RUL MSE / RMSE   : {metrics['rul_mse']:.6f} / {metrics['rul_rmse']:.6f}")
                for j, (acc, mp) in enumerate(zip(metrics["status_micro_acc"], metrics["status_macro_prec"])):
                    print(f"status{j}: micro_acc={acc*100:.2f}%  macro_prec={mp*100:.2f}%  CE={metrics['status_ce'][j]:.4f}")

                results.append({
                    "name": f"{label}@{root.name}",
                    "root": root.name,
                    "params": total_params,
                    "size_mb": size_mb,
                    "fwd_ms": fwd_ms,
                    "fwd_thr": fwd_thr,
                    **metrics
                })

    # -------- Summary --------
    print("\n" + "="*90)
    print("SUMMARY (TEST) — lower loss/RMSE better; higher accuracy/precision/speed better")
    print("="*90)

    if not results:
        print("No models evaluated. (Missing checkpoints?)")
        return

    print("\nTotal loss (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["avg_total_loss"])):
        print(f"{i+1}. {r['name']} {r['avg_total_loss']:.4f}")

    print("\nRUL RMSE (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["rul_rmse"])):
        print(f"{i+1}. {r['name']} {r['rul_rmse']:.6f}")

    H = len(cfg["status_classes"])
    for j in range(H):
        print(f"\nstatus{j} micro accuracy (desc):")
        for i, r in enumerate(sorted(results, key=lambda r: r["status_micro_acc"][j], reverse=True)):
            print(f"{i+1}. {r['name']} {r['status_micro_acc'][j]*100:.2f}%")

    for j in range(H):
        print(f"\nstatus{j} macro precision (desc):")
        for i, r in enumerate(sorted(results, key=lambda r: r["status_macro_prec"][j], reverse=True)):
            print(f"{i+1}. {r['name']} {r['status_macro_prec'][j]*100:.2f}%")

    print("\nParameter count (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["params"])):
        print(f"{i+1}. {r['name']} {r['params']:,} params (~{r['size_mb']:.2f} MB)")

    print("\nModel size MB (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: r["size_mb"])):
        print(f"{i+1}. {r['name']} ~{r['size_mb']:.2f} MB")

    print("\nForward avg ms/batch (asc):")
    for i, r in enumerate(sorted(results, key=lambda r: (float('inf') if math.isnan(r['fwd_ms']) else r['fwd_ms']))):
        print(f"{i+1}. {r['name']} {r['fwd_ms']:.3f} ms")

    print("\nForward throughput (desc):")
    for i, r in enumerate(sorted(results, key=lambda r: (-1.0 if math.isnan(r['fwd_thr']) else r['fwd_thr']), reverse=True)):
        print(f"{i+1}. {r['name']} {r['fwd_thr']:.1f} samples/s")


if __name__ == "__main__":
    main()
