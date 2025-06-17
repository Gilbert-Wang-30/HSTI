#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

def load_matrix(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No file found at {pkl_path}")
    with open(pkl_path, "rb") as f:
        matrix = pickle.load(f)
    return matrix

def plot_heatmap(matrix, output_path, title):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel("Feature j")
    plt.ylabel("Feature i")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and Save PCMCI Adjacency Matrix Heatmap")
    parser.add_argument("--start", type=int, required=True, help="Start cycle index")
    parser.add_argument("--end", type=int, required=True, help="End cycle index")
    parser.add_argument("--lag", type=int, required=True, help="Lag to visualize (e.g., 0 or 1 or 2...)")
    parser.add_argument("--dir", type=str, default="pcmci", help="Directory containing PCMCI .pkl files")
    parser.add_argument("--outdir", type=str, default="heatmaps", help="Directory to save heatmap PNG")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.lag == 0:
        filename = f"pcmci_instant_strength_matrix_cycles_{args.start}_to_{args.end}_lag0.pkl"
    else:
        filename = f"pcmci_strength_matrix_cycles_{args.start}_to_{args.end}_lag{args.lag}.pkl"

    file_path = os.path.join(args.dir, filename)
    output_img = os.path.join(args.outdir, f"heatmap_cycles_{args.start}_to_{args.end}_lag{args.lag}.png")

    try:
        W = load_matrix(file_path)
        print(f"[INFO] Loaded matrix of shape {W.shape} from: {file_path}")
        plot_heatmap(W, output_img,
                     title=f"PCMCI Strength Graph (Lag {args.lag}) - Cycles {args.start} to {args.end}")
        print(f"[INFO] Heatmap saved to: {output_img}")
    except Exception as e:
        print(f"[ERROR] {e}")
