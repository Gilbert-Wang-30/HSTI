# heatmap.py

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

def plot_heatmap(matrix, output_path, title="NOTears Adjacency Matrix Heatmap"):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel("Feature j")
    plt.ylabel("Feature i")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and Save NOTears Adjacency Matrix Heatmap")
    parser.add_argument("--index", type=int, required=True, help="Index of the matrix to visualize")
    parser.add_argument("--dir", type=str, default="noTears", help="Directory containing .pkl file")
    parser.add_argument("--outdir", type=str, default="heatmaps", help="Directory to save heatmap PNG")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    file_path = os.path.join(args.dir, f"notears_result_index_{args.index}.pkl")
    output_img = os.path.join(args.outdir, f"heatmap_index_{args.index}.png")

    try:
        W = load_matrix(file_path)
        print(f"[INFO] Loaded matrix of shape {W.shape} from: {file_path}")
        plot_heatmap(W, output_img, title=f"NOTears DAG - Index {args.index}")
        print(f"[INFO] Heatmap saved to: {output_img}")
    except Exception as e:
        print(f"[ERROR] {e}")
