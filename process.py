import numpy as np
import torch
from py2neo import Graph, Node, Relationship
from scipy.stats import skew, kurtosis
from dtaidistance import dtw
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.linalg import expm
import rul_datasets  
import pickle
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset


# 1. Data loading
def load_rul_data():
    reader = rul_datasets.CmapssReader(fd=1, window_size=300)
    dm = rul_datasets.RulDataModule(reader, batch_size=16)
    dm.prepare_data()
    dm.setup()

    # Get a batch from train dataloader
    features, targets = next(iter(dm.train_dataloader()))

    # Fix dimension
    if features.shape[1] == 14:  # If shape is (B, 14, 300)
        features = features.permute(0, 2, 1)  # → (B, 300, 14)

    print(f"Input shape: {features.shape}")   # (16, 300, 14)
    print(f"Target shape: {targets.shape}")   # (16,)
    # first example shape
    print("First sequence shape:", features[0].shape)  # (300, 14)
    print("First RUL label:", targets[0])  # scalar value

    print("Second sequence:\n", features[1])
    print("RUL label:", targets[1])
    # Print last example
    print("Last sequence:\n", features[-1])
    print("RUL label:", targets[-1])


if __name__ == "__main__":
    # data loading
    raw_data = load_rul_data()
    