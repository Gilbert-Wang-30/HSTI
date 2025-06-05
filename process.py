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

# 1. Data loading
def load_rul_data():
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1) # FD001 subset
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size = 32)
    dm.prepare_data()
    dm.setup()
    
    train_data = []
    for features, _ in dm.train_dataloader():
        train_data.append(features.numpy())
    raw_data = np.concatenate(train_data, axis=0)

    sample_len = 10
    raw_data = raw_data[:sample_len]
    return raw_data

# Statistical feature extraction
def extract_features(slice_data): # 10 features
    features = {
        'mean': np.mean(slice_data, axis=1),
        'var': np.var(slice_data, axis=1),
        'std': np.std(slice_data, axis=1),
        'skew': skew(slice_data, axis=1),
        'kurt': kurtosis(slice_data, axis=1),
        'max': np.max(slice_data, axis=1),
        'min': np.min(slice_data, axis=1),
        'pulse_count': np.sum(np.abs(np.diff(slice_data, axis=1)) > 0, axis=1),
        'peak': np.max(np.abs(slice_data), axis=1),
        'amplitude': np.max(slice_data, axis=1) - np.min(slice_data, axis=1)
    }
    return features

# 2. Sensor instance construction
def construct_sensor_instances(slices):
    slices = slices.transpose(1, 0, 2)
    sensor_instances = []
    for k, slice_data in enumerate(slices):
        print(k, slice_data.shape)
        features = extract_features(slice_data)
        si = {'slice': slice_data, 'features': features, 'index': k}
        sensor_instances.append(si)
    return sensor_instances

if __name__ == "__main__":
    # data loading
    raw_data = load_rul_data()
    sensor_instances = construct_sensor_instances(raw_data)