import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

def cluster_sensors_dtw(sensor_data, n_clusters=10):
    """
    Cluster 17 sensor time-series using DTW-based KMeans.
    
    Parameters:
    -----------
    sensor_data : array-like 
        Sensor time series for one sample window, shape (17, T). 
        Each row is the time series for one sensor. If sensor series have unequal 
        lengths, provide as a list of arrays or a padded 2D array.
    n_clusters : int (default=10)
        Number of clusters for KMeans.
    
    Returns:
    --------
    A : numpy.ndarray of shape (17, 17)
        Binary association matrix where A[i, j] = 1 if sensors i and j are in the same cluster, else 0.
    labels : numpy.ndarray of shape (17,)
        Cluster label (0 to n_clusters-1) for each sensor's time series.
    """
    # Convert input to a time-series dataset format expected by tslearn (3D array).
    # If sensor_data is a list of 1D arrays (varying lengths), pad them; 
    # if it's already a 2D array, just add the feature dimension.
    if isinstance(sensor_data, np.ndarray):
        if sensor_data.ndim == 2:
            # shape (17, T) -> (17, T, 1)
            X = sensor_data[:, :, np.newaxis]
        else:
            X = sensor_data  # Already in desired shape
    else:
        # If input is a list of arrays (one per sensor), pad to time_series_dataset
        X = to_time_series_dataset(sensor_data)
    
    # Define TimeSeriesKMeans with DTW metric
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    # Fit the model and predict cluster for each sensor time-series
    labels = model.fit_predict(X)
    
    # Build the 17x17 association matrix A
    labels = np.asarray(labels)
    A = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                A[i, j] = 1
    
    print(f"[Info] Clustering complete: {n_clusters} clusters found.")
    # Print the labels matrix for debugging
    print(f"[Info] Cluster labels: {labels}")
    print(f"[Info] Association matrix shape: {A.shape}")

    return A, labels

if __name__ == "__main__":
    import pickle
    import torch
    from torch.utils.data import DataLoader
    from data_loader import data_loader # for loading pkl files


    # === CONFIGURATION ===
    data_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train.pkl"
    batch_index = 0    # which batch to pick
    sample_index = 0   # which sample in batch to pick
    window_index = 0   # which 10s window in sample to pick

    # === LOAD PKL ===
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    batch = next(iter(loader))  # (x100, x10, x1), y
    (x100, x10, x1), y = batch

    # === EXTRACT SPECIFIC 10s WINDOW ===
    x100_sample = x100[batch_index][window_index]  # (7, 1000)
    x10_sample  = x10[batch_index][window_index]   # (2, 100)
    x1_sample   = x1[batch_index][window_index]    # (8, 10)

    # Pad all to the same shape (1000) for simplicity
    def pad_to(arr, target_len):
        return np.pad(arr, ((0, 0), (0, target_len - arr.shape[1])), mode='edge')

    s100 = x100_sample.numpy()
    s10  = pad_to(x10_sample.numpy(), 1000)
    s1   = pad_to(x1_sample.numpy(), 1000)
    sensor_window = np.concatenate([s100, s10, s1], axis=0)  # (17, 1000)

    # === CLUSTER SENSORS ===
    from dtw_kmean import cluster_sensors_dtw
    A, labels = cluster_sensors_dtw(sensor_window, n_clusters=10)

    print("Association matrix A:")
    print(A)
