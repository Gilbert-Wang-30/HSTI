import numpy as np
from tslearn.metrics import dtw_path, dtw
from tslearn.barycenters import dtw_barycenter_averaging

def dtw_kmeans_cluster(sensor_series, n_clusters=2, max_iter=10):
    """
    Cluster a list of sensor time series using K-Means with DTW distance.
    Returns cluster_labels, cluster_centers, adjacency_matrix.
    
    Parameters:
        sensor_series (list of arrays): List of time-series arrays for each sensor.
        n_clusters (int): Desired number of clusters (K).
        max_iter (int): Max iterations for K-Means.
    
    Returns:
        labels (list of int): Cluster assignment for each sensor (length = len(sensor_series)).
        centers (list of arrays): DTW barycenter time series for each cluster.
        A (numpy.ndarray): Adjacency matrix (binary, shape = [N,N] for N sensors).
    """
    N = len(sensor_series)
    # 1. Compute pairwise DTW distance matrix (optional: for analysis or initialization)
    dtw_dist_matrix = np.zeros((N, N))
    print(f"[DEBUG] Computing DTW distance matrix for {N} series...")

    for i in range(N):
        for j in range(i+1, N):
            print(f"    [DTW] Computing distance between series {i} and {j}")
            # Compute DTW distance between series i and j
            dist_ij = dtw(sensor_series[i], sensor_series[j])
            dtw_dist_matrix[i, j] = dist_ij
            dtw_dist_matrix[j, i] = dist_ij
    
    # 2. Initialize cluster centers (pick n_clusters random series as initial centroids)
    rng = np.random.RandomState(0)
    initial_idxs = rng.choice(N, size=n_clusters, replace=False)
    centers = [np.array(sensor_series[idx], copy=True) for idx in initial_idxs]
    
    labels = [None] * N  # cluster labels for sensors
    
    for iteration in range(max_iter):
        # 3. Assignment Step: assign each series to nearest centroid by DTW distance
        labels_changed = False
        new_labels = []
        for i in range(N):
            # Compute DTW distance from sensor_series[i] to each centroid
            distances = [dtw(sensor_series[i], centers[k]) for k in range(n_clusters)]
            closest_cluster = int(np.argmin(distances))
            new_labels.append(closest_cluster)
            if labels[i] != closest_cluster:
                labels_changed = True
        labels = new_labels
        
        # If no label changed, clustering has converged
        if not labels_changed:
            break
        
        # 4. Update Step: recompute each cluster's centroid using DTW barycenter (DBA)
        new_centers = []
        for k in range(n_clusters):
            # Collect all series in cluster k
            cluster_series = [sensor_series[i] for i in range(N) if labels[i] == k]
            if len(cluster_series) == 0:
                # If a cluster lost all points, re-initialize its centroid randomly
                new_centers.append(np.array(sensor_series[rng.choice(N)], copy=True))
            else:
                # Compute DTW barycenter averaging for cluster k
                # This gives the "average" time series (centroid) aligned under DTW
                centroid_k = dtw_barycenter_averaging(cluster_series, max_iter=30)
                new_centers.append(centroid_k[:, 0] if centroid_k.ndim > 1 else centroid_k)
        centers = new_centers
    
    # 5. Construct adjacency matrix A (1 if same cluster, else 0)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                A[i, j] = 1
    
    return labels, centers, A


def compute_co_occurrence_matrix():
    # Load the dataset from the pickle file
    try:
        with open('/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train.pkl', 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize a 17x17 co-occurrence matrix with zeros
    co_occ_matrix = np.zeros((17, 17), dtype=int)

    # Process the first 20 samples (or all samples if fewer than 20)
    for idx, sample in enumerate(data):
        print(f"[INFO] Processing sample {idx}...")

        if idx >= 20:
            break
        (x100, x10, x1), label = sample  # unpack the sample tuple
        try:
            # Flatten each sensor's 6 time windows into one 1D time series
            sensor_series = []
            # Flatten 100Hz sensors (7 sensors, each 6x1000 array -> length 6000)
            for s in range(x100.shape[0]):  # x100 shape is (7, 6, 1000)
                series = x100[s].reshape(-1)   # flatten 6*1000
                sensor_series.append(series)
            # Flatten 10Hz sensors (2 sensors, each 6x100 array -> length 600)
            for s in range(x10.shape[0]):   # x10 shape is (2, 6, 100)
                series = x10[s].reshape(-1)   # flatten 6*100
                sensor_series.append(series)
            # Flatten 1Hz sensors (8 sensors, each 6x10 array -> length 60)
            for s in range(x1.shape[0]):    # x1 shape is (8, 6, 10)
                series = x1[s].reshape(-1)    # flatten 6*10
                sensor_series.append(series)

            # Cluster the 17 sensor series using DTW-based k-means
            labels = dtw_kmeans_cluster(sensor_series, n_clusters=8, max_iter=10)
        except Exception as e:
            # Handle any error (e.g., clustering failure) gracefully
            print(f"Sample {idx}: error during clustering - {e}. Skipping this sample.")
            continue

        # Update co-occurrence counts for sensors in the same cluster
        # Ensure we count each pair once per sample
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] == labels[j]:
                    co_occ_matrix[i, j] += 1

    # Print the co-occurrence matrix
    print("Co-occurrence matrix (counts of co-clustering in 20 samples):")
    print(co_occ_matrix)

    # Convert to binary matrix with threshold 80% (>=16 out of 20)
    binary_matrix = (co_occ_matrix >= 16).astype(int)
    print("\nBinary co-occurrence matrix (1 if co-occurred in >=16 samples, else 0):")
    print(binary_matrix)
# Example usage (assuming sensor_series is a list of 17 arrays, one per sensor):
# labels, centers, A = dtw_kmeans_cluster(sensor_series, n_clusters=3, max_iter=10)
if __name__ == "__main__":
    import pickle
    from data_loader import data_loader
    # Path to your processed training set (pkl)
    pkl_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train.pkl"
    
    # Load 1 batch of raw data (x100, x10, x1) from pkl
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    for i in range(100):
        (x100, x10, x1), y = dataset[i]
        # print(f"data set: {i}, target: {y.item()}")


    compute_co_occurrence_matrix()
    # Choose the first sample (index 0)
    s = 99
    print(s)
    (x100, x10, x1), y = dataset[s]  # shapes: (7, 6, 1000), (2, 6, 100), (8, 6, 10)

    # Flatten and concatenate all sensor channels across the 6 time windows
    sensor_series = []
    for i in range(7):  # 100Hz
        sensor_series.append(x100[i].reshape(-1).numpy())
    for i in range(2):  # 10Hz
        sensor_series.append(x10[i].reshape(-1).numpy())
    for i in range(8):  # 1Hz
        sensor_series.append(x1[i].reshape(-1).numpy())

    # Cluster the 17 sensors using DTW-KMeans
    labels, centers, A = dtw_kmeans_cluster(sensor_series, n_clusters=8, max_iter=10)

    # Print basic info
    print("\nCluster labels for 17 sensors:")
    print(labels)
    print("\nAdjacency matrix A (1 = same cluster):")
    print(A.astype(int))
    print("\nCluster center lengths:")
    for i, center in enumerate(centers):
        print(f"  Cluster {i}: length {len(center)}")
    print("Target RUL value:", y.item())
