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

    for i in range(N):
        for j in range(i+1, N):
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
        print(f"[DEBUG] Iteration {iteration + 1}/{max_iter}...")
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
    for i in range(20):
        print(f"[INFO] Processing sample {i}...")

        (x100, x10, x1), label = data[i]  # unpack the sample tuple
        try:
            # Flatten each sensor's 6 time windows into one 1D time series
            sensor_series = []
            # Flatten 100Hz sensors (7 sensors, each 6x1000 array -> length 6000)
            for s in range(7):  # x100 shape is (7, 6, 1000)
                series = x100[s].reshape(-1).numpy()   # flatten 6*1000
                sensor_series.append(series)
            # Flatten 10Hz sensors (2 sensors, each 6x100 array -> length 600)
            for s in range(2):   # x10 shape is (2, 6, 100)
                series = x10[s].reshape(-1).numpy()   # flatten 6*100
                sensor_series.append(series)
            # Flatten 1Hz sensors (8 sensors, each 6x10 array -> length 60)
            for s in range(8):    # x1 shape is (8, 6, 10)
                series = x1[s].reshape(-1).numpy()    # flatten 6*10
                sensor_series.append(series)

            # Cluster the 17 sensor series using DTW-based k-means
            labels, centers, A = dtw_kmeans_cluster(sensor_series, n_clusters=8, max_iter=10)
            print(f"\nCluster labels for {i}:")
            print(labels)
            print(f"\nAdjacency matrix A {i}: ")
            print(A.astype(int))

        except Exception as e:
            # Handle any error (e.g., clustering failure) gracefully
            print(f"Sample {i}: error during clustering - {e}. Skipping this sample.")
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

    return binary_matrix, co_occ_matrix

# Example usage (assuming sensor_series is a list of 17 arrays, one per sensor):
# labels, centers, A = dtw_kmeans_cluster(sensor_series, n_clusters=3, max_iter=10)
if __name__ == "__main__":
    import os
    import pickle
    from data_loader import data_loader

    result = compute_co_occurrence_matrix()
    if result is not None:
        binary_matrix, full_matrix = result

    os.makedirs("co-kmean-cluster", exist_ok=True)
    with open("/home/wangyuxiao/project/gilbert_copy/HSTI/co-kmean-cluster/binary_co_matrix.pkl", "wb") as f:
        pickle.dump(binary_matrix, f)
    with open("/home/wangyuxiao/project/gilbert_copy/HSTI/co-kmean-cluster/full_co_matrix.pkl", "wb") as f:
        pickle.dump(full_matrix, f)

    print("[INFO] Saved binary and full co-occurrence matrices to 'co-kmean-cluster/'")