# -*- coding: utf-8 -*-
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

# 1. Data loading and preprocessing (based on official example)
def load_rul_data():
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)  # FD001 subset
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
    dm.prepare_data()
    dm.setup()
    # Extract feature data from the training dataloader
    train_data = []
   
    for features, _ in dm.train_dataloader():
        train_data.append(features.numpy())  # Convert to NumPy array
    raw_data = np.concatenate(train_data, axis=0)  # Return multi-sensor time series data
    print("Raw data shape:", raw_data.shape) # (13818, 14, 30)
    sample_len = 10
    raw_data = raw_data[:sample_len] # shape: (10, 14, 30)
    return raw_data


# EWMA smoothing
def ewma_smoothing(data, alpha=0.3):
    smoothed = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            smoothed[i, j, 0] = data[i, j, 0]
            for t in range(1, data.shape[2]):
                smoothed[i, j, t] = alpha * data[i, j, t] + (1 - alpha) * smoothed[i, j, t-1]
    return smoothed

# Sliding window segmentation
def sliding_window(data, window_size, step_size):
    slices = []
    for i in range(0, data.shape[2] - window_size + 1, step_size):
        slice_data = data[:, :, i:i + window_size]
        slices.append(slice_data)
    return slices

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
    sensor_instances = []
    for k, slice_data in enumerate(slices):
        features = extract_features(slice_data)
        si = {'slice': slice_data, 'features': features, 'index': k}
        sensor_instances.append(si)
    return sensor_instances

# 3. Sensor instance relationship mining
# A. DTW-KMeans clustering for sensors based on features from dim 0 and dim 2
def dtw_kmeans_clustering(raw_data, max_clusters=10, use_dtw=True):
    n_instances, n_sensors, window_size = raw_data.shape  # shape: (10, 14, 30)
    sensor_features = raw_data.transpose(1, 0, 2)
    instance_features = sensor_features.reshape(n_sensors, -1)
    
    if use_dtw:
        # Compute DTW distance matrix between sensors
        distance_matrix = np.zeros((n_sensors, n_sensors)) # 14*14
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                dist = dtw.distance(instance_features[i], instance_features[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        # Use distance matrix for KMeans
        input_data = distance_matrix
    else:
        # Use feature matrix directly for KMeans
        input_data = instance_features
    
    # Determine optimal k using Elbow Method and Silhouette Score
    inertia_values = []
    silhouette_scores = []
    
    # Try k values from 2 to max_clusters
    for k in range(2, max_clusters + 1):
        if k > n_sensors:
            break  # Avoid k exceeding the number of sensors
        
        kmeans = KMeans(n_clusters=k, random_state=42).fit(input_data)
        inertia_values.append(kmeans.inertia_)
        
        # Compute silhouette score
        if k > 1 and len(np.unique(kmeans.labels_)) > 1:
            silhouette_scores.append(silhouette_score(input_data, kmeans.labels_))
        else:
            silhouette_scores.append(0)  # Handle edge cases
    
    # Find optimal k using Elbow Method (simple heuristic)
    optimal_k_elbow = None
    if len(inertia_values) > 1:
        # Compute differences in inertia
        diffs = np.diff(inertia_values)
        # Find the "elbow" point (where the difference starts to level off)
        optimal_k_elbow = np.argmax(diffs) + 2  # +2 because k starts from 2
    
    # Find optimal k using Silhouette Score
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2 if silhouette_scores else None
    
    # Choose the optimal k based on both methods
    if optimal_k_elbow is not None and optimal_k_silhouette is not None:
        optimal_k = min(optimal_k_elbow, optimal_k_silhouette)
    elif optimal_k_elbow is not None:
        optimal_k = optimal_k_elbow
    elif optimal_k_silhouette is not None:
        optimal_k = optimal_k_silhouette
    else:
        optimal_k = 2  # Default to 2 if no method works
    
    # Perform final KMeans with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(input_data)
    
    return final_kmeans.labels_, optimal_k

# B. No-Tears causality discovery
def notears_linear(X, lambda1=0.1, loss_type='l2', max_iter=50, h_tol=1e-6, rho_max=1e+10, w_threshold=0.3):
    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        E = expm(W * W)  # Matrix exponential
        h = np.trace(E) - W.shape[0]
        G_h = E.T * W * 2
        return h, G_h

    def _func(w):
        d = int(np.sqrt(len(w)))  # Calculate d from the length of w
        W = w.reshape((d, d))
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        # Compute L1 regularization gradient
        G_l1 = lambda1 * np.sign(W)
        # Combine gradients and flatten
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = (G_smooth + G_l1).flatten()  # Ensure gradient is 1D
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(w).sum()
        return obj, g_obj

    n, d = X.shape
    w_est = np.zeros(d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, None) for _ in range(d * d)]  # Ensure non-negative

    # Use tqdm to monitor progress
    for _ in tqdm(range(max_iter), desc="NoTears Optimization"):
        while rho < rho_max:
            sol = minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(w_new.reshape((d, d)))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = w_est.reshape((d, d))
    W_est[np.abs(W_est) < w_threshold] = 0  # Thresholding
    return W_est

# 4. SFG construction and storage in Neo4j
def build_sfg_neo4j(sensor_instances, cluster_labels, causality_matrix):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    graph.delete_all()  # Clear the database
    nodes = {}
    for si in sensor_instances:
        k = si['index']
        # Create data slice node
        # v_node = Node("DataSlice", index=k, slice=si['slice'].tolist())  # Save slice data

        flattened_slice = si['slice'].flatten().tolist()  # 展平为 (14*30) 的列表
        v_node = Node("DataSlice", index=k, slice=flattened_slice)

        # print(v_node)
        graph.create(v_node)
        nodes[f"v_{k}"] = v_node
        
        # Create feature nodes and establish affiliation relationships
        for feat_name, feat_values in si['features'].items():
            for sensor_idx, feat_val in enumerate(feat_values):
                f_node = Node("Feature", name=f"{feat_name}_{sensor_idx}", value=float(feat_val))
                graph.create(f_node)
                nodes[f"f_{k}_{feat_name}_{sensor_idx}"] = f_node
                graph.create(Relationship(v_node, "AFFILIATION", f_node, weight=1))

    # Add correlation edges
    for i in range(len(sensor_instances)):
        for j in range(i + 1, len(sensor_instances)):
            if cluster_labels[i] == cluster_labels[j]:
                w_corr = float(2 / np.pi * np.arctan(dtw.distance(sensor_instances[i]['slice'].flatten(), 
                                                            sensor_instances[j]['slice'].flatten())))
                graph.create(Relationship(nodes[f"v_{i}"], "CORRELATION", nodes[f"v_{j}"], weight=w_corr))

    # Add causality edges
    feature_keys = [f"f_{k}_{feat_name}_{sensor_idx}" 
                    for k in range(len(sensor_instances)) 
                    for feat_name in sensor_instances[0]['features'] 
                    for sensor_idx in range(len(sensor_instances[0]['features']['mean']))]
    for i in range(causality_matrix.shape[0]):
        for j in range(causality_matrix.shape[1]):
            if causality_matrix[i, j] > 0.0:
                w_caus = float(2 / np.pi * np.arctan(causality_matrix[i, j]))
                graph.create(Relationship(nodes[feature_keys[i]], "CAUSALITY", nodes[feature_keys[j]], weight=w_caus))

# 5. Data completion
def data_completion(graph, missing_si_index):
    missing_node = graph.nodes.match("DataSlice", index=missing_si_index).first()
    correlated_nodes = [rel.end_node for rel in graph.match((missing_node,), r_type="CORRELATION")]
    correlated_slices = [np.array(node["slice"]) for node in correlated_nodes if "slice" in node]
    if correlated_slices:
        estimated_slice = np.mean(correlated_slices, axis=0)
        return estimated_slice
    return None

# Main program
if __name__ == "__main__":
    # Data loading and preprocessing
    raw_data = load_rul_data() # shape: (10, 14, 30)
    smoothed_data = ewma_smoothing(raw_data)
    slices = sliding_window(smoothed_data, window_size=10, step_size=5)
    slices = raw_data
    sensor_instances = construct_sensor_instances(slices) # 10 * {'slice': slice_data, 'features': features, 'index': k}

    # Relationship mining
    # Extract features for clustering based on sensor_instances
    cluster_labels, optimal_k = dtw_kmeans_clustering(raw_data,use_dtw=False)
    features_list = [np.concatenate([v.flatten() for v in si['features'].values()]) for si in sensor_instances]
    features_matrix = np.array(features_list)
    print("features_matrix: ", features_matrix.shape)
    causality_matrix = notears_linear(features_matrix, lambda1=0.1, loss_type='l2')
    print("causality_matrix: ", causality_matrix.shape)
    # print("sum of causality_matrix: ", sum(causality_matrix))

    # Save variables
    variables_to_save = {
        'raw_data': raw_data,
        'sensor_instances': sensor_instances,
        'cluster_labels': cluster_labels,
        'features_matrix': features_matrix,
        'causality_matrix': causality_matrix
    }
    with open('variables.pkl', 'wb') as f:
        pickle.dump(variables_to_save, f)
    print("Variables saved to variables.pkl")

    # Load variables
    with open('variables.pkl', 'rb') as f:
        variables = pickle.load(f)
        raw_data = variables['raw_data']
        sensor_instances = variables['sensor_instances']
        cluster_labels = variables['cluster_labels']
        features_matrix = variables['features_matrix']
        causality_matrix = variables['causality_matrix']

    # SFG construction
    build_sfg_neo4j(sensor_instances, cluster_labels, causality_matrix)