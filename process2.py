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
    return np.concatenate(train_data, axis=0)  # Return multi-sensor time series data

# EWMA smoothing
def ewma_smoothing(data, alpha=0.3):
    smoothed = np.zeros_like(data)
    for i in range(data.shape[0]):
        smoothed[i, 0] = data[i, 0]
        for t in range(1, data.shape[1]):
            smoothed[i, t] = alpha * data[i, t] + (1 - alpha) * smoothed[i, t-1]
    return smoothed

# Sliding window segmentation
def sliding_window(data, window_size, step_size):
    slices = []
    for i in range(0, data.shape[1] - window_size + 1, step_size):
        slice_data = data[:, i:i + window_size]
        if slice_data.ndim == 1:
            slice_data = np.expand_dims(slice_data, axis=0)  # 确保是二维
        slices.append(slice_data)
    return slices

# Statistical feature extraction
def extract_features(slice_data):
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
        if slice_data.ndim == 1:
            slice_data = np.expand_dims(slice_data, axis=0)  # 确保是二维
        features = extract_features(slice_data)
        si = {'slice': slice_data, 'features': features, 'index': k}
        sensor_instances.append(si)
    return sensor_instances

# 3. Sensor instance relationship mining
# A. DTW-KMeans clustering for sensors based on features from dim 0 and dim 2
def dtw_kmeans_clustering(raw_data, max_clusters=10, use_dtw=True):
    n_slices, n_sensors, window_size = raw_data.shape  # 14000, 14, 30
    sensor_features = raw_data.transpose(1, 0, 2)  # 14, 14000, 30
    sensor_features = sensor_features.reshape(n_sensors, -1)
    
    if use_dtw:
        # Compute DTW distance matrix between sensors
        distance_matrix = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                dist = dtw.distance(sensor_features[i], sensor_features[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        # Use distance matrix for KMeans
        input_data = distance_matrix
    else:
        # Use feature matrix directly for KMeans
        input_data = sensor_features
    
    # Determine optimal k using Elbow Method and Silhouette Score
    inertia_values = []
    silhouette_scores = []
    
    # Try k values from 2 to max_clusters
    for k in range(2, max_clusters + 1):
        if k > n_sensors:
            break  # Avoid k exceeding the number of sensors
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(input_data)
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
    final_kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42).fit(input_data)
    
    return final_kmeans.labels_, optimal_k

# B. DYNOTEARS causality discovery
def dnotears(X, Y, lambda_W=0.1, lambda_A=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    n, d = X.shape
    p = Y.shape[1] // d  # Number of lagged time slices
    W = np.zeros((d, d))
    A = np.zeros((p*d, d))
    
    def _loss(W, A):
        X_pred = X @ W + Y @ A
        loss = 0.5 / n * np.linalg.norm(X - X_pred, 'fro')**2
        reg_W = lambda_W * np.sum(np.abs(W))
        reg_A = lambda_A * np.sum(np.abs(A))
        return loss + reg_W + reg_A
    
    def _h(W):
        E = expm(W * W)
        h = np.trace(E) - d
        return h
    
    def _func(w_a):
        W_flat = w_a[:d*d].reshape((d, d))
        A_flat = w_a[d*d:].reshape((p*d, d))
        loss = _loss(W_flat, A_flat)
        h = _h(W_flat)
        return loss + 0.5 * rho * h**2 + alpha * h
    
    rho, alpha, h = 1.0, 0.0, np.inf
    w_a = np.concatenate([W.flatten(), A.flatten()])
    for _ in tqdm(range(max_iter), desc="DYNOTEARS Optimization"):
        while rho < rho_max:
            sol = minimize(_func, w_a, method='L-BFGS-B', jac=False)
            w_a_new = sol.x
            W_new = w_a_new[:d*d].reshape((d, d))
            h_new = _h(W_new)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_a, h = w_a_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    
    W = w_a[:d*d].reshape((d, d))
    A = w_a[d*d:].reshape((p*d, d))
    
    # Thresholding
    W[np.abs(W) < w_threshold] = 0
    A[np.abs(A) < w_threshold] = 0
    
    return W, A

# 4. SFG construction and storage in Neo4j
def build_sfg_neo4j(sensor_instances, cluster_labels, W_est, A_est):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    graph.delete_all()  # Clear the database
    nodes = {}
    for si in sensor_instances:
        k = si['index']
        # Create data slice node
        v_node = Node("DataSlice", index=k, slice=si['slice'].tolist())  # Save slice data
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
                w_corr = 2 / np.pi * np.arctan(dtw.distance(sensor_instances[i]['slice'].flatten(), 
                                                            sensor_instances[j]['slice'].flatten()))
                graph.create(Relationship(nodes[f"v_{i}"], "CORRELATION", nodes[f"v_{j}"], weight=w_corr))

    # Add causality edges from W_est
    for i in range(W_est.shape[0]):
        for j in range(W_est.shape[1]):
            if W_est[i, j] != 0:
                w_caus = 2 / np.pi * np.arctan(W_est[i, j])
                graph.create(Relationship(nodes[f"f_{i}_mean_0"], "CAUSALITY", nodes[f"f_{j}_mean_0"], weight=w_caus))

    # Add causality edges from A_est
    for i in range(A_est.shape[0]):
        for j in range(A_est.shape[1]):
            if A_est[i, j] != 0:
                w_caus = 2 / np.pi * np.arctan(A_est[i, j])
                graph.create(Relationship(nodes[f"f_{i}_mean_0"], "LAGGED_CAUSALITY", nodes[f"f_{j}_mean_0"], weight=w_caus))

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
    raw_data = load_rul_data()
    print(raw_data.shape)  # 检查形状

    # 如果 raw_data 是三维的，将其转换为二维
    if len(raw_data.shape) == 3:
        n_samples, n_timesteps, n_features = raw_data.shape
        raw_data = raw_data.reshape(n_samples * n_timesteps, n_features)
    
    # 构建传感器实例
    sensor_instances = construct_sensor_instances(raw_data)

    # Relationship mining
    cluster_labels = dtw_kmeans_clustering(raw_data, use_dtw=False)
    
    # Prepare data for DYNOTEARS
    p = 3  # 滞后阶数
    n, d = raw_data.shape
    X = raw_data[p:]  # 当前时间片
    Y = np.zeros((n - p, p * d))  # 滞后时间片
    for i in range(p):
        if i > 0:
            Y[:, i*d:(i+1)*d] = raw_data[p - i - 1:-i - 1]
        else:
            Y[:, i*d:(i+1)*d] = raw_data[:n - p]
    
    # 使用DYNOTEARS进行因果结构学习
    W_est, A_est = dnotears(X, Y, lambda_W=0.1, lambda_A=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3)
    
    # 保存所有变量到文件
    variables_to_save = {
        'raw_data': raw_data,
        'sensor_instances': sensor_instances,
        'cluster_labels': cluster_labels,
        'W_est': W_est,
        'A_est': A_est
    }
    with open('variables.pkl', 'wb') as f:
        pickle.dump(variables_to_save, f)
    print("Variables saved to variables.pkl")

    # 加载保存的变量
    with open('variables.pkl', 'rb') as f:
        variables = pickle.load(f)
    
    # 提取变量
    raw_data = variables['raw_data']
    sensor_instances = variables['sensor_instances']
    cluster_labels = variables['cluster_labels']
    W_est = variables['W_est']
    A_est = variables['A_est']

    # SFG construction
    build_sfg_neo4j(sensor_instances, cluster_labels, W_est, A_est)
    
    # Data completion (assuming the 0th instance is missing)
    # estimated_slice = data_completion(graph, missing_si_index=0)
    # print("Estimated missing slice:", estimated_slice)