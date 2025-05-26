import torch
import numpy as np
import rul_datasets
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import networkx as nx
from fastdtw import fastdtw
from scipy.optimize import minimize  
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import json


# 定义 EWMA 平滑函数
def ewma(data, alpha):
    ewma_data = np.zeros_like(data)
    ewma_data[0] = data[0]
    for t in range(1, len(data)):
        ewma_data[t] = alpha * data[t] + (1 - alpha) * ewma_data[t-1]
    return ewma_data

# 定义统计特征提取函数
def extract_features(slice_data):
    features = {
        'mean': np.mean(slice_data, axis=0),
        'std': np.std(slice_data, axis=0),
        'var': np.var(slice_data, axis=0),
        'max': np.max(slice_data, axis=0),
        'min': np.min(slice_data, axis=0),
        'pulse_count': np.sum(np.abs(np.diff(slice_data, axis=0)) > 0, axis=0),  # 脉冲数
        'peak': np.max(np.abs(slice_data), axis=0),  # 峰值
        'amplitude': np.max(slice_data, axis=0) - np.min(slice_data, axis=0)  # 幅值
    }
    return features

# 定义 DTW 距离计算函数

def dtw_distance(series1, series2):
    series1 = series1.ravel()  # 使用 ravel 确保一维
    series2 = series2.ravel()
    distance, _ = fastdtw(series1, series2, dist=euclidean)
    return distance

# 定义基于 DTW 和 K-Means 的聚类函数
def dtw_kmeans_clustering(data, n_clusters=10):
    """基于 DTW 和 K-Means 的聚类"""
    n_samples, n_features, n_timesteps = data.shape
    data_reshaped = data.reshape(n_samples, -1)  # 展平每个样本

    # 计算 DTW 距离矩阵
    dtw_matrix = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples)):
        for j in range(n_samples):
            dtw_matrix[i, j] = dtw_distance(data_reshaped[i], data_reshaped[j])

    # 使用 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(dtw_matrix)
    labels = kmeans.labels_

    return labels
   

def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
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
        E = expm(W * W)  # 替换 np.linalg.expm 为 expm
        h = np.trace(E) - W.shape[0]
        G_h = E.T * W * 2
        return h, G_h

    def _func(w):
        W = w.reshape((d, d))
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.sum(np.abs(w))
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = G_smooth + lambda1
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(d * d), 1.0, 0.0, np.inf
    bnds = [(0, None) for _ in range(d * d)]
    for _ in tqdm(range(max_iter)):
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
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def build_sfg(data, features, labels, W, sensing_instances):
    """构建双层传感特征图"""
    G = nx.DiGraph()

    # 添加时间序列数据块节点
    for i in range(data.shape[0]):
        G.add_node(f"v_{i}", type="data_block", name=f"Data Block {i}")

    # 添加统计特征节点
    for i in range(features.shape[1]):
        G.add_node(f"f_{i}", type="feature", name=f"Feature {i}")

    # 添加关联边
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if labels[i] == labels[j] and i != j:
                G.add_edge(f"v_{i}", f"v_{j}", type="correlation", name="Correlation", weight=1.0)

    # 添加因果边
    for i in range(features.shape[1]):
        for j in range(features.shape[1]):
            if W[i, j] != 0:
                G.add_edge(f"f_{i}", f"f_{j}", type="causality", name="Causality", weight=W[i, j])

      # 添加隶属边（单独特征与数据块之间建立隶属边）
    for i in range(data.shape[0]):  # 遍历每个数据块
        for sensing_instance in sensing_instances:  # 遍历每个传感实例
            if np.array_equal(sensing_instance['data'], data[i]):  # 如果数据块匹配
                for feature_name in feature_names:  # 遍历每个特征
                    for j in range(features.shape[1]):  # 遍历每个特征维度
                        if np.isclose(sensing_instance['features'][feature_name][j], features[i, j]):
                            G.add_edge(f"v_{i}", f"f_{feature_name}_{j}", type="affiliation", name="Affiliation", weight=1.0)

    return G

def draw_sfg(G):
    """
    绘制信号流图 (SFG)
    :param G: networkx.DiGraph 对象
    """
    # 设置布局
    pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局，seed 确保布局一致

    # 绘制节点
    node_colors = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "data_block":
            node_colors.append("skyblue")
        elif data.get("type") == "feature":
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightgrey")

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)

    # 绘制节点标签
    node_labels = {node: data.get("type", "") for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # 绘制边
    edge_colors = []
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if data.get("type") == "correlation":
            edge_colors.append("orange")
        elif data.get("type") == "causality":
            edge_colors.append("red")
        elif data.get("type") == "affiliation":
            edge_colors.append("gray")
        edge_labels[(u, v)] = data.get("type", "")

    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=10, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # 添加标题并显示
    plt.title("Signal Flow Graph (SFG)", fontsize=14)
    plt.axis("off")  # 关闭坐标轴
    plt.show()
    plt.savefig("SFG.png")

def save_graph(G, filename="graph.json"):
    """保存图 G 到文件"""
    data = nx.node_link_data(G, edges="edges")  # 显式指定 edges 参数
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_graph(filename="graph.json"):
    """从文件加载图 G"""
    with open(filename, 'r') as f:
        data = json.load(f)
    G = nx.node_link_graph(data, edges="edges")  # 显式指定 edges 参数
    return G

# 主函数
def main():
        
    # 加载数据集
    cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
    dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
    dm.prepare_data()
    dm.setup()

    # 获取训练数据
    train_data, train_labels = next(iter(dm.train_dataloader()))
    train_data = train_data.numpy()  # 转换为 NumPy 数组

    # 数据平滑
    alpha = 0.5  # 权重因子
    smoothed_data = np.zeros_like(train_data)
    for i in range(train_data.shape[1]):  # 遍历每个传感器
        smoothed_data[:, i, :] = ewma(train_data[:, i, :], alpha)

    # 时序数据切割
    delta_t = 10  # 时间区间
    num_slices = smoothed_data.shape[2] // delta_t
    slices = []
    for k in range(num_slices):
        start = k * delta_t
        end = (k + 1) * delta_t
        slices.append(smoothed_data[:, :, start:end])

    # 统计特征提取
    features_list = []
    for slice_data in slices:
        features = extract_features(slice_data)
        features_list.append(features)

    # 构建特征矩阵
    feature_names = ['mean', 'std', 'var', 'max', 'min', 'pulse_count', 'peak', 'amplitude']
    n_samples = len(features_list)  # 修改为 features_list 的长度
    n_features = len(feature_names)
    n_dimensions = smoothed_data.shape[1]
    n_time_steps = delta_t

    features_matrix = np.zeros((n_samples, n_features * n_dimensions * n_time_steps))

    for i in range(n_samples):  # 修改循环范围
        for j, feature_name in enumerate(feature_names):
            start_idx = j * n_dimensions * n_time_steps
            end_idx = (j + 1) * n_dimensions * n_time_steps
            features_matrix[i, start_idx:end_idx] = features_list[i][feature_name].flatten()

    # 打印特征矩阵形状
    print("features_matrix shape:", features_matrix.shape)

    # DWT-K-Means 聚类
    print("Correlation Discovery:")
    labels = dtw_kmeans_clustering(smoothed_data, n_clusters=3)

    # No-Tears 因果关系发现
    print("Causality Discovery:")
    W = notears_linear(features_matrix, lambda1=0.1, loss_type='l2')

    # 构建 SFG
    G = build_sfg(smoothed_data, features_matrix, labels, W, sensing_instances)

    # 打印结果
    print("SFG constructed with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges")
    for node in G.nodes(data=True):
        print(node)
    for edge in G.edges(data=True):
        print(edge)

    # 绘制 SFG
    draw_sfg(G)

    # 保存图
    save_graph(G, "graph.json")


    # main()

if __name__ == '__main__':
    main()