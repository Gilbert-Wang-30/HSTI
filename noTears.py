import os
import pickle
import numpy as np

import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

from data_loader import data_loader # ensure dataset class is defined for unpickling
from high_level_feature_extraction import extract_high_level_features, load_train_data

def notears_linear(X, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for i in range(max_iter):
        print(f"[NOTears] Iteration {i+1}/{max_iter} | h={h:.2e} | rho={rho:.1e}")

        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def run_notears(index: int):
    train_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/processed_data/train.pkl"
    train_data = load_train_data(train_path, batch_size=1, shuffle=False)
    feature_batches, label_batches, raw_x100_batches, raw_x10_batches, raw_x1_batches = extract_high_level_features(train_data)
    # Squeeze batches axis
    X = np.stack([f.squeeze(0).numpy() for f in feature_batches], axis=0)  # shape: (B, C, W, F) -> (C, W, F)
    y = np.stack([l.squeeze(0).numpy() for l in label_batches], axis=0)  # shape: (B, C, W, F) -> (C, W, F)

    print(f"Index: {index} selected for NOTears analysis.")
    print(f"Target RUL value: {y[index].item()}")
    # get the specific index
    if 0 <= index < len(X):
        X_selected = X[index]
        X_selected = X_selected.transpose(1, 0, 2).reshape(6, 170)  # → (6, 170)

    else:
        print(f"Index {index} is out of bounds for the input data.")
        return None
    
    print(f"Selected input shape: {X_selected.shape}")
    # Run NOTears on the selected data
    W_est = notears_linear(X_selected, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3)
    return W_est


if __name__ == "__main__":
    # Example usage
    index = 6  # Change this to the desired index
    W_est = run_notears(index)
    print(f"Estimated DAG for index {index}:\n{W_est}")
    

    # Store the result under noTears folder
    output_dir = "/home/wangyuxiao/project/gilbert_copy/HSTI/noTears"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"notears_result_index_{index}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(W_est, f)
    print(f"Results saved to {output_path}")
    print("[Info] NOTears analysis complete.")