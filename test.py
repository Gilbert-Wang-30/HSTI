# test.py
import torch
import pickle
from models.ll import MultiTaskModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datasets.data_loader import data_loader
import numpy as np
import numpy as np  # make sure it's imported

# Config
status_classes = [3, 4, 3, 4]
input_dim = 1020
raw_input_dim = 43680  # For raw data, if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MultiTaskModel(in_features=input_dim, status_classes=status_classes).to(device)
model.load_state_dict(torch.load('models/best_ll_model.pth', map_location=device, weights_only=True))
model.eval()

raw_model = MultiTaskModel(in_features=raw_input_dim, status_classes=status_classes).to(device)
raw_model.load_state_dict(torch.load('models/best_raw_model.pth', map_location=device, weights_only=True))
raw_model.eval()

# Evaluate one split
def evaluate(pkl_path, split_name):
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    X, rul_y, status_y = [], [], [[] for _ in range(4)]
    for (_, _, _), feat, rul, status in dataset:
        feat = torch.nan_to_num(feat.flatten(start_dim=0), nan=0.0, posinf=1e3, neginf=-1e3)
        X.append(feat.numpy())
        rul_y.append(rul.item())
        for j in range(4):
            status_y[j].append(status[j].item())


    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    status_y = [torch.tensor(np.array(s), dtype=torch.float32).long().to(device) for s in status_y]
    rul_y = torch.tensor(rul_y, dtype=torch.float32).to(device)

    with torch.no_grad():
        rul_pred, status_logits, _ = model(X)
        rul_loss = torch.nn.functional.mse_loss(rul_pred, rul_y).item()

        print(f"\n===== {split_name.upper()} SET =====")
        print(f"RUL MSE Loss: {rul_loss:.4f}")
        for j in range(4):
            y_true = status_y[j].cpu().numpy()
            y_pred = status_logits[j].argmax(dim=1).cpu().numpy()
            print(f"Status {j}: "
                  f"Acc {accuracy_score(y_true, y_pred):.4f}, "
                  f"Prec {precision_score(y_true, y_pred, average='macro'):.4f}, "
                  f"Recall {recall_score(y_true, y_pred, average='macro'):.4f}, "
                  f"F1 {f1_score(y_true, y_pred, average='macro'):.4f}")

# Run evaluations
evaluate("data/processed/train.pkl", "train")
evaluate("data/processed/val.pkl", "val")
evaluate("data/processed/test.pkl", "test")
