# ts1_train.py

import os
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.missing_data_loader import missing_data_loader  
from models.LSTM import LSTMModel  

# ----------------- Config -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--sensor", type=str, default="TS1", help="Sensor to predict (must be 1Hz)")
args = parser.parse_args()
SENSOR = args.sensor.upper()
BATCH_SIZE = 32
EPOCHS = 150
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Data -----------------
def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

train_data = load_dataset(f"data/processed/{SENSOR.lower()}_train.pkl")
dev_data = load_dataset(f"data/processed/{SENSOR.lower()}_dev.pkl")

# ----------------- Collate Function -----------------
def collate_ts1(batch):
    X_list, y_list = [], []
    for (x100, x10, x1), ts1_target in batch:
        sensor_groups = {
            "100hz": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"],
            "10hz": ["FS1", "FS2"],
            "1hz": ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
        }

        for freq, sensors in sensor_groups.items():
            if SENSOR in sensors:
                sensor_index = sensors.index(SENSOR)
                if freq == "1hz":
                    full_seq = x1[sensor_index].reshape(-1)  # (60,)
                elif freq == "10hz":
                    full_seq = x10[sensor_index].reshape(-1)  # (600,)
                elif freq == "100hz":
                    full_seq = x100[sensor_index].reshape(-1)  # (6000,)
                break
        else:
            raise ValueError(f"Unknown sensor: {SENSOR}")

        input_seq = full_seq[:-1].unsqueeze(-1)  # (59, 1) — exclude last
        X_list.append(input_seq)
        y_list.append(ts1_target.unsqueeze(0))  # scalar target
    return torch.stack(X_list), torch.stack(y_list)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_ts1)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_ts1)

# ----------------- Model, Loss, Optim -----------------
model = LSTMModel(input_size=1, hidden_size=64, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------- TensorBoard Setup -----------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/{SENSOR.lower()}/{run_id}"
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logging to: {log_dir}")

# ----------------- Training Loop -----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    avg_train_loss = train_loss / len(train_loader.dataset)

    # -------- Validation(for validation loss graph) --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dev_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    avg_val_loss = val_loss / len(dev_loader.dataset)

    # Logging
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    print(f"[Epoch {epoch:03d}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

# ----------------- Save Model -----------------
os.makedirs("models", exist_ok=True)
model_path = f"models/{SENSOR.lower()}_lstm.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
# ----------------- Close TensorBoard Writer -----------------
writer.flush()
writer.close()

