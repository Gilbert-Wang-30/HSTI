# ts1_train.py

import os
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.missing_data_loader import missing_data_loader  
from models.LSTM import LSTMModel  

# ----------------- Config -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--sensor", type=str, default="TS1", help="Sensor to predict (must be 1Hz/10Hz/100Hz)")
args = parser.parse_args()
SENSOR = args.sensor.upper()
BATCH_SIZE = 32
EPOCHS = 250
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Data -----------------
def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

train_data = load_dataset(f"data/processed/{SENSOR.lower()}_train.pkl")
dev_data = load_dataset(f"data/processed/{SENSOR.lower()}_dev.pkl")

# ----------------- Normalization Stats -----------------
def get_input_stats(train_data):
    sensor_groups = {
        "100hz": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"],
        "10hz": ["FS1", "FS2"],
        "1hz": ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
    }
    inputs = []
    for (x100, x10, x1), ts1_target in train_data:
        for freq, sensors in sensor_groups.items():
            if SENSOR in sensors:
                sensor_index = sensors.index(SENSOR)
                if freq == "1hz":
                    full_seq = x1[sensor_index].reshape(-1)
                elif freq == "10hz":
                    full_seq = x10[sensor_index].reshape(-1)
                elif freq == "100hz":
                    full_seq = x100[sensor_index].reshape(-1)
                break
        input_seq = full_seq[:-1]
        inputs.append(input_seq)
    all_inputs = torch.stack(inputs).flatten()
    mean = all_inputs.mean().item()
    std = all_inputs.std().item()
    return mean, std

norm_mean, norm_std = get_input_stats(train_data)
print(f"[Normalization] mean={norm_mean:.4f}, std={norm_std:.4f}")

# ----------------- Collate Function -----------------
def collate_ts1(batch):
    X_list, y_list = [], []
    sensor_groups = {
        "100hz": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"],
        "10hz": ["FS1", "FS2"],
        "1hz": ["TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]
    }
    for (x100, x10, x1), ts1_target in batch:
        for freq, sensors in sensor_groups.items():
            if SENSOR in sensors:
                sensor_index = sensors.index(SENSOR)
                if freq == "1hz":
                    full_seq = x1[sensor_index].reshape(-1)
                elif freq == "10hz":
                    full_seq = x10[sensor_index].reshape(-1)
                elif freq == "100hz":
                    full_seq = x100[sensor_index].reshape(-1)
                break
        input_seq = full_seq[:-1].unsqueeze(-1)
        input_seq = (input_seq - norm_mean) / norm_std
        y_list.append((ts1_target.unsqueeze(0) - norm_mean) / norm_std)
        X_list.append(input_seq)
    return torch.stack(X_list), torch.stack(y_list)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_ts1)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_ts1)

# ----------------- Model, Loss, Optim -----------------
model = LSTMModel(input_size=1, hidden_size=64, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# ----------------- TensorBoard Setup -----------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/{SENSOR.lower()}/{run_id}"
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logging to: {log_dir}")

# ----------------- Model Checkpointing -----------------
os.makedirs("models", exist_ok=True)
best_val_loss = float("inf")
best_epoch = 0
model_path = f"models/{SENSOR.lower()}_lstm.pth"

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

    # -------- Validation (for validation loss graph) --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dev_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    avg_val_loss = val_loss / len(dev_loader.dataset)

    # Learning rate decay step
    scheduler.step()

    # Logging
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
    print(f"[Epoch {epoch:03d}] Train: {avg_train_loss:.7f} | Val: {avg_val_loss:.7f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model if improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), model_path)
        print(f"  [Checkpoint] Saved improved model at epoch {epoch} with val loss {avg_val_loss:.7f}")

# ----------------- Close TensorBoard Writer -----------------
writer.flush()
writer.close()
print(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}")

# Save normalization stats for future use
np.savez(f"models/{SENSOR.lower()}_lstm_norm_stats.npz", mean=norm_mean, std=norm_std)
print(f"Saved normalization stats to models/{SENSOR.lower()}_lstm_norm_stats.npz")
