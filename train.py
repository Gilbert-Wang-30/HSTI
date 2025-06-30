# train.py

# ─── Imports ─────────────────────────────────────────────────────────────
import yaml
import pickle
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from models.stgcn import STGCN
from models.ll import LinearLayer  # Assuming this is a simple linear layer model for testing
from datasets.data_loader import data_loader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ─── Configuration & Hyperparameters ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config' / 'train.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

batch_size = config.get('batch_size', 16)
learning_rate = config.get('learning_rate', 1e-3)
epochs = config.get('epochs', 10)
hidden_channels = config.get('hidden_channels', 64)
num_nodes = config.get('num_nodes', 17)
time_steps = config.get('time_steps', 6)
optim_name = config.get('optimizer', 'AdamW')         # e.g., "Adam" or "SGD"
loss_name = config.get('loss', 'MSELoss')            # e.g., "MSELoss" or "CrossEntropyLoss"


# ─── TensorBoard Logging Setup ───────────────────────────────────────────
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = BASE_DIR / 'runs' / 'experiment_1' / run_id
writer = SummaryWriter(log_dir=str(log_dir))
print("TensorBoard log path:", writer.log_dir)


# ─── Load Dataset ────────────────────────────────────────────────────────
with open('data/processed/train.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
with open('data/processed/val.pkl', 'rb') as f:
    dev_dataset = pickle.load(f)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# ─── Load Adjacency Matrix ───────────────────────────────────────────────
file_path = BASE_DIR / 'data' / 'causality' / 'pcmci_instant_adj_matrix_cycles_0_to_2204_lag0.pkl'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"No file found at {file_path}")
with open(file_path, "rb") as f:
    matrix = pickle.load(f)
adjacency_matrix = torch.tensor(matrix, dtype=torch.float32)  # Convert to tensor


# ─── Initialize Model ────────────────────────────────────────────────────
model = LinearLayer(1020, 1)  # Example: input features = 1020, output = 1 (RUL value)
model.train()  # set model to training mode (optional since new model is train by default)

# ─── Optimizer & Loss ────────────────────────────────────────────────────
if optim_name.lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Set up loss function
if loss_name.lower() == 'crossentropyloss':
    criterion = nn.CrossEntropyLoss()
elif loss_name.lower() == 'mseloss':
    criterion = nn.MSELoss()
else:
    # Attempt to get the loss class from torch.nn by name, if a different string is provided
    criterion_class = getattr(nn, loss_name, None)
    criterion = criterion_class() if criterion_class else nn.MSELoss()

# ─── Training Loop ───────────────────────────────────────────────────────
for epoch in range(epochs):
    total_loss = 0.0
    for batch in train_loader:
        # Assuming each batch is a tuple (inputs, targets)
        (tensor_100, tensor_10, tensor_1), features, rul_value, status_value = batch
        inputs = features  # Use features as input to the model
        inputs = inputs.flatten(start_dim=1)  # Flatten features to shape (batch_size, feature_dim)
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e3, neginf=-1e3)

        targets = rul_value  # Use RUL value as target
        # Forward pass: compute model predictions
        outputs = model(inputs)

        # print("outputs.shape:", outputs.shape)
        # print("targets.shape:", targets.shape)
        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    # print average loss for the epoch for monitoring
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Validation step
    model.eval()  # switch to eval mode for validation
    val_loss = 0.0
    with torch.no_grad():
        for batch in dev_loader:  # loop over validation data
            (tensor_100, tensor_10, tensor_1), features, rul_value, status_value = batch
            inputs = features  # Use features as input to the model
            inputs = inputs.flatten(start_dim=1)  # Flatten features to shape (batch_size, feature_dim)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e3, neginf=-1e3)
            targets = rul_value  # Use RUL value as target
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(dev_loader)  # average validation loss
    writer.add_scalar('Loss/val', val_loss, epoch)
    model.train()  # switch back to training mode


# 6. Save the trained model
model_save_path = 'models/ll_trained.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

writer.flush()
writer.close()
