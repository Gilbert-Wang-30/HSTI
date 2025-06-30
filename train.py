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
from models.ll import MultiTaskModel  # Assuming this is a simple linear layer model for testing
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
status_classes = [3, 4, 3, 4]
model = MultiTaskModel(1020, status_classes)  # Example: input features = 1020, output = 1 (RUL value)
model.train()  # set model to training mode (optional since new model is train by default)

# ─── Optimizer & Loss ────────────────────────────────────────────────────
if optim_name.lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)

# Set up loss function
criterion_rul = nn.MSELoss()
criterion_status = nn.CrossEntropyLoss()

# ─── Training Loop ───────────────────────────────────────────────────────
for epoch in range(epochs):
    model.train()  # set model to training mode
    total_rul_loss = 0.0
    correct_status = [0, 0, 0, 0]
    total_samples = 0

    for batch in train_loader:
        # Assuming each batch is a tuple (inputs, targets)
        (tensor_100, tensor_10, tensor_1), features, rul_value, status_value = batch
        inputs = features  # Use features as input to the model
        inputs = inputs.flatten(start_dim=1)  # Flatten features to shape (batch_size, feature_dim)
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e3, neginf=-1e3)

        # Forward pass
        rul_pred, status_logits, _ = model(inputs)

        # loss computation
        rul_loss = criterion_rul(rul_pred, rul_value)
        status_losses = [
            criterion_status(logits, status_value[:, j].long())
            for j, logits in enumerate(status_logits)
        ]


        total_loss = rul_loss + sum(status_losses)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Backward pass and optimization step
        
        total_rul_loss += rul_loss.item()
        total_samples += rul_value.size(0)

        for j, logits in enumerate(status_logits):
            preds = logits.argmax(dim=1)
            correct_status[j] += (preds == status_value[:, j].long()).sum().item()


    
    # print average loss for the epoch for monitoring
    avg_train_rul_loss = total_rul_loss / len(train_loader)
    precision_train = [correct / total_samples for correct in correct_status]

    writer.add_scalar('rul_loss/train', avg_train_rul_loss, epoch)
    for j in range(4):
        writer.add_scalar(f'status{j}_precision/train', precision_train[j], epoch)


    # Validation step
    model.eval()
    val_rul_loss = 0.0
    correct_status_val = [0, 0, 0, 0]
    total_val_samples = 0

    with torch.no_grad():
        for batch in dev_loader:
            (tensor_100, tensor_10, tensor_1), features, rul_value, status_value = batch
            inputs = features.flatten(start_dim=1)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e3, neginf=-1e3)

            rul_pred, status_logits, _ = model(inputs)
            val_rul_loss += criterion_rul(rul_pred, rul_value).item()
            total_val_samples += rul_value.size(0)

            for j, logits in enumerate(status_logits):
                preds = logits.argmax(dim=1)
                correct_status_val[j] += (preds == status_value[:, j].long()).sum().item()

    avg_val_rul_loss = val_rul_loss / len(dev_loader)
    precision_val = [correct / total_val_samples for correct in correct_status_val]

    writer.add_scalar('rul_loss/val', avg_val_rul_loss, epoch)
    for j in range(4):
        writer.add_scalar(f'status{j}_precision/val', precision_val[j], epoch)

    print(f"Epoch {epoch+1}/{epochs}, RUL Loss: {avg_train_rul_loss:.4f}, " +
          " ".join([f"S{j}_Prec: {precision_train[j]*100:.1f}%" for j in range(4)]))

# ─── Save Trained Model ───
model_save_path = 'models/ll_trained.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

writer.flush()
writer.close()
