import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from models.raw_prediction_fix_model import RawFix  # or adjust import if you have it in another file

BATCH_SIZE = 16
EPOCHS = 50
LR = 0.01
input_size = 10    # e.g. for 100Hz, change as needed
num_features = 14    # for your high-level features

DATA_ROOT = Path("data/processed")
LOG_DIR = Path("runs/lstm_fix")

def custom_collate(batch):
    # Batch is a list of (input_dict, target)
    lstm_raw = []
    lstm_feat = []
    lr_feat = []
    y = []
    for input_dict, target in batch:
        lstm_raw.append(input_dict["lstm_pred_present"])
        lstm_feat.append(input_dict["lstm_pred_present_highlevel"])
        lr_feat.append(input_dict["lr_pred_present"])
        y.append(target)
    lstm_raw = np.stack(lstm_raw)   # (batch, input_size)
    y        = np.stack(y)
    # Normalize
    lstm_raw = (lstm_raw - mean_x) / (std_x + 1e-6)
    y        = (y        - mean_y) / (std_y + 1e-6)
    # The features are already normalized or ~unit-scale, so just keep as float32
    lstm_feat = np.stack(lstm_feat)
    lr_feat = np.stack(lr_feat)
    # Convert to tensor
    lstm_raw = torch.tensor(lstm_raw, dtype=torch.float32)
    lstm_feat = torch.tensor(lstm_feat, dtype=torch.float32)
    lr_feat = torch.tensor(lr_feat, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return lstm_raw, lstm_feat, lr_feat, y

def load_pkl(name):
    with open(DATA_ROOT / name, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Choose sensor type, file, etc.
    data_file = "train_1hz.pkl"
    val_file  = "val_1hz.pkl"
    train_data = load_pkl(data_file)
    val_data = load_pkl(val_file)
    
    
    all_train_x = []
    all_train_y = []
    for inp, tgt in train_data:
        all_train_x.append(inp["lstm_pred_present"])
        all_train_y.append(tgt)
    all_train_x = np.stack(all_train_x)  # (N, input_size)
    all_train_y = np.stack(all_train_y)  # (N, input_size)

    mean_x = all_train_x.mean(axis=0)  # shape: (input_size,)
    std_x  = all_train_x.std(axis=0)   # shape: (input_size,)
    mean_y = all_train_y.mean(axis=0)
    std_y  = all_train_y.std(axis=0)
    print("mean_x:", mean_x)
    print("std_x:", std_x)
    print("mean_y:", mean_y)
    print("std_y:", std_y)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = RawFix(input_size=input_size, num_features=num_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = torch.nn.MSELoss()
    
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    best_val_loss = float("inf")
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0
        for x, x_feat, lr_feat, y in train_loader:
            x, x_feat, lr_feat, y = x.to(device), x_feat.to(device), lr_feat.to(device), y.to(device)
            # replace NaNs in lr_feat with 0
            lr_feat = torch.nan_to_num(lr_feat, nan=0.0)
            optimizer.zero_grad()
            y_pred = model(x, x_feat, lr_feat)
            mask = ~torch.isnan(y)
            # print(f"Epoch {epoch}, Batch Size: {x.size(0)}, Masked Elements: {mask.sum()/mask.numel():.4f}")
            # print("Mask shape:", mask.shape, "Sum:", mask.sum().item(), "Numel:", mask.numel())
            # print("Any NaN in y:", torch.isnan(y).any().item())

            loss = ((y_pred - y)[mask]) ** 2
            loss = loss.mean()
            # print(f"Epoch {epoch}, Batch Loss: {loss.item():.6f}")
            loss.backward()
            optimizer.step()
            # print(f"Epoch {epoch}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            train_loss += loss.item() * x.size(0)
            # print(f"Epoch {epoch}, X mean: {x.mean().item():.4f}, Y mean: {y.mean().item():.4f}")
            # print(f"Epoch {epoch}, Y Pred mean: {y_pred.mean().item():.4f}")
            # print(f"Epoch {epoch}, X Features mean: {x_feat.mean().item():.4f}")
            # print(f"Epoch {epoch}, LR Features mean: {lr_feat.mean().item():.4f}")

        train_loss /= len(train_loader.dataset)
        scheduler.step()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, x_feat, lr_feat, y in val_loader:
                x, x_feat, lr_feat, y = x.to(device), x_feat.to(device), lr_feat.to(device), y.to(device)
                lr_feat = torch.nan_to_num(lr_feat, nan=0.0)
                y_pred = model(x, x_feat, lr_feat)
                mask = ~torch.isnan(y)
                loss = ((y_pred - y)[mask]) ** 2
                loss = loss.mean()

                val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
        
        print(f"[Epoch {epoch:02d}] Train: {train_loss:.6f}  Val: {val_loss:.6f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "rawfix_best.pth")
            print(f"  [Checkpoint] Saved at epoch {epoch} (val loss {val_loss:.6f})")

    writer.close()
    print("Training complete. Best val loss: {:.6f}".format(best_val_loss))
