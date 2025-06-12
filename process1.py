# process1.py
import os
import pickle
from torch.utils.data import DataLoader
from high_level_feature_extraction import extract_high_level_features, extract_stat_features, load_train_data

def main():
    # Define paths
    base_dir = "/home/wangyuxiao/project/gilbert_copy/HSTI"
    train_pkl_path = os.path.join(base_dir, "processed_data/train.pkl")

    # Load the training data (preserve batch structure)
    train_loader = load_train_data(train_pkl_path, batch_size=16, shuffle=True)

    # Step 1: Extract high-level statistical features
    print("[Step 1] Extracting high-level statistical features...")
    features, labels, x100_raw, x10_raw, x1_raw = extract_high_level_features(train_loader)
    print(f"  > Extracted {len(features)} batches.")
    print(f"  > Feature shape (per batch): {features[0].shape}")
    print(f"  > Label shape (per batch): {labels[0].shape}")
    print(f"  > Raw 100Hz shape (per batch): {x100_raw[0].shape}")
    print(f"  > Raw 10Hz shape (per batch): {x10_raw[0].shape}")
    print(f"  > Raw 1Hz shape (per batch): {x1_raw[0].shape}")

    # (Future steps go here)
    # Step 2: Causality analysis
    # Step 3: Structural Feature Graph (SFG) construction
    # Step 4: Graph modeling or temporal reasoning
    print("[Info] Pipeline stage 1 complete. Ready for causality/SFG steps.")

if __name__ == "__main__":
    main()
