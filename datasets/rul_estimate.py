import pandas as pd
import numpy as np

# Load profile.txt (assuming space-separated and only first 4 columns are used)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
profile_path = BASE_DIR / "data" / "raw" / "profile.txt"
output_path = BASE_DIR / "data" / "raw" / "rul_profile.txt"
df = pd.read_csv(profile_path, delim_whitespace=True, header=None, usecols=[0, 1, 2, 3])
df.columns = ["cooler", "valve", "pump", "accumulator"]


# ─── Define Mapping Dictionaries ───────────────────────────────────────────
cooler_map = {3: 0, 20: 1, 100: 2}
valve_map = {100: 0, 90: 1, 80: 2, 73: 3}
pump_map = {0: 0, 1: 1, 2: 2}
accumulator_map = {130: 0, 115: 1, 100: 2, 90: 3}

# ─── Apply Mappings ────────────────────────────────────────────────────────
cooler_class = df["cooler"].map(cooler_map)
valve_class = df["valve"].map(valve_map)
pump_class = df["pump"].map(pump_map)
accumulator_class = df["accumulator"].map(accumulator_map)

# Final RUL score
rul_score = rul_score = 1.0 - (np.arange(len(df)) / len(df))

# ─── Save to File ──────────────────────────────────────────────────────────
output_df = pd.DataFrame({
    "RUL": rul_score.round(4),
    "cooler_class": cooler_class,
    "valve_class": valve_class,
    "pump_class": pump_class,
    "accumulator_class": accumulator_class
})

# Save to file (no header, no index)
output_df.to_csv(output_path, index=False, header=False, float_format="%.4f")

print(f"RUL and class labels saved to {output_path}")