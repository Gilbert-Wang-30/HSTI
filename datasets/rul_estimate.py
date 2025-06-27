import pandas as pd
import numpy as np

# Load profile.txt (assuming space-separated and only first 4 columns are used)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
profile_path = BASE_DIR / "data" / "raw" / "profile.txt"
output_path = BASE_DIR / "data" / "raw" / "rul_profile.txt"
df = pd.read_csv(profile_path, delim_whitespace=True, header=None, usecols=[0, 1, 2, 3])
df.columns = ["cooler", "valve", "pump", "accumulator"]


# Normalize each component
cooler_score = df["cooler"]
valve_score = df["valve"]
pump_score = df["pump"]
acc_score = df["accumulator"]

# Final RUL score
rul_score = rul_score = 1.0 - (np.arange(len(df)) / len(df))

# Save to rul_profile.txt
output_df = pd.DataFrame({"RUL": rul_score.round(4), "cooler_score": cooler_score,
                          "valve_score": valve_score,
                          "pump_score": pump_score, "accumulator_score": acc_score})
# Save to rul_profile.txt (one score per line, no header or index)
output_df.to_csv(output_path, index=False, header=False, float_format="%.4f")

print(f"RUL scores saved to {output_path}")