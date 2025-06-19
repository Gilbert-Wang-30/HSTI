import pandas as pd

# Load profile.txt (assuming space-separated and only first 4 columns are used)
profile_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/data/profile.txt"
df = pd.read_csv(profile_path, delim_whitespace=True, header=None, usecols=[0, 1, 2, 3])
df.columns = ["cooler", "valve", "pump", "accumulator"]


# Normalize helper
def normalize(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

# Normalize each component
cooler_score = normalize(df["cooler"], 3, 100).clip(0, 1)
valve_score = normalize(df["valve"], 73, 100).clip(0, 1)
pump_score = (1 - normalize(df["pump"], 0, 2)).clip(0, 1)
acc_score = normalize(df["accumulator"], 90, 130).clip(0, 1)

# Stack into one DataFrame
scores = pd.concat([cooler_score, valve_score, pump_score, acc_score], axis=1)
scores.columns = ["cooler", "valve", "pump", "accumulator"]

# Compute dynamic penalty based on worst component
min_score = scores.min(axis=1)
weight = 0.3 * min_score + 0.7

# Final RUL score
rul_score = weight * scores.mean(axis=1) * 100  # In percentage
# Save to rul_profile.txt
output_df = pd.DataFrame({"RUL": rul_score.round(2)})
output_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/data/rul_profile.txt"
# Save to rul_profile.txt (one score per line, no header or index)
output_df.to_csv(output_path, index=False, header=False, float_format="%.2f")
