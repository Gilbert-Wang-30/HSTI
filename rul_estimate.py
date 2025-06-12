import pandas as pd

# Load profile.txt (assuming space-separated and only first 4 columns are used)
profile_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/data/profile.txt"
df = pd.read_csv(profile_path, delim_whitespace=True, header=None, usecols=[0, 1, 2, 3])
df.columns = ["cooler", "valve", "pump", "accumulator"]

# Normalize scores for each component
def normalize(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

# Normalize each component based on domain knowledge
cooler_score = normalize(df["cooler"], 3, 100)
valve_score = normalize(df["valve"], 73, 100)
pump_score = normalize(df["pump"], 0, 2)
acc_score = normalize(df["accumulator"], 90, 130)

# Clamp scores to [0, 1] in case of small deviations
cooler_score = cooler_score.clip(0, 1)
valve_score = valve_score.clip(0, 1)
pump_score = pump_score.clip(0, 1)
pump_score = 1 - pump_score  # Invert pump score since lower is better
acc_score = acc_score.clip(0, 1)

# Compute weighted average RUL score
rul_score = (cooler_score + valve_score + pump_score + acc_score) / 4 * 100

# Save to rul_profile.txt
output_df = pd.DataFrame({"RUL": rul_score.round(2)})
output_path = "/home/wangyuxiao/project/gilbert_copy/HSTI/data/rul_profile.txt"
# Save to rul_profile.txt (one score per line, no header or index)
output_df.to_csv(output_path, index=False, header=False, float_format="%.2f")
