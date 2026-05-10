import pandas as pd

input_path = "data/flight.csv"
output_path = "data/balanced_flight.csv"

df = pd.read_csv(input_path)
df = df[df["cancelled"] == 0].copy()
df = df[df["diverted"] == 0].copy()

important_columns = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
    "origin",
    "dest",
    "crs_dep_time",
    "distance",
    "dep_delay",
    "arr_delay",
]

df = df[important_columns]
df = df.dropna(subset=["arr_delay", "crs_dep_time", "distance", "dep_delay"])
df["IsDelayed"] = (df["arr_delay"] > 15).astype(int)

df_balanced = df.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced.to_csv(output_path, index=False)

print("Saved balanced_flight.csv")
print("Rows:", len(df_balanced))
print(df_balanced["IsDelayed"].value_counts())