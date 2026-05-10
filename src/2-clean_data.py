import pandas as pd

input_path = "data/balanced_flight.csv"
output_path = "data/cleaned_flight.csv"

df = pd.read_csv(input_path)

if "cancelled" in df.columns:
    df = df[df["cancelled"] == 0].copy()

required_columns = [
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

df = df[required_columns]
df = df.dropna()
df = df.drop_duplicates().reset_index(drop=True)
df["IsDelayed"] = (df["arr_delay"] > 15).astype(int)
df = df.drop(columns=["arr_delay"])

df.to_csv(output_path, index=False)

print("Saved cleaned_flight.csv")
print("Rows:", len(df))
print(df["IsDelayed"].value_counts())
