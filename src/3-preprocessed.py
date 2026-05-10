import pandas as pd

input_path = "data/cleaned_flight.csv"
output_dir = "data"

df = pd.read_csv(input_path)

target_col = "IsDelayed"
categorical_cols = ["op_unique_carrier", "origin", "dest"]

test_size = 0.2
class_0 = df[df[target_col] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
class_1 = df[df[target_col] == 1].sample(frac=1, random_state=42).reset_index(drop=True)

test_count_0 = int(len(class_0) * test_size)
test_count_1 = int(len(class_1) * test_size)

test_df = pd.concat([class_0.iloc[:test_count_0], class_1.iloc[:test_count_1]], ignore_index=True)
train_df = pd.concat([class_0.iloc[test_count_0:], class_1.iloc[test_count_1:]], ignore_index=True)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

for col in categorical_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.drop(columns=[target_col]).copy()
y_train = train_df[[target_col]].copy()
y_test = test_df[[target_col]].copy()

X_train = pd.get_dummies(X_train_raw, columns=categorical_cols, drop_first=False)
X_test = pd.get_dummies(X_test_raw, columns=categorical_cols, drop_first=False)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

print("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Train class ratio:", y_train[target_col].mean())
print("Test class ratio:", y_test[target_col].mean())
