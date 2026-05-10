import pandas as pd
import joblib

rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")
features = joblib.load("models/feature_names.pkl")

print("\nFlight Delay Prediction")
print("=" * 40)

row = {}

for col in features:
    val = input(f"Enter {col}: ")
    row[col] = float(val)

input_df = pd.DataFrame([row])

rf_pred = rf.predict(input_df)[0]
rf_prob = rf.predict_proba(input_df)[0][1]

xgb_pred = xgb.predict(input_df)[0]
xgb_prob = xgb.predict_proba(input_df)[0][1]

final_prob = (0.4 * rf_prob) + (0.6 * xgb_prob)
final_pred = 1 if final_prob >= 0.5 else 0

label = {0: "No Delay", 1: "Delayed"}

print("\n" + "=" * 40)
print("Prediction Results")
print("=" * 40)
print(f"Random Forest : {label[rf_pred]} ({rf_prob:.1%})")
print(f"XGBoost       : {label[xgb_pred]} ({xgb_prob:.1%})")
print("-" * 40)
print(f"Final Result  : {label[final_pred]} ({final_prob:.1%})")

if final_prob < 0.4:
    risk = "Low Risk"
elif final_prob < 0.7:
    risk = "Medium Risk"
else:
    risk = "High Risk"

print(f"Risk Level    : {risk}")

if rf_pred != xgb_pred:
    print("\nModels disagree — Moderate confidence")