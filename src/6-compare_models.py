import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=4,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)


print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

ensemble_prob = (0.4 * rf_prob) + (0.6 * xgb_prob)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)


def evaluate(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
    }

rf_metrics = evaluate("Random Forest", y_test, rf_pred, rf_prob)
xgb_metrics = evaluate("XGBoost", y_test, xgb_pred, xgb_prob)
ensemble_metrics = evaluate("Ensemble", y_test, ensemble_pred, ensemble_prob)

results_df = pd.DataFrame([
    rf_metrics,
    xgb_metrics,
    ensemble_metrics,
]).set_index("Model")

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results_df)
print("=" * 60)

for metric in results_df.columns:
    winner = results_df[metric].idxmax()
    print(f"Best {metric}: {winner}")

plt.figure(figsize=(10, 6))

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

x = np.arange(len(metrics))
width = 0.25

rf_vals = [rf_metrics[m] for m in metrics]
xgb_vals = [xgb_metrics[m] for m in metrics]
ens_vals = [ensemble_metrics[m] for m in metrics]

plt.bar(x - width, rf_vals, width, label="Random Forest")
plt.bar(x, xgb_vals, width, label="XGBoost")
plt.bar(x + width, ens_vals, width, label="Ensemble")

plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("graph/model_comparison.png", dpi=150)
plt.show()