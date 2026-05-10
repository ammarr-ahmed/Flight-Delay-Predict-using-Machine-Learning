import customtkinter as ctk
import joblib
import pandas as pd
import os
import threading
from collections import defaultdict


# ─────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DARK_BG = "#0f1117"
CARD_BG = "#1a1d27"
CARD2_BG = "#1e2130"

ACCENT = "#4f8ef7"

SUCCESS = "#22c55e"
DANGER = "#ef4444"

TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"

BORDER = "#2d3148"


# ─────────────────────────────────────────────
# Train models
# ─────────────────────────────────────────────
def train_and_save():

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    os.makedirs("models", exist_ok=True)

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
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

    xgb.fit(X_train, y_train)

    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb, "models/xgboost.pkl")
    joblib.dump(list(X_train.columns), "models/feature_names.pkl")


# ─────────────────────────────────────────────
# Detect feature types
# ─────────────────────────────────────────────
def detect_feature_types(features):

    numeric = []
    ohe_groups = defaultdict(list)

    for col in features:

        parts = col.rsplit("_", 1)

        if len(parts) == 2:

            prefix, suffix = parts

            if sum(1 for f in features if f.startswith(prefix + "_")) > 1:
                ohe_groups[prefix].append(suffix)
                continue

        numeric.append(col)

    return numeric, ohe_groups


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
class FlightApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("Flight Delay Predictor")

        self.geometry("1100x750")

        self.configure(fg_color=DARK_BG)

        self.rf = None
        self.xgb = None
        self.features = None

        self.numeric_fields = []
        self.ohe_groups = {}

        self.input_widgets = {}

        self._build_ui()

        self._load_models_async()

    # ─────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────
    def _build_ui(self):

        header = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=0,
            height=64
        )

        header.pack(fill="x")

        ctk.CTkLabel(
            header,
            text="✈ Flight Delay Predictor",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=20, pady=16)

        self.status_label = ctk.CTkLabel(
            header,
            text="Loading models...",
            text_color=TEXT_SECONDARY
        )

        self.status_label.pack(side="right", padx=20)

        body = ctk.CTkFrame(self, fg_color=DARK_BG)

        body.pack(fill="both", expand=True, padx=20, pady=20)

        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        # LEFT PANEL
        self.left_panel = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=16
        )

        self.left_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        ctk.CTkLabel(
            self.left_panel,
            text="Flight Details",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=24, pady=(20, 8))

        self.form_frame = ctk.CTkScrollableFrame(
            self.left_panel,
            fg_color="transparent"
        )

        self.form_frame.pack(
            fill="both",
            expand=True,
            padx=16,
            pady=10
        )

        self.predict_btn = ctk.CTkButton(
            self.left_panel,
            text="Predict",
            height=50,
            fg_color=ACCENT,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._on_predict
        )

        self.predict_btn.pack(
            fill="x",
            padx=24,
            pady=20
        )

        # RIGHT PANEL
        self.right_panel = ctk.CTkFrame(
            body,
            fg_color=CARD_BG,
            corner_radius=16
        )

        self.right_panel.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        ctk.CTkLabel(
            self.right_panel,
            text="Prediction Result",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=24, pady=(20, 10))

        # Verdict card
        self.verdict_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=CARD2_BG,
            corner_radius=14
        )

        self.verdict_frame.pack(
            fill="x",
            padx=20,
            pady=(0, 20)
        )

        self.verdict_icon = ctk.CTkLabel(
            self.verdict_frame,
            text="—",
            font=ctk.CTkFont(size=42),
            text_color=TEXT_SECONDARY
        )

        self.verdict_icon.pack(pady=(20, 5))

        self.verdict_label = ctk.CTkLabel(
            self.verdict_frame,
            text="Awaiting Input",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_SECONDARY
        )

        self.verdict_label.pack()

        self.verdict_sub = ctk.CTkLabel(
            self.verdict_frame,
            text="",
            text_color=TEXT_SECONDARY
        )

        self.verdict_sub.pack(pady=(5, 20))

        # RF CARD
        self.rf_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=CARD2_BG,
            corner_radius=12
        )

        self.rf_frame.pack(fill="x", padx=20, pady=10)

        self._build_model_card(
            self.rf_frame,
            "Random Forest",
            "rf"
        )

        # XGB CARD
        self.xgb_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=CARD2_BG,
            corner_radius=12
        )

        self.xgb_frame.pack(fill="x", padx=20, pady=10)

        self._build_model_card(
            self.xgb_frame,
            "XGBoost",
            "xgb"
        )

    # ─────────────────────────────────────────
    # Build model card
    # ─────────────────────────────────────────
    def _build_model_card(self, parent, title, key):

        row = ctk.CTkFrame(parent, fg_color="transparent")

        row.pack(fill="x", padx=16, pady=14)

        row.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            row,
            text=title,
            width=130,
            anchor="w",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_PRIMARY
        ).grid(row=0, column=0)

        pred_lbl = ctk.CTkLabel(
            row,
            text="—",
            text_color=TEXT_SECONDARY
        )

        pred_lbl.grid(row=0, column=1)

        prob_lbl = ctk.CTkLabel(
            row,
            text="",
            text_color=TEXT_SECONDARY
        )

        prob_lbl.grid(row=0, column=2)

        bar_frame = ctk.CTkFrame(
            parent,
            fg_color=BORDER,
            height=4
        )

        bar_frame.pack(fill="x", padx=16, pady=(0, 10))

        bar_fill = ctk.CTkFrame(
            bar_frame,
            fg_color=ACCENT,
            height=4,
            width=0
        )

        bar_fill.place(x=0, y=0)

        setattr(self, f"{key}_pred_lbl", pred_lbl)
        setattr(self, f"{key}_prob_lbl", prob_lbl)
        setattr(self, f"{key}_bar_frame", bar_frame)
        setattr(self, f"{key}_bar_fill", bar_fill)

    # ─────────────────────────────────────────
    # Load models
    # ─────────────────────────────────────────
    def _load_models_async(self):

        def _load():

            os.makedirs("models", exist_ok=True)

            if not os.path.exists("models/random_forest.pkl"):

                self.status_label.configure(
                    text="Training models..."
                )

                train_and_save()

            self.rf = joblib.load("models/random_forest.pkl")

            self.xgb = joblib.load("models/xgboost.pkl")

            self.features = joblib.load(
                "models/feature_names.pkl"
            )

            self.numeric_fields, self.ohe_groups = detect_feature_types(
                self.features
            )

            self.after(0, self._build_form)

            self.after(
                0,
                lambda: self.status_label.configure(
                    text="Models Ready",
                    text_color=SUCCESS
                )
            )

        threading.Thread(target=_load, daemon=True).start()

    # ─────────────────────────────────────────
    # Build form
    # ─────────────────────────────────────────
    def _build_form(self):

        # Numeric fields
        for col in self.numeric_fields:

            label = ctk.CTkLabel(
                self.form_frame,
                text=col,
                text_color=TEXT_SECONDARY
            )

            label.pack(anchor="w", padx=10, pady=(8, 2))

            entry = ctk.CTkEntry(
                self.form_frame,
                height=38
            )

            entry.pack(fill="x", padx=10)

            self.input_widgets[col] = entry

        # Categorical fields
        for prefix, values in self.ohe_groups.items():

            label = ctk.CTkLabel(
                self.form_frame,
                text=prefix,
                text_color=TEXT_SECONDARY
            )

            label.pack(anchor="w", padx=10, pady=(8, 2))

            combo = ctk.CTkComboBox(
                self.form_frame,
                values=sorted(values),
                height=38
            )

            combo.pack(fill="x", padx=10)

            combo.set(sorted(values)[0])

            self.input_widgets[prefix] = combo

    # ─────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────
    def _on_predict(self):

        threading.Thread(
            target=self._run_prediction,
            daemon=True
        ).start()

    def _run_prediction(self):

        row = {col: 0 for col in self.features}

        # Numeric
        for col in self.numeric_fields:

            widget = self.input_widgets[col]

            try:
                row[col] = float(widget.get() or 0)

            except:
                row[col] = 0

        # Categorical
        for prefix, values in self.ohe_groups.items():

            widget = self.input_widgets[prefix]

            value = widget.get()

            if f"{prefix}_{value}" in row:
                row[f"{prefix}_{value}"] = 1

        df = pd.DataFrame([row])[self.features]

        rf_pred = self.rf.predict(df)[0]
        rf_prob = self.rf.predict_proba(df)[0][1]

        xgb_pred = self.xgb.predict(df)[0]
        xgb_prob = self.xgb.predict_proba(df)[0][1]

        self.after(
            0,
            lambda: self._show_results(
                int(rf_pred),
                rf_prob,
                int(xgb_pred),
                xgb_prob
            )
        )

    # ─────────────────────────────────────────
    # Show Results
    # ─────────────────────────────────────────
    def _show_results(self, rf_pred, rf_prob, xgb_pred, xgb_prob):

        label = {0: "No Delay", 1: "Delayed"}

        color = {
            0: SUCCESS,
            1: DANGER
        }

        icon = {
            0: "✓",
            1: "⚠"
        }

        # Individual models
        for key, pred, prob in [
            ("rf", rf_pred, rf_prob),
            ("xgb", xgb_pred, xgb_prob)
        ]:

            getattr(self, f"{key}_pred_lbl").configure(
                text=label[pred],
                text_color=color[pred]
            )

            getattr(self, f"{key}_prob_lbl").configure(
                text=f"{prob:.1%}",
                text_color=color[pred]
            )

            bar_frame = getattr(self, f"{key}_bar_frame")

            bar_fill = getattr(self, f"{key}_bar_fill")

            bar_fill.configure(
                fg_color=color[pred]
            )

            self._animate_bar(
                bar_fill,
                bar_frame,
                prob
            )

        # Ensemble prediction
        final_prob = (0.4 * rf_prob) + (0.6 * xgb_prob)

        final_pred = 1 if final_prob >= 0.5 else 0

        # Risk level
        if final_prob < 0.4:

            risk = "Low Risk"

        elif final_prob < 0.7:

            risk = "Medium Risk"

        else:

            risk = "High Risk"

        # Final verdict
        self.verdict_icon.configure(
            text=icon[final_pred],
            text_color=color[final_pred]
        )

        self.verdict_label.configure(
            text=f"{label[final_pred]} • {risk}",
            text_color=color[final_pred]
        )

        # Agreement status
        if rf_pred != xgb_pred:

            self.verdict_sub.configure(
                text=f"Models disagree • Confidence {final_prob:.1%}",
                text_color="#f59e0b"
            )

        else:

            self.verdict_sub.configure(
                text=f"Both models agree • Confidence {final_prob:.1%}",
                text_color=TEXT_SECONDARY
            )

    # ─────────────────────────────────────────
    # Animate bar
    # ─────────────────────────────────────────
    def _animate_bar(self, bar_fill, bar_frame, prob, step=0):

        target_rel = prob

        current = step / 40

        if current <= target_rel:

            bar_frame.update_idletasks()

            w = bar_frame.winfo_width()

            bar_fill.configure(
                width=int(w * current)
            )

            self.after(
                12,
                lambda: self._animate_bar(
                    bar_fill,
                    bar_frame,
                    prob,
                    step + 1
                )
            )


# ─────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────
if __name__ == "__main__":

    app = FlightApp()

    app.mainloop()