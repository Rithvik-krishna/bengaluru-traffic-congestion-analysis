"""
ml_congestion_prediction.py
===========================
Machine Learning module for Bengaluru Traffic Congestion Analysis.

Predicts: Congestion Level (regression)
Models  : Linear Regression (baseline) | Random Forest Regressor (primary)

Outputs:
  outputs/ml_actual_vs_predicted.png
  outputs/ml_feature_importance.png
  docs/ml_metrics.json
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────────────────────
# 0. Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "cleaned_traffic_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load Cleaned Dataset (DO NOT modify the original CSV)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading cleaned dataset...")
df_clean = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df_clean):,}  |  Columns: {df_clean.shape[1]}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Create a Separate ML DataFrame
# ──────────────────────────────────────────────────────────────────────────────
ML_FEATURES = [
    "traffic_volume",
    "average_speed",
    "travel_time_index",
    "road_capacity_utilization",
    "incident_reports",
    "environmental_impact",
    "public_transport_usage",
    "traffic_signal_compliance",
    "parking_usage",
    "pedestrian_and_cyclist_count",
    "weather_conditions",
    "roadwork_and_construction_activity",
    "area_name",
]
TARGET = "congestion_level"

df_ml = df_clean[ML_FEATURES + [TARGET]].copy()
print(f"\nML DataFrame shape: {df_ml.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. One-Hot Encode Categorical Columns
# ──────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["weather_conditions", "roadwork_and_construction_activity", "area_name"]
print("\nApplying one-hot encoding to:", CATEGORICAL_COLS)

df_encoded = pd.get_dummies(df_ml, columns=CATEGORICAL_COLS, drop_first=True)
print(f"  Encoded DataFrame shape: {df_encoded.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Train / Test Split (80 / 20, random_state=42)
# ──────────────────────────────────────────────────────────────────────────────
X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Model 1 — Linear Regression (Baseline)
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Model 1: Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"  MAE  : {mae_lr:.4f}")
print(f"  RMSE : {rmse_lr:.4f}")
print(f"  R²   : {r2_lr:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Model 2 — Random Forest Regressor (Primary)
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Model 2: Random Forest Regressor ---")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print(f"  MAE  : {mae_rf:.4f}")
print(f"  RMSE : {rmse_rf:.4f}")
print(f"  R²   : {r2_rf:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Model Comparison
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Model Comparison ---")
print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 60)
print(f"{'Linear Regression':<30} {mae_lr:>8.4f} {rmse_lr:>8.4f} {r2_lr:>8.4f}")
print(f"{'Random Forest Regressor':<30} {mae_rf:>8.4f} {rmse_rf:>8.4f} {r2_rf:>8.4f}")

best_model = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
if best_model == "Random Forest":
    summary_text = (
        "The Random Forest model achieved a higher R\u00b2 score and lower prediction "
        "error, making it the preferred model for congestion prediction."
    )
else:
    summary_text = (
        "The Linear Regression baseline achieved competitive performance, "
        "suggesting the relationships in this dataset are largely linear."
    )
print(f"\nBest Model: {best_model}")
print(f"Summary   : {summary_text}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Feature Importance (Top 10)
# ──────────────────────────────────────────────────────────────────────────────
feature_names = X.columns.tolist()
importances = rf.feature_importances_

feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
top10 = feat_series.head(10)
top10_list = [{"feature": f, "importance": round(float(v), 6)} for f, v in top10.items()]
print("\nTop 10 Feature Importances:")
for item in top10_list:
    print(f"  {item['feature']:<45} {item['importance']:.6f}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. Save Metrics to JSON
# ──────────────────────────────────────────────────────────────────────────────
metrics = {
    "linear_regression": {
        "mae":  round(float(mae_lr),  4),
        "rmse": round(float(rmse_lr), 4),
        "r2":   round(float(r2_lr),   4),
    },
    "random_forest": {
        "mae":  round(float(mae_rf),  4),
        "rmse": round(float(rmse_rf), 4),
        "r2":   round(float(r2_rf),   4),
    },
    "best_model": best_model,
    "summary": summary_text,
    "top_features": top10_list,
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
}

metrics_path = os.path.join(DOCS_DIR, "ml_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {metrics_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 10. Visualization 1 — Actual vs Predicted (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────
ACCENT  = "#0EA5E9"
PRIMARY = "#0F172A"
GRID_C  = "#E2E8F0"

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")

ax.scatter(
    y_test, y_pred_rf,
    alpha=0.45, s=18, color=ACCENT, edgecolors="none", label="Predictions"
)

# Diagonal perfect-prediction line
lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
ax.plot(lims, lims, color="#EF4444", linewidth=1.8, linestyle="--", label="Perfect Fit")

ax.set_xlabel("Actual Congestion Level (%)", fontsize=12, color=PRIMARY)
ax.set_ylabel("Predicted Congestion Level (%)", fontsize=12, color=PRIMARY)
ax.set_title(
    "Actual vs Predicted Congestion Level\n(Random Forest Regressor)",
    fontsize=14, fontweight="bold", color=PRIMARY, pad=16
)

# Metrics annotation box
annotation = (
    f"R² = {r2_rf:.4f}\n"
    f"MAE = {mae_rf:.4f}\n"
    f"RMSE = {rmse_rf:.4f}"
)
ax.text(
    0.04, 0.96, annotation,
    transform=ax.transAxes,
    fontsize=10, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID_C, alpha=0.9)
)

ax.grid(True, color=GRID_C, linewidth=0.7, linestyle="-")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9)
ax.tick_params(colors=PRIMARY, labelsize=10)

plt.tight_layout()
out_path_avp = os.path.join(OUT_DIR, "ml_actual_vs_predicted.png")
plt.savefig(out_path_avp, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path_avp}")

# ──────────────────────────────────────────────────────────────────────────────
# 11. Visualization 2 — Feature Importance (Top 10)
# ──────────────────────────────────────────────────────────────────────────────
# Clean feature names for display
def clean_feat_name(name):
    """Convert snake_case / dummy-encoded names to a readable label."""
    name = name.replace("_", " ")
    # Capitalise first letter of each word
    return " ".join(w.capitalize() for w in name.split())

labels     = [clean_feat_name(f) for f in top10.index]
values     = top10.values
bar_colors = [ACCENT] + ["#38BDF8"] * (len(values) - 1)  # highlight top feature

fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.patch.set_facecolor("#F8FAFC")
ax2.set_facecolor("#F8FAFC")

bars = ax2.barh(labels[::-1], values[::-1], color=bar_colors[::-1],
                height=0.62, edgecolor="none")

# Value labels on bars
for bar, val in zip(bars, values[::-1]):
    ax2.text(
        val + 0.0005, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=9, color=PRIMARY
    )

ax2.set_xlabel("Feature Importance (Gini)", fontsize=11, color=PRIMARY)
ax2.set_title(
    "Top 10 Feature Importances\n(Random Forest Regressor)",
    fontsize=14, fontweight="bold", color=PRIMARY, pad=14
)

legend_patches = [
    mpatches.Patch(color=ACCENT,   label="Top Feature"),
    mpatches.Patch(color="#38BDF8", label="Other Features"),
]
ax2.legend(handles=legend_patches, fontsize=9, framealpha=0.9)

ax2.grid(True, axis="x", color=GRID_C, linewidth=0.7, linestyle="-")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.tick_params(axis="y", length=0, labelsize=10, colors=PRIMARY)
ax2.tick_params(axis="x", colors=PRIMARY, labelsize=9)

plt.tight_layout()
out_path_fi = os.path.join(OUT_DIR, "ml_feature_importance.png")
plt.savefig(out_path_fi, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {out_path_fi}")

print("\nML pipeline complete. All outputs saved successfully.")
