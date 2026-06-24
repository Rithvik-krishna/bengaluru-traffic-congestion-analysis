"""
ml_congestion_prediction.py  (v2 — data-leakage-free, academically rigorous)
=============================================================================
Changes from v1:
  - REMOVED environmental_impact (R²=1.0 w.r.t. traffic_volume+speed = data leakage)
  - Added: model comparison bar chart
  - Added: feature correlation heatmap
  - Added: residual plot
  - Added: train/test split pie chart
  - Saves: docs/ml_metrics.json  (updated, no env_impact)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "cleaned_traffic_data.csv")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
DOCS_DIR  = os.path.join(BASE_DIR, "docs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Design tokens (match website palette)
C_PRIMARY  = "#0F172A"
C_ACCENT   = "#0EA5E9"
C_SUCCESS  = "#10B981"
C_WARN     = "#F59E0B"
C_DANGER   = "#EF4444"
C_BG       = "#F8FAFC"
C_BORDER   = "#E2E8F0"
C_BLUE2    = "#38BDF8"

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  Cols: {df.shape[1]}")

# ── 2. ML feature set (environmental_impact REMOVED — data leakage) ────────────
ML_FEATURES = [
    "traffic_volume",
    "average_speed",
    "travel_time_index",
    "road_capacity_utilization",
    "incident_reports",
    "public_transport_usage",
    "traffic_signal_compliance",
    "parking_usage",
    "pedestrian_and_cyclist_count",
    "weather_conditions",
    "roadwork_and_construction_activity",
    "area_name",
]
TARGET = "congestion_level"

df_ml = df[ML_FEATURES + [TARGET]].copy()
print(f"  ML features: {len(ML_FEATURES)} (environmental_impact excluded — data leakage)")

# ── 3. Encode categoricals ─────────────────────────────────────────────────────
CAT_COLS = ["weather_conditions", "roadwork_and_construction_activity", "area_name"]
df_enc   = pd.get_dummies(df_ml, columns=CAT_COLS, drop_first=True)
print(f"  Encoded shape: {df_enc.shape}")

X = df_enc.drop(columns=[TARGET])
y = df_enc[TARGET]

# ── 4. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── 5. Models ──────────────────────────────────────────────────────────────────
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
yp_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, yp_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, yp_lr))
r2_lr   = r2_score(y_test, yp_lr)
print(f"  MAE={mae_lr:.4f}  RMSE={rmse_lr:.4f}  R2={r2_lr:.4f}")

print("Training Random Forest (200 trees)...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yp_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, yp_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, yp_rf))
r2_rf   = r2_score(y_test, yp_rf)
print(f"  MAE={mae_rf:.4f}  RMSE={rmse_rf:.4f}  R2={r2_rf:.4f}")

# ── 6. Feature importance ──────────────────────────────────────────────────────
feat_imp   = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top10      = feat_imp.head(10)
top10_list = [{"feature": f, "importance": round(float(v), 6)} for f, v in top10.items()]

# ── 7. Save metrics JSON ───────────────────────────────────────────────────────
best_model   = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
summary_text = (
    "The Random Forest model achieved a higher R\u00b2 score and lower prediction "
    "error, making it the preferred model for congestion prediction."
) if best_model == "Random Forest" else (
    "The Linear Regression baseline achieved competitive performance."
)

metrics = {
    "linear_regression": {"mae": round(float(mae_lr),4), "rmse": round(float(rmse_lr),4), "r2": round(float(r2_lr),4)},
    "random_forest":     {"mae": round(float(mae_rf),4), "rmse": round(float(rmse_rf),4), "r2": round(float(r2_rf),4)},
    "best_model":  best_model,
    "summary":     summary_text,
    "top_features": top10_list,
    "train_size":  int(len(X_train)),
    "test_size":   int(len(X_test)),
    "leakage_note": "environmental_impact removed (perfectly derived from traffic_volume+average_speed, R2=1.0)",
}
with open(os.path.join(DOCS_DIR, "ml_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved ml_metrics.json")

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")
    return path

def styled_ax(ax, fig):
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=C_BORDER, linewidth=0.7)
    ax.tick_params(colors=C_PRIMARY, labelsize=9)
    return ax

# ── VIZ 1 : Actual vs Predicted ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
styled_ax(ax, fig)
ax.scatter(y_test, yp_rf, alpha=0.40, s=15, color=C_ACCENT, edgecolors="none", label="Predictions")
lims = [min(y_test.min(), yp_rf.min()), max(y_test.max(), yp_rf.max())]
ax.plot(lims, lims, color=C_DANGER, linewidth=1.8, linestyle="--", label="Perfect Fit")
ax.set_xlabel("Actual Congestion Level (%)", fontsize=11, color=C_PRIMARY)
ax.set_ylabel("Predicted Congestion Level (%)", fontsize=11, color=C_PRIMARY)
ax.set_title("Actual vs Predicted Congestion Level\n(Random Forest — environmental_impact excluded)",
             fontsize=13, fontweight="bold", color=C_PRIMARY, pad=14)
ann = f"R\u00b2 = {r2_rf:.4f}\nMAE = {mae_rf:.4f}\nRMSE = {rmse_rf:.4f}"
ax.text(0.04, 0.96, ann, transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=C_BORDER, alpha=0.9))
ax.legend(fontsize=9)
savefig(fig, "ml_actual_vs_predicted.png")

# ── VIZ 2 : Feature Importance (top 10) ───────────────────────────────────────
def clean(name):
    return " ".join(w.capitalize() for w in name.replace("_"," ").split())

labels = [clean(f) for f in top10.index]
values = top10.values
colors = [C_ACCENT] + [C_BLUE2]*(len(values)-1)

fig, ax = plt.subplots(figsize=(9, 6))
styled_ax(ax, fig)
ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor="none")
for bar, val in zip(ax.patches, values[::-1]):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", ha="left", fontsize=8, color=C_PRIMARY)
ax.set_xlabel("Feature Importance (Gini)", fontsize=11, color=C_PRIMARY)
ax.set_title("Top 10 Feature Importances — Random Forest\n(Environmental Impact excluded — data leakage)",
             fontsize=13, fontweight="bold", color=C_PRIMARY, pad=14)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
patches = [mpatches.Patch(color=C_ACCENT, label="Highest"),
           mpatches.Patch(color=C_BLUE2,  label="Others")]
ax.legend(handles=patches, fontsize=9)
savefig(fig, "ml_feature_importance.png")

# ── VIZ 3 : Model Comparison Bar Chart ────────────────────────────────────────
metrics_compare = {
    "R\u00b2 Score":   [r2_lr,   r2_rf],
    "MAE\n(lower=better)":  [mae_lr,  mae_rf],
    "RMSE\n(lower=better)": [rmse_lr, rmse_rf],
}
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Model Performance Comparison: Linear Regression vs Random Forest",
             fontsize=13, fontweight="bold", color=C_PRIMARY, y=1.01)

bar_labels = ["Linear\nRegression", "Random\nForest"]
bar_colors = [C_WARN, C_SUCCESS]

for ax, (metric, vals) in zip(axes, metrics_compare.items()):
    ax.set_facecolor(C_BG)
    bars = ax.bar(bar_labels, vals, color=bar_colors, width=0.45, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_PRIMARY)
    ax.set_title(metric, fontsize=11, fontweight="bold", color=C_PRIMARY)
    ax.set_ylim(0, max(vals)*1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="y", color=C_BORDER, linewidth=0.7)
    ax.tick_params(axis="x", colors=C_PRIMARY, labelsize=10)

patches = [mpatches.Patch(color=C_WARN,    label="Linear Regression"),
           mpatches.Patch(color=C_SUCCESS,  label="Random Forest")]
fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
savefig(fig, "ml_model_comparison.png")

# ── VIZ 4 : Residual Plot ──────────────────────────────────────────────────────
residuals = y_test.values - yp_rf

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Residual Analysis — Random Forest Regressor",
             fontsize=13, fontweight="bold", color=C_PRIMARY)

# Residuals vs Predicted
ax1 = axes[0]
ax1.set_facecolor(C_BG)
ax1.scatter(yp_rf, residuals, alpha=0.35, s=14, color=C_ACCENT, edgecolors="none")
ax1.axhline(0, color=C_DANGER, linewidth=1.5, linestyle="--")
ax1.set_xlabel("Predicted Congestion Level (%)", fontsize=11, color=C_PRIMARY)
ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=11, color=C_PRIMARY)
ax1.set_title("Residuals vs Predicted", fontsize=11, fontweight="bold", color=C_PRIMARY)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.grid(True, color=C_BORDER, linewidth=0.7)
ax1.tick_params(colors=C_PRIMARY, labelsize=9)

# Residual distribution histogram
ax2 = axes[1]
ax2.set_facecolor(C_BG)
ax2.hist(residuals, bins=40, color=C_ACCENT, edgecolor="white", linewidth=0.4, alpha=0.85)
ax2.axvline(0, color=C_DANGER, linewidth=1.5, linestyle="--", label="Zero Error")
ax2.set_xlabel("Residual Value", fontsize=11, color=C_PRIMARY)
ax2.set_ylabel("Frequency", fontsize=11, color=C_PRIMARY)
ax2.set_title("Residual Distribution", fontsize=11, fontweight="bold", color=C_PRIMARY)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.grid(True, color=C_BORDER, linewidth=0.7)
ax2.tick_params(colors=C_PRIMARY, labelsize=9)
ax2.legend(fontsize=9)

ann2 = f"Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}"
ax2.text(0.97, 0.97, ann2, transform=ax2.transAxes, fontsize=9,
         verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_BORDER, alpha=0.9))

plt.tight_layout()
savefig(fig, "ml_residual_plot.png")

# ── VIZ 5 : Correlation Heatmap ────────────────────────────────────────────────
CORR_FEATURES = [
    "traffic_volume", "average_speed", "travel_time_index",
    "road_capacity_utilization", "incident_reports",
    "public_transport_usage", "traffic_signal_compliance",
    "parking_usage", "pedestrian_and_cyclist_count", "congestion_level",
]
corr_labels = [
    "Traffic Volume", "Avg Speed", "Travel Time Idx",
    "Road Cap. Util.", "Incident Reports",
    "Public Transport", "Signal Compliance",
    "Parking Usage", "Pedestrian Count", "Congestion Level",
]
corr_matrix = df[CORR_FEATURES].corr()

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# Use a diverging colormap
import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn
n    = len(CORR_FEATURES)
im   = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

# Annotate cells
for i in range(n):
    for j in range(n):
        val = corr_matrix.values[i, j]
        txt_color = "white" if abs(val) > 0.6 else C_PRIMARY
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=txt_color, fontweight="bold" if abs(val) > 0.7 else "normal")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr_labels, rotation=40, ha="right", fontsize=9, color=C_PRIMARY)
ax.set_yticklabels(corr_labels, fontsize=9, color=C_PRIMARY)
ax.set_title("Feature Correlation Matrix\n(Numerical Features vs Congestion Level)",
             fontsize=13, fontweight="bold", color=C_PRIMARY, pad=16)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(labelsize=9)
cbar.set_label("Pearson Correlation", fontsize=10, color=C_PRIMARY)
plt.tight_layout()
savefig(fig, "ml_correlation_heatmap.png")

# ── VIZ 6 : Train/Test Split Pie ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
sizes  = [len(X_train), len(X_test)]
labels = [f"Training\n{len(X_train):,} records\n(80%)", f"Testing\n{len(X_test):,} records\n(20%)"]
wedges, texts = ax.pie(sizes, labels=labels, colors=[C_ACCENT, C_SUCCESS],
                       startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
                       textprops=dict(fontsize=11, color=C_PRIMARY, fontweight="bold"))
ax.set_title("Train / Test Split Distribution\n(random_state=42)",
             fontsize=13, fontweight="bold", color=C_PRIMARY, pad=16)
savefig(fig, "ml_train_test_split.png")

print("\nAll 6 ML charts generated successfully.")
print(f"\nFinal metrics (no data leakage):")
print(f"  Linear Regression : MAE={mae_lr:.4f}  RMSE={rmse_lr:.4f}  R2={r2_lr:.4f}")
print(f"  Random Forest     : MAE={mae_rf:.4f}  RMSE={rmse_rf:.4f}  R2={r2_rf:.4f}")
print(f"  Best Model        : {best_model}")
