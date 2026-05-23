"""
TraceFake AI — EXIF XGBoost Training (ArtiFact / Injected EXIF)

Leaky features removed:
  - has_camera_info  (+1.000 corr) — real always has camera, fake never
  - exif_total_tags  (+0.975 corr) — directly counts tags = reveals class

Remaining 10 features all have genuine forensic signal (0.23–0.94 corr)
without being perfect predictors.
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

try:
    from config import METADATA_CSV, MODEL_DIR, REPORT_DIR, EXIF_FEATURE_COLS
except ImportError:
    METADATA_CSV = Path("data/metadata.csv")
    MODEL_DIR    = Path("src/models")
    REPORT_DIR   = Path("reports")
    EXIF_FEATURE_COLS = [
        "missing_count", "has_camera_info", "has_software",
        "software_suspicious", "has_timestamp", "timestamp_consistent",
        "timestamp_plausible", "timestamp_future",
        "exif_total_tags", "has_gps", "has_flash", "has_orientation",
    ]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Remove leaky features ─────────────────────────────────────────────────────
# has_camera_info: corr=+1.000 (perfect predictor — not a forensic signal)
# exif_total_tags: corr=+0.975 (counts all tags — same problem as missing_count)
LEAKY = {"has_camera_info", "exif_total_tags"}
TRAIN_COLS = [c for c in EXIF_FEATURE_COLS if c not in LEAKY]

print("=" * 55)
print("EXIF MODEL TRAINING")
print("=" * 55)
print(f"Removed leaky : {LEAKY}")
print(f"Training on   : {len(TRAIN_COLS)} features")
print(f"Features      : {TRAIN_COLS}")


# ── Load data ─────────────────────────────────────────────────────────────────
if not METADATA_CSV.exists():
    raise FileNotFoundError(f"metadata.csv not found at {METADATA_CSV}")

df = pd.read_csv(METADATA_CSV)
print(f"\nLoaded {len(df):,} samples")
print(f"Class distribution:\n{df['label'].value_counts()}")

for col in TRAIN_COLS:
    if col not in df.columns:
        print(f"Warning: '{col}' missing — filling with 0")
        df[col] = 0

X = df[TRAIN_COLS]
y = df["label"]

# ── Correlation check ─────────────────────────────────────────────────────────
print("\nFeature correlations after removing leaky:")
for col in TRAIN_COLS:
    corr = df[col].corr(df['label'])
    flag = " ← ⚠️  still leaky" if abs(corr) > 0.95 else (
           " ← strong signal"   if abs(corr) > 0.60 else
           " ← medium signal"   if abs(corr) > 0.30 else
           " ← weak signal")
    print(f"  {col:<28}: {corr:+.3f}{flag}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 3)
print(f"scale_pos_weight: {spw}")


# ── Hyperparameter search ─────────────────────────────────────────────────────
param_dist = {
    "n_estimators":     [100, 200, 300, 400],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 0.5, 1.0],
    "reg_lambda":       [1, 2, 5],
    "min_child_weight": [1, 3, 5],
}

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=spw,
    random_state=42,
    n_jobs=-1,
)

search = RandomizedSearchCV(
    xgb, param_dist,
    n_iter=40, cv=5,
    scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}")

# Warn if still suspiciously perfect
if search.best_score_ > 0.97:
    print("\n⚠️  CV AUC still very high — check remaining features for leakage")
    print("   Acceptable range for injected EXIF: 0.80 – 0.95")


# ── Evaluation ────────────────────────────────────────────────────────────────
best      = search.best_estimator_
y_pred    = best.predict(X_test)
y_proba   = best.predict_proba(X_test)[:, 1]
test_auc  = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print(f"Test AUC: {test_auc:.4f}")

# Feature importance
importance = (
    pd.DataFrame({"feature": TRAIN_COLS,
                  "importance": best.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\nFeature Importance:")
print(importance.to_string(index=False))

# Plots
plt.figure(figsize=(10, 5))
sns.barplot(data=importance, x="importance", y="feature", palette="viridis")
plt.title("EXIF Feature Importance — TraceFake AI")
plt.tight_layout()
plt.savefig(REPORT_DIR / "exif_feature_importance.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.title("EXIF Model Confusion Matrix")
plt.savefig(REPORT_DIR / "confusion_matrix_exif.png", dpi=150)
plt.close()


# ── Save model + metadata ─────────────────────────────────────────────────────
joblib.dump(best, MODEL_DIR / "exif_xgb.pkl")
print(f"\n✅ Model saved → {MODEL_DIR / 'exif_xgb.pkl'}")

meta = {
    "feature_cols":   TRAIN_COLS,        # non-leaky list — used by predict_system
    "removed_leaky":  list(LEAKY),
    "best_params":    search.best_params_,
    "best_cv_auc":    round(search.best_score_, 4),
    "test_auc":       round(test_auc, 4),
    "n_features":     len(TRAIN_COLS),
}
with open(MODEL_DIR / "exif_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ Metadata saved → {MODEL_DIR / 'exif_model_meta.json'}")
print(f"\nExpected AUC range for injected EXIF: 0.80 – 0.95")
print(f"Your test AUC: {test_auc:.4f} {'✅' if 0.75 < test_auc < 0.97 else '⚠️  check above'}")