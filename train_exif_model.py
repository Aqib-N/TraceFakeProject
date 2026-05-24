"""
TraceFake AI — EXIF XGBoost Training (v3 — auto-removes leaky/useless features)
"""

import pandas as pd
import numpy as np
import joblib, json, warnings
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

# ── Load data ─────────────────────────────────────────────────────────────────
if not METADATA_CSV.exists():
    raise FileNotFoundError(f"metadata.csv not found: {METADATA_CSV}")

df = pd.read_csv(METADATA_CSV)
print(f"Loaded {len(df):,} samples")
print(f"Class balance: {df['label'].value_counts().to_dict()}")

# ── Auto-detect and remove leaky / zero-variance features ────────────────────
print("\nAuto-detecting features to remove:")
LEAKY_CORR_THRESHOLD = 0.95
remove = set()

for col in EXIF_FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0
        continue
    std  = df[col].std()
    if std == 0:
        print(f"  REMOVE {col:<28} — zero variance (useless)")
        remove.add(col)
        continue
    corr = abs(df[col].corr(df['label']))
    if corr > LEAKY_CORR_THRESHOLD:
        print(f"  REMOVE {col:<28} — corr={corr:.3f} (leaky)")
        remove.add(col)
    else:
        print(f"  KEEP   {col:<28} — corr={corr:.3f}")

TRAIN_COLS = [c for c in EXIF_FEATURE_COLS if c not in remove]
print(f"\nTraining on {len(TRAIN_COLS)} features: {TRAIN_COLS}")

if len(TRAIN_COLS) < 3:
    print("❌ Too few features remaining — re-run inject_exif_v3.py")
    sys.exit(1)

X = df[TRAIN_COLS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 3)

# ── Train ─────────────────────────────────────────────────────────────────────
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

search = RandomizedSearchCV(
    XGBClassifier(objective="binary:logistic", eval_metric="auc",
                  scale_pos_weight=spw, random_state=42, n_jobs=-1),
    param_dist, n_iter=40, cv=5,
    scoring="roc_auc", n_jobs=-1, random_state=42, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest CV AUC : {search.best_score_:.4f}")
print(f"Best params : {search.best_params_}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
best     = search.best_estimator_
y_pred   = best.predict(X_test)
y_proba  = best.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 50)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print(f"Test AUC: {test_auc:.4f}")

if test_auc > 0.97:
    print("⚠️  AUC still very high — injected EXIF may still be too deterministic")
elif test_auc > 0.75:
    print("✅ AUC in healthy range — EXIF model is learning real patterns")
else:
    print("⚠️  Low AUC — EXIF signal may be too weak, increase fusion weight for CNN")

importance = (
    pd.DataFrame({"feature": TRAIN_COLS,
                  "importance": best.feature_importances_})
    .sort_values("importance", ascending=False)
)
print("\nFeature Importance:")
print(importance.to_string(index=False))

# Plots
plt.figure(figsize=(9, 4))
sns.barplot(data=importance, x="importance", y="feature", palette="viridis")
plt.title("EXIF Feature Importance — TraceFake AI")
plt.tight_layout()
plt.savefig(REPORT_DIR / "exif_feature_importance.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"])
plt.title("EXIF Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix_exif.png", dpi=150)
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(best, MODEL_DIR / "exif_xgb.pkl")
meta = {
    "feature_cols":  TRAIN_COLS,
    "removed":       list(remove),
    "best_params":   search.best_params_,
    "best_cv_auc":   round(search.best_score_, 4),
    "test_auc":      round(test_auc, 4),
    "n_features":    len(TRAIN_COLS),
}
with open(MODEL_DIR / "exif_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Model → {MODEL_DIR / 'exif_xgb.pkl'}")
print(f"✅ Meta  → {MODEL_DIR / 'exif_model_meta.json'}")