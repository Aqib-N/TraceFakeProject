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
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
warnings.filterwarnings("ignore")

try:
    from config import METADATA_CSV, MODEL_DIR, REPORT_DIR, EXIF_FEATURE_COLS
except ImportError:
    METADATA_CSV    = Path("data/metadata.csv")
    MODEL_DIR       = Path("src/models")
    REPORT_DIR      = Path("reports")
    EXIF_FEATURE_COLS = [
        "missing_count", "has_camera_info", "has_software",
        "software_suspicious", "has_timestamp", "timestamp_consistent",
        "timestamp_plausible", "timestamp_future",
        "exif_total_tags", "has_gps", "has_flash", "has_orientation",
    ]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# FIX: these two features directly reveal whether EXIF exists at all.
# They are perfectly correlated with fake/real in most datasets →
# remove them so the model learns actual forensic patterns.
LEAKY_FEATURES = {"missing_count", "exif_total_tags"}
TRAIN_COLS = [c for c in EXIF_FEATURE_COLS if c not in LEAKY_FEATURES]
print(f"Removed leaky features: {LEAKY_FEATURES}")
print(f"Training on {len(TRAIN_COLS)} features: {TRAIN_COLS}")


# ── Load Data ─────────────────────────────────────────────────────────────────
if not METADATA_CSV.exists():
    raise FileNotFoundError(f"Data file not found at {METADATA_CSV.absolute()}")

df = pd.read_csv(METADATA_CSV)
print(f"\nLoaded {len(df)} samples")
print(f"Class distribution:\n{df['label'].value_counts()}")

for col in TRAIN_COLS:
    if col not in df.columns:
        print(f"Warning: '{col}' missing — filling with 0")
        df[col] = 0

X = df[TRAIN_COLS]
y = df["label"]

# FIX: check for remaining perfect predictors before training
print("\nFeature correlation with label:")
for col in TRAIN_COLS:
    corr = df[col].corr(df["label"])
    flag = " ← SUSPICIOUS (leaky?)" if abs(corr) > 0.9 else ""
    print(f"  {col:30s}: {corr:+.3f}{flag}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining: {len(X_train)} | Test: {len(X_test)}")

# class balance ratio for scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 3)
print(f"scale_pos_weight = {spw}  (neg={neg}, pos={pos})")


# ── Hyperparameter Search ─────────────────────────────────────────────────────
param_dist = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 2, 5],
    "min_child_weight": [1, 3, 5],
}

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=spw,           # FIX: handles class imbalance
    random_state=42,
    n_jobs=-1,
)

search = RandomizedSearchCV(
    xgb, param_dist,
    n_iter=30, cv=5,
    scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest parameters : {search.best_params_}")
print(f"Best CV AUC     : {search.best_score_:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
best_model   = search.best_estimator_
y_pred       = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_auc     = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 50)
print("XGBoost EXIF Model — Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print(f"Test AUC: {test_auc:.4f}")

# Warn if still suspiciously perfect
if test_auc > 0.98:
    print("\n⚠️  WARNING: AUC still near 1.0 after removing leaky features.")
    print("   Your dataset likely still has a structural difference between")
    print("   fake and real EXIF (e.g. all fakes lack timestamps).")
    print("   Consider regenerating metadata.csv with EXIF-stripped real images,")
    print("   or use the EXIF model only as a soft signal (low fusion weight).")

importance = (
    pd.DataFrame({"feature": TRAIN_COLS,
                  "importance": best_model.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\nFeature Importance:")
print(importance.to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=importance, x="importance", y="feature", palette="viridis")
plt.title("EXIF Feature Importance — TraceFake AI (leaky features removed)")
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


# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_DIR / "exif_xgb.pkl")
print(f"\n✅ XGBoost model saved → {MODEL_DIR / 'exif_xgb.pkl'}")

meta = {
    "feature_cols":  TRAIN_COLS,           # IMPORTANT: save non-leaky list
    "removed_leaky": list(LEAKY_FEATURES),
    "best_params":   search.best_params_,
    "best_cv_auc":   round(search.best_score_, 4),
    "test_auc":      round(test_auc, 4),
    "n_features":    len(TRAIN_COLS),
}
with open(MODEL_DIR / "exif_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ Metadata saved → {MODEL_DIR / 'exif_model_meta.json'}")