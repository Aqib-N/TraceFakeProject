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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

# Config
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


# Load Data 
if not METADATA_CSV.exists():
    raise FileNotFoundError(f"Data file not found at {METADATA_CSV.absolute()}")

df = pd.read_csv(METADATA_CSV)
print(f"Loaded {len(df)} samples")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Ensure all feature columns exist; fill missing with 0
for col in EXIF_FEATURE_COLS:
    if col not in df.columns:
        print(f"Warning: '{col}' not found — initialising with zeros")
        df[col] = 0

X = df[EXIF_FEATURE_COLS]   # canonical column order — matches inference
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining: {len(X_train)} | Test: {len(X_test)}")

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples    : {len(X_test)}")


# Hyperparameter Search 
param_dist = {
    "n_estimators":     [100, 200, 300, 400],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 0.5, 1.0],
    "reg_lambda":       [1, 2, 5],
    "min_child_weight": [1, 3, 5],
}

total_combinations = np.prod([len(v) for v in param_dist.values()])
print(f"\nTotal parameter space: {total_combinations:,} combinations")
print("Running RandomizedSearchCV with n_iter=30 (vs 384 in original GridSearch)")

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)


search = RandomizedSearchCV(
    xgb,
    param_dist,
    n_iter=30,
    cv=5,                  
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

try:
    search.fit(X_train, y_train)
except KeyboardInterrupt:
    print("\nSearch interrupted — saving best model found so far")

print(f"\nBest parameters : {search.best_params_}")
print(f"Best CV AUC     : {search.best_score_:.4f}")


# Evaluation 
best_model     = search.best_estimator_
y_pred         = best_model.predict(X_test)
y_pred_proba   = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("XGBoost EXIF Model — Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
importance = (
    pd.DataFrame({
        "feature":    EXIF_FEATURE_COLS,
        "importance": best_model.feature_importances_,
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\nFeature Importance:")
print(importance.to_string(index=False))

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x="importance", y="feature", palette="viridis")
plt.title("EXIF Feature Importance — TraceFake AI")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(REPORT_DIR / "exif_feature_importance.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"])
plt.title("EXIF Model Confusion Matrix — TraceFake AI")
plt.savefig(REPORT_DIR / "confusion_matrix_exif.png", dpi=150)
plt.close()


# Save Model & Metadata 
joblib.dump(best_model, MODEL_DIR / "exif_xgb.pkl")
print(f"\n✅ XGBoost model saved to {MODEL_DIR / 'exif_xgb.pkl'}")

feature_meta = {
    "feature_cols":  EXIF_FEATURE_COLS,
    "best_params":   search.best_params_,
    "best_cv_auc":   round(search.best_score_, 4),
    "test_auc":      round(roc_auc_score(y_test, y_pred_proba), 4),
    "n_features":    len(EXIF_FEATURE_COLS),
}
with open(MODEL_DIR / "exif_model_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
print(f"✅ Model metadata saved to {MODEL_DIR / 'exif_model_meta.json'}")
