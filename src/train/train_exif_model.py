import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# LOAD DATA
data_path = Path("data/exif_features.csv")
if not data_path.exists():
    # Fallback: also check old name
    alt = Path("data/metadata.csv")
    if alt.exists():
        data_path = alt
        print(f"⚠️  Using fallback path: {alt}")
    else:
        raise FileNotFoundError(
            f"Data file not found.\n"
            f"Expected: {data_path.absolute()}\n"
            f"Run preprocess_images.py first to generate it."
        )

df = pd.read_csv(data_path)
print(f"Loaded {len(df)} samples")
print(f"Class distribution:\n{df['label'].value_counts()}")

feature_cols = [
    "missing_count", "has_camera_info", "has_software",
    "software_suspicious", "has_timestamp", "timestamp_consistent",
    "exif_total_tags", "has_gps", "has_flash", "has_orientation",
]

# Fill missing columns with 0 (defensive)
for col in feature_cols:
    if col not in df.columns:
        print(f"⚠️  Column '{col}' not found — filling with 0")
        df[col] = 0

X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining: {len(X_train)} | Test: {len(X_test)}")

# HYPERPARAMETER TUNING
param_grid = {
    "n_estimators":     [200, 300],
    "max_depth":        [4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 0.9],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha":        [0, 0.1],
    "reg_lambda":       [1, 2],
}

print(f"\nGrid Search — {np.prod([len(v) for v in param_grid.values()])} combinations (3-fold CV)...")

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

grid = GridSearchCV(xgb, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)

try:
    grid.fit(X_train, y_train)
except KeyboardInterrupt:
    print("\nInterrupted. Reduce param_grid or run overnight.")
    raise SystemExit(1)

print(f"\nBest params : {grid.best_params_}")
print(f"Best CV AUC : {grid.best_score_:.4f}")

# EVALUATE
best_model = grid.best_estimator_
y_pred       = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("XGBoost EXIF Model — Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

importance = pd.DataFrame({
    "feature":    feature_cols,
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# PLOTS
Path("reports").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x="importance", y="feature", palette="viridis")
plt.title("EXIF Feature Importance — TraceFake AI")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/exif_feature_importance.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.title("EXIF Model Confusion Matrix")
plt.savefig("reports/confusion_matrix_exif.png", dpi=150)
plt.close()

# SAVE MODEL
Path("src/models").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, "src/models/exif_xgb.pkl")
print("\n✅ XGBoost model saved → src/models/exif_xgb.pkl")

feat_path = Path("src/models/exif_features.txt")
feat_path.write_text("\n".join(feature_cols) + "\n")
print(f"✅ Feature names saved → {feat_path}")