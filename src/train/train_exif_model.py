"""
TraceFake AI — EXIF Metadata XGBoost Training
Research shows XGBoost achieves 96-98% with proper features [^16^][^18^]
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD ENRICHED DATA
# =========================
df = pd.read_csv("data/metadata.csv")

# Enhanced feature set based on research [^18^]
feature_cols = [
    "missing_count", "has_camera_info", "has_software",
    "software_suspicious", "has_timestamp", "timestamp_consistent",
    "exif_total_tags", "has_gps", "has_flash", "has_orientation"
]

# Ensure all columns exist, fill missing with 0
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# =========================
# HYPERPARAMETER TUNING
# =========================
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2]
}

print("\nRunning Grid Search (this may take a few minutes)...")

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    xgb, param_grid, 
    cv=3, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.4f}")

# =========================
# FINAL MODEL
# =========================
best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("XGBoost EXIF Model — Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
plt.title('EXIF Feature Importance — TraceFake AI')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig("reports/exif_feature_importance.png", dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['FAKE', 'REAL'],
            yticklabels=['FAKE', 'REAL'])
plt.title('EXIF Model Confusion Matrix')
plt.savefig("reports/confusion_matrix_exif.png", dpi=150)
plt.close()

# Save model
Path("src/models").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, "src/models/exif_xgb.pkl")
print("\n✅ XGBoost EXIF model saved to src/models/exif_xgb.pkl")