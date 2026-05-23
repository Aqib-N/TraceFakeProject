"""
TraceFake AI — Threshold Tuning

Finds the best decision threshold after CNN training.
Run this ONCE after train_cnn_v2.py completes.

Current problem:
  threshold=0.50 → precision=0.999, recall=0.38 (misses 62% of fakes)

Goal: find threshold where both precision AND recall are above 0.80
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import DATA_DIR, MODEL_DIR, REPORT_DIR, IMG_SIZE, BATCH_SIZE
except ImportError:
    DATA_DIR   = Path("data/processed")
    MODEL_DIR  = Path("src/models")
    REPORT_DIR = Path("reports")
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32

# ── Load model ────────────────────────────────────────────────────────────────
cnn_path = MODEL_DIR / "cnn_best.keras"
if not cnn_path.exists():
    cnn_path = MODEL_DIR / "cnn.keras"

print(f"Loading model from {cnn_path}...")
model = tf.keras.models.load_model(str(cnn_path))
print("✅ Model loaded")

# ── Get validation predictions ────────────────────────────────────────────────
val_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
).flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=["fake", "real"],
    shuffle=False,
    seed=42,
)

print(f"Running predictions on {val_gen.samples} validation images...")
val_steps  = int(np.ceil(val_gen.samples / BATCH_SIZE))
y_true     = val_gen.classes
y_proba    = model.predict(val_gen, steps=val_steps, verbose=1).flatten()
y_proba    = y_proba[:len(y_true)]

print(f"\nAUC: {roc_auc_score(y_true, y_proba):.4f}")

# ── Sweep thresholds ──────────────────────────────────────────────────────────
thresholds  = np.arange(0.10, 0.90, 0.01)
results     = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    # precision/recall for FAKE class (label=0, so flip)
    y_pred_fake = 1 - y_pred
    y_true_fake = 1 - y_true
    prec = precision_score(y_true_fake, y_pred_fake, zero_division=0)
    rec  = recall_score(y_true_fake,    y_pred_fake, zero_division=0)
    f1   = f1_score(y_true_fake,        y_pred_fake, zero_division=0)
    acc  = (y_pred == y_true).mean()
    results.append({
        "threshold": round(float(t), 2),
        "precision_fake": round(prec, 4),
        "recall_fake":    round(rec,  4),
        "f1_fake":        round(f1,   4),
        "accuracy":       round(acc,  4),
    })

results_df = __import__('pandas').DataFrame(results)

# ── Find best threshold ───────────────────────────────────────────────────────
# Best = highest F1 for FAKE class (balances precision and recall)
best_row = results_df.loc[results_df["f1_fake"].idxmax()]
best_t   = float(best_row["threshold"])

# Also find balanced threshold (recall_fake >= 0.80 AND precision_fake >= 0.80)
balanced = results_df[
    (results_df["recall_fake"]    >= 0.80) &
    (results_df["precision_fake"] >= 0.80)
]
balanced_t = float(balanced.iloc[0]["threshold"]) if len(balanced) > 0 else best_t

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS (FAKE class detection)")
print("=" * 60)
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
print("-" * 55)
for _, row in results_df[results_df["threshold"].isin(
    [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
)].iterrows():
    marker = " ← current" if row["threshold"] == 0.50 else (
             " ← BEST F1" if row["threshold"] == best_t else (
             " ← balanced" if row["threshold"] == balanced_t else ""))
    print(f"{row['threshold']:>10.2f} {row['precision_fake']:>10.4f} "
          f"{row['recall_fake']:>8.4f} {row['f1_fake']:>8.4f} "
          f"{row['accuracy']:>10.4f}{marker}")

print(f"\n✅ Best F1 threshold  : {best_t:.2f}")
print(f"✅ Balanced threshold : {balanced_t:.2f}  (prec≥0.80 AND recall≥0.80)")
print(f"\nRecommendation: use threshold = {balanced_t:.2f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("TraceFake CNN — Threshold Tuning", fontweight="bold")

axes[0].plot(results_df["threshold"], results_df["precision_fake"],
             "b-", label="Precision (FAKE)", linewidth=2)
axes[0].plot(results_df["threshold"], results_df["recall_fake"],
             "r-", label="Recall (FAKE)",    linewidth=2)
axes[0].plot(results_df["threshold"], results_df["f1_fake"],
             "g-", label="F1 (FAKE)",        linewidth=2)
axes[0].axvline(x=0.50,        color="gray",   linestyle="--", alpha=0.5, label="Default (0.50)")
axes[0].axvline(x=balanced_t,  color="orange", linestyle="--", alpha=0.8, label=f"Balanced ({balanced_t:.2f})")
axes[0].axvline(x=best_t,      color="green",  linestyle="--", alpha=0.8, label=f"Best F1 ({best_t:.2f})")
axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(results_df["threshold"], results_df["accuracy"],
             "purple", linewidth=2)
axes[1].axvline(x=balanced_t, color="orange", linestyle="--", alpha=0.8)
axes[1].axvline(x=best_t,     color="green",  linestyle="--", alpha=0.8)
axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy vs Threshold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / "threshold_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Plot saved → {REPORT_DIR / 'threshold_tuning.png'}")

# ── Save best threshold to config ─────────────────────────────────────────────
threshold_cfg = MODEL_DIR / "threshold.json"
with open(threshold_cfg, "w") as f:
    json.dump({
        "threshold":          balanced_t,
        "best_f1_threshold":  best_t,
        "default_threshold":  0.50,
    }, f, indent=2)
print(f"✅ Threshold saved → {threshold_cfg}")
print("\npredict_system.py will auto-load this threshold on next run.")
