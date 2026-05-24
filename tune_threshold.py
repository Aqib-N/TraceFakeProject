import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import json, pandas as pd

sys.path.insert(0, "/kaggle/working/TraceFakeProject")
from config import DATA_DIR, MODEL_DIR, IMG_SIZE, BATCH_SIZE

print(f"TF version: {tf.__version__}")

cnn_path = MODEL_DIR / "cnn_best.keras"
if not cnn_path.exists():
    cnn_path = MODEL_DIR / "cnn.keras"

print(f"Loading {cnn_path}...")

# FIX: compile=False skips loss deserialization entirely
# We only need the model for inference (predict), not training
model = tf.keras.models.load_model(str(cnn_path), compile=False)
print("✅ Loaded")

val_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2).flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation",
    classes=["fake", "real"], shuffle=False, seed=42,
)

val_steps = int(np.ceil(val_gen.samples / BATCH_SIZE))
y_true    = val_gen.classes
y_proba   = model.predict(val_gen, steps=val_steps, verbose=1).flatten()[:len(y_true)]

print(f"\nAUC: {roc_auc_score(y_true, y_proba):.4f}")

results = []
for t in np.arange(0.10, 0.90, 0.01):
    y_pred      = (y_proba >= t).astype(int)
    y_pred_fake = 1 - y_pred
    y_true_fake = 1 - y_true
    results.append({
        "threshold":      round(float(t), 2),
        "precision_fake": round(precision_score(y_true_fake, y_pred_fake, zero_division=0), 4),
        "recall_fake":    round(recall_score(y_true_fake,    y_pred_fake, zero_division=0), 4),
        "f1_fake":        round(f1_score(y_true_fake,        y_pred_fake, zero_division=0), 4),
        "accuracy":       round((y_pred == y_true).mean(), 4),
    })

df     = pd.DataFrame(results)
best_t = float(df.loc[df["f1_fake"].idxmax(), "threshold"])
bal    = df[(df["recall_fake"] >= 0.80) & (df["precision_fake"] >= 0.80)]
bal_t  = float(bal.iloc[0]["threshold"]) if len(bal) > 0 else best_t

print(f"\n{'Thresh':>7} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8}")
for _, r in df[df["threshold"].isin([0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60])].iterrows():
    m = " ← default" if r.threshold==0.50 else (" ← BEST F1" if r.threshold==best_t else "")
    print(f"{r.threshold:>7.2f} {r.precision_fake:>10.4f} {r.recall_fake:>8.4f} {r.f1_fake:>8.4f} {r.accuracy:>8.4f}{m}")

print(f"\n✅ Best F1 threshold  : {best_t:.2f}")
print(f"✅ Balanced threshold : {bal_t:.2f}")

with open(MODEL_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": bal_t, "best_f1_threshold": best_t, "default": 0.50}, f, indent=2)
print(f"✅ Saved → {MODEL_DIR / 'threshold.json'}")
