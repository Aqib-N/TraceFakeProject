"""
train_cnn.py
TraceFake AI — Two-Phase CNN Training (EfficientNetB0)
"""

import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_PHASE1 = 10   # frozen backbone — head only
EPOCHS_PHASE2 = 15   # fine-tune top 30% of backbone

DATA_DIR   = Path("data/processed")
MODEL_DIR  = Path("src/models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. DATA PIPELINE
# FIX 1: val_datagen must NOT have augmentation, but it was missing
#         validation_split — so it was reading the full dataset instead of
#         the held-out 20%, making val metrics unreliable.
# =============================================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,   # FIX 1: must match train_datagen split
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=["fake", "real"],   # fake=0, real=1
    shuffle=True,
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",        # FIX 1: was missing subset='validation'
    classes=["fake", "real"],
    shuffle=False,
    seed=42,
)

print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Class indices     : {train_gen.class_indices}")  # FIX 2: log so you can verify fake=0 real=1

# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================
base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False  # Phase 1: frozen

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="swish")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="swish")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(base.input, output)

print(f"\nTotal params     : {model.count_params():,}")
print(f"Trainable params : {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

# =============================================================================
# 3. PHASE 1 — TRAIN HEAD (frozen backbone)
# =============================================================================
initial_lr = 1e-3

model.compile(
    optimizer=optimizers.Adam(learning_rate=initial_lr),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("\n" + "=" * 60)
print("PHASE 1: Training head (backbone frozen)")
print("=" * 60)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_phase1.keras"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        # FIX 3: log LR each epoch so you can see ReduceLROnPlateau working
        callbacks.CSVLogger(str(REPORT_DIR / "phase1_log.csv")),
    ],
)

# =============================================================================
# 4. PHASE 2 — FINE-TUNE (unfreeze top 30% of backbone)
# =============================================================================
base.trainable = True
fine_tune_at = len(base.layers) - int(len(base.layers) * 0.30)

for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

fine_tune_lr = initial_lr / 10   # 1e-4

# FIX 4: must recompile AFTER changing trainable flags
model.compile(
    optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("\n" + "=" * 60)
print(f"PHASE 2: Fine-tuning top 30% ({len(base.layers) - fine_tune_at} layers)")
print(f"Unfrozen from layer {fine_tune_at} of {len(base.layers)}")
print("=" * 60)

# FIX 5: LearningRateScheduler epoch index starts from initial_epoch,
#         so offset calculation was wrong and LR could go negative.
#         Use a closure that captures the correct start epoch.
phase1_end_epoch = history1.epoch[-1] + 1

def cosine_decay(epoch):
    """Cosine decay relative to phase-2 start."""
    relative = epoch - phase1_end_epoch
    relative = max(0, relative)   # guard against negative
    return fine_tune_lr * 0.5 * (1 + np.cos(np.pi * relative / EPOCHS_PHASE2))

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=phase1_end_epoch + EPOCHS_PHASE2,
    initial_epoch=phase1_end_epoch,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=7,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_best.keras"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.LearningRateScheduler(cosine_decay, verbose=0),
        callbacks.CSVLogger(str(REPORT_DIR / "phase2_log.csv")),
    ],
)

# =============================================================================
# 5. SAVE FINAL MODEL
# =============================================================================
model.save(str(MODEL_DIR / "cnn.keras"))
print(f"\n✅ Final model saved → {MODEL_DIR / 'cnn.keras'}")

# =============================================================================
# 6. TRAINING GRAPHS
# =============================================================================
def plot_training_history(h1, h2):
    # FIX 6: guard against EarlyStopping cutting phase2 short —
    #         use .get() so missing keys don't crash
    def merge(key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    acc       = merge("accuracy");       val_acc       = merge("val_accuracy")
    loss      = merge("loss");           val_loss      = merge("val_loss")
    precision = merge("precision");      val_precision = merge("val_precision")
    recall    = merge("recall");         val_recall    = merge("val_recall")
    auc       = merge("auc");            val_auc       = merge("val_auc")

    n          = len(acc)
    phase1_end = len(h1.history.get("accuracy", []))
    epochs_x   = range(1, n + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("TraceFake AI — CNN Training Report", fontsize=14, fontweight="bold")

    for idx, (train_m, val_m, title) in enumerate([
        (acc,       val_acc,       "Accuracy"),
        (loss,      val_loss,      "Loss"),
        (precision, val_precision, "Precision"),
        (recall,    val_recall,    "Recall"),
        (auc,       val_auc,       "AUC"),
    ]):
        ax = axes[idx // 3][idx % 3]
        ax.plot(epochs_x, train_m, "b-",  label="Train",      linewidth=2)
        ax.plot(epochs_x, val_m,   "r--", label="Validation",  linewidth=2)
        if phase1_end < n:
            ax.axvline(x=phase1_end, color="g", linestyle=":", alpha=0.7, label="Fine-tune start")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary panel
    ax = axes[1][2]
    ax.axis("off")
    best_epoch = int(np.argmax(val_acc)) + 1
    summary = (
        "BEST VALIDATION RESULTS\n"
        "=======================\n"
        f"Epoch     : {best_epoch}\n"
        f"Accuracy  : {max(val_acc):.4f}\n"
        f"Precision : {max(val_precision):.4f}\n"
        f"Recall    : {max(val_recall):.4f}\n"
        f"AUC       : {max(val_auc):.4f}\n\n"
        "Training Config:\n"
        f"  Backbone   : EfficientNetB0\n"
        f"  Epochs     : {n}\n"
        f"  Phase 1    : {phase1_end}\n"
        f"  Phase 2    : {n - phase1_end}\n"
        f"  Batch size : {BATCH_SIZE}\n"
        f"  Final LR   : {fine_tune_lr:.0e}"
    )
    ax.text(
        0.05, 0.5, summary,
        fontsize=10, family="monospace", verticalalignment="center",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    out = REPORT_DIR / "training_graphs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training graphs → {out}")


plot_training_history(history1, history2)


# =============================================================================
# 7. SAMPLE PREDICTIONS
# =============================================================================
def plot_sample_predictions(model, val_gen, n_samples: int = 8):
    val_gen.reset()
    images, labels = [], []

    # FIX 7: use iter() instead of calling next() in a while loop —
    #         avoids infinite loop if val set has fewer than n_samples images
    for batch_img, batch_lbl in val_gen:
        for i in range(len(batch_img)):
            images.append(batch_img[i])
            labels.append(batch_lbl[i])
            if len(images) >= n_samples:
                break
        if len(images) >= n_samples:
            break

    if not images:
        print("⚠️  No validation images found for sample predictions.")
        return

    images = np.array(images[:n_samples])
    labels = np.array(labels[:n_samples])
    preds  = model.predict(images, verbose=0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("TraceFake AI — Sample Predictions", fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis("off")
            continue

        true_label  = "REAL" if labels[idx] == 1 else "FAKE"
        pred_prob   = float(preds[idx][0])
        pred_label  = "REAL" if pred_prob > 0.5 else "FAKE"
        confidence  = pred_prob if pred_label == "REAL" else 1.0 - pred_prob
        color       = "#00C98A" if pred_label == true_label else "#FF4D6D"

        ax.imshow(images[idx])
        ax.set_title(
            f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1%}",
            color=color, fontweight="bold", fontsize=9,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    out = REPORT_DIR / "sample_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔍 Sample predictions → {out}")


plot_sample_predictions(model, val_gen)


# =============================================================================
# 8. FINAL EVALUATION REPORT
# =============================================================================
# FIX 8: imports moved to top of file (were buried inside section 8)
all_val_acc       = history1.history["val_accuracy"]  + history2.history.get("val_accuracy",  [])
all_val_auc       = history1.history["val_auc"]       + history2.history.get("val_auc",        [])
all_val_precision = history1.history["val_precision"] + history2.history.get("val_precision",  [])
all_val_recall    = history1.history["val_recall"]    + history2.history.get("val_recall",     [])

print(f"\nBest Val Accuracy  : {max(all_val_acc):.4f}")
print(f"Best Val AUC       : {max(all_val_auc):.4f}")
print(f"Best Val Precision : {max(all_val_precision):.4f}")
print(f"Best Val Recall    : {max(all_val_recall):.4f}")

# FIX 9: reset val_gen before prediction — otherwise you get partial batches
#         from wherever the generator stopped after sample_predictions
val_gen.reset()
y_true  = val_gen.classes
y_pred  = (model.predict(val_gen, verbose=1) > 0.5).astype(int).flatten()

# FIX 10: length mismatch guard — EarlyStopping can leave generator mid-batch
if len(y_pred) != len(y_true):
    min_len = min(len(y_true), len(y_pred))
    print(f"⚠️  Length mismatch — trimming to {min_len} samples")
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

print("\n" + "=" * 60)
print("FINAL CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=["FAKE", "REAL"])
print(report)

report_path = REPORT_DIR / "cnn_classification_report.txt"
with open(report_path, "w") as f:
    f.write("TraceFake AI — CNN Classification Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write("\n\nTraining Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Best Val Accuracy  : {max(all_val_acc):.4f}\n")
    f.write(f"Best Val AUC       : {max(all_val_auc):.4f}\n")
    f.write(f"Best Val Precision : {max(all_val_precision):.4f}\n")
    f.write(f"Best Val Recall    : {max(all_val_recall):.4f}\n")
    f.write(f"Total Epochs       : {len(all_val_acc)}\n")
    f.write(f"Phase 1 Epochs     : {len(history1.history['accuracy'])}\n")
    f.write(f"Phase 2 Epochs     : {len(history2.history.get('accuracy', []))}\n")
print(f"📝 Classification report → {report_path}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"],
)
plt.title("Confusion Matrix — TraceFake AI CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_path = REPORT_DIR / "confusion_matrix_cnn.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Confusion matrix → {cm_path}")

print(f"\n✅ All reports saved to {REPORT_DIR}/")