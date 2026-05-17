import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Config 
try:
    from config import (
        IMG_SIZE, BATCH_SIZE, DATA_DIR, MODEL_DIR, REPORT_DIR,
        EPOCHS_PHASE1, EPOCHS_PHASE2, INITIAL_LR,
    )
    TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2
except ImportError:
    IMG_SIZE      = (224, 224)
    BATCH_SIZE    = 32
    EPOCHS_PHASE1 = 10
    EPOCHS_PHASE2 = 15
    TOTAL_EPOCHS  = 25
    INITIAL_LR    = 1e-3
    DATA_DIR      = Path("data/processed")
    MODEL_DIR     = Path("src/models")
    REPORT_DIR    = Path("reports")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Data Pipeline 
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=["fake", "real"],
    shuffle=True,
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=["fake", "real"],
    shuffle=False,
    seed=42,
)

print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Class indices     : {train_gen.class_indices}")

steps_per_epoch = int(np.ceil(train_gen.samples  / BATCH_SIZE))
val_steps       = int(np.ceil(val_gen.samples    / BATCH_SIZE))


# 2. Model Architecture 
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
)
base.trainable = False  # Phase 1: frozen backbone

x      = base.output
x      = layers.GlobalAveragePooling2D()(x)
x      = layers.BatchNormalization()(x)
x      = layers.Dense(256, activation="swish")(x)
x      = layers.Dropout(0.5)(x)
x      = layers.Dense(128, activation="swish")(x)
x      = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)

outputs = layers.Dense(1, activation="sigmoid")(x)


# 3. Phase 1: Train Head (Frozen Backbone) 
model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("\n" + "=" * 60)
print("PHASE 1: Training classifier head")
print("=" * 60)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_auc", patience=5,
            restore_best_weights=True, mode="max",
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_phase1.keras"),
            monitor="val_auc", save_best_only=True, mode="max",
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1,
        ),
    ],
)

phase1_epochs_run = len(history1.history["accuracy"])
print(f"\nPhase 1 completed: {phase1_epochs_run} epochs")


# 4. Phase 2: Fine-Tune (Unfreeze top 30% of backbone) 
base.trainable = True
fine_tune_at   = len(base.layers) - int(len(base.layers) * 0.30)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

fine_tune_lr = INITIAL_LR / 10  # 1e-4

# The original lambda scheduler used history1.epoch[-1] as an offset,
# which breaks if Phase 1 early-stops, causing the LR to start mid-cycle.
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=fine_tune_lr,
    decay_steps=EPOCHS_PHASE2 * steps_per_epoch,
    alpha=1e-6,  # minimum LR floor
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("\n" + "=" * 60)
print(f"PHASE 2: Fine-tuning top 30% backbone layers")
print(f"Unfrozen from layer {fine_tune_at} of {len(base.layers)}")
print("=" * 60)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    initial_epoch=phase1_epochs_run,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_auc", patience=7,
            restore_best_weights=True, mode="max",
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_best.keras"),
            monitor="val_auc", save_best_only=True, mode="max",
        ),
    ],
)


# 5. Save Final Model 
model.save(str(MODEL_DIR / "cnn.keras"))
print("\n✅ Final model saved")


# 6. Helper: safe metric extraction

def get_metric(history, base_name: str):
    """
    FIX: Keras appends _1, _2 suffixes to duplicate metric names when
    model.compile() is called a second time for Phase 2. This helper
    finds the right key regardless of suffix.
    """
    for key in history.history:
        if key == base_name or key.startswith(base_name):
            return history.history[key]
    return []


# 7. Training Graphs 

def plot_training_history(h1, h2):
    acc       = get_metric(h1, "accuracy")       + get_metric(h2, "accuracy")
    val_acc   = get_metric(h1, "val_accuracy")   + get_metric(h2, "val_accuracy")
    loss      = get_metric(h1, "loss")            + get_metric(h2, "loss")
    val_loss  = get_metric(h1, "val_loss")        + get_metric(h2, "val_loss")
    prec      = get_metric(h1, "precision")       + get_metric(h2, "precision")
    val_prec  = get_metric(h1, "val_precision")   + get_metric(h2, "val_precision")
    rec       = get_metric(h1, "recall")          + get_metric(h2, "recall")
    val_rec   = get_metric(h1, "val_recall")      + get_metric(h2, "val_recall")
    auc_vals  = get_metric(h1, "auc")             + get_metric(h2, "auc")
    val_auc   = get_metric(h1, "val_auc")         + get_metric(h2, "val_auc")

    epochs_range = range(1, len(acc) + 1)
    phase1_end   = len(h1.history["accuracy"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("TraceFake AI — CNN Training Report", fontsize=14, fontweight="bold")

    metrics_data = [
        (acc,      val_acc,  "Accuracy"),
        (loss,     val_loss, "Loss"),
        (prec,     val_prec, "Precision"),
        (rec,      val_rec,  "Recall"),
        (auc_vals, val_auc,  "AUC"),
    ]

    for idx, (train, val, title) in enumerate(metrics_data):
        ax = axes[idx // 3, idx % 3]
        ax.plot(epochs_range, train, "b-",  label="Train",      linewidth=2)
        ax.plot(epochs_range, val,   "r--", label="Validation",  linewidth=2)
        ax.axvline(x=phase1_end, color="g", linestyle=":", alpha=0.7, label="Fine-tune start")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary panel
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = f"""
  BEST VALIDATION RESULTS
  =======================
  Epoch     : {np.argmax(val_acc) + 1}
  Accuracy  : {max(val_acc):.4f}
  Precision : {max(val_prec):.4f}
  Recall    : {max(val_rec):.4f}
  AUC       : {max(val_auc):.4f}

  Training Config:
  • Backbone  : EfficientNetB0
  • Total Epochs: {len(acc)}
  • Phase 1 (frozen): {phase1_end} epochs
  • Phase 2 (fine-tune): {len(h2.history['accuracy'])} epochs
  • Batch Size: {BATCH_SIZE}
  • Fine-tune LR: {fine_tune_lr:.0e}
    """
    ax.text(
        0.05, 0.5, summary_text, fontsize=9, family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "training_graphs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training graphs saved to {REPORT_DIR / 'training_graphs.png'}")


plot_training_history(history1, history2)


# 8. Sample Predictions 

def plot_sample_predictions(model, val_gen, n_samples: int = 8):
    """
    FIX: bounded loop prevents infinite iteration if val_gen is exhausted.
    """
    val_gen.reset()
    images, labels = [], []
    max_batches = int(np.ceil(n_samples / val_gen.batch_size)) + 2

    for _ in range(max_batches):
        if len(images) >= n_samples:
            break
        try:
            batch_img, batch_lbl = next(val_gen)
        except StopIteration:
            break
        for i in range(len(batch_img)):
            if len(images) >= n_samples:
                break
            images.append(batch_img[i])
            labels.append(batch_lbl[i])

    images = np.array(images[:n_samples])
    labels = np.array(labels[:n_samples])
    preds  = model.predict(images, verbose=0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("TraceFake AI — Sample Test Predictions", fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis("off")
            continue

        true_label = "REAL" if labels[idx] == 1 else "FAKE"
        pred_prob  = preds[idx][0]
        pred_label = "REAL" if pred_prob > 0.5 else "FAKE"
        confidence = pred_prob if pred_label == "REAL" else 1 - pred_prob
        color      = "#00C98A" if pred_label == true_label else "#FF4D6D"

        ax.imshow(images[idx])
        ax.set_title(
            f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1%}",
            color=color, fontweight="bold", fontsize=10,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔍 Sample predictions saved to {REPORT_DIR / 'sample_predictions.png'}")


plot_sample_predictions(model, val_gen)


# 9. Final Evaluation 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# FINAL EVALUATION
val_gen.reset()
y_true       = val_gen.classes
y_pred_proba = model.predict(val_gen, steps=val_steps, verbose=0)
y_pred       = (y_pred_proba > 0.5).astype(int).flatten()[: len(y_true)]

# Summary from training history
all_val_acc  = get_metric(history1, "val_accuracy")  + get_metric(history2, "val_accuracy")
all_val_auc  = get_metric(history1, "val_auc")        + get_metric(history2, "val_auc")
all_val_prec = get_metric(history1, "val_precision")  + get_metric(history2, "val_precision")
all_val_rec  = get_metric(history1, "val_recall")     + get_metric(history2, "val_recall")

print(f"\nBest Validation Accuracy : {max(all_val_acc):.4f}")
print(f"Best Validation AUC      : {max(all_val_auc):.4f}")
print(f"Best Validation Precision: {max(all_val_prec):.4f}")
print(f"Best Validation Recall   : {max(all_val_rec):.4f}")

print("\n" + "=" * 60)
print("FINAL CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=["FAKE", "REAL"])
print(report)

with open(REPORT_DIR / "cnn_classification_report.txt", "w") as f:
    f.write("TraceFake AI — CNN Classification Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write(f"\nBest Val Accuracy : {max(all_val_acc):.4f}\n")
    f.write(f"Best Val AUC      : {max(all_val_auc):.4f}\n")
    f.write(f"Best Val Precision: {max(all_val_prec):.4f}\n")
    f.write(f"Best Val Recall   : {max(all_val_rec):.4f}\n")
    f.write(f"Total Epochs      : {len(all_val_acc)}\n")
    f.write(f"Phase 1 Epochs    : {phase1_epochs_run}\n")
    f.write(f"Phase 2 Epochs    : {len(history2.history['accuracy'])}\n")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"])
plt.title("Confusion Matrix — TraceFake AI CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(REPORT_DIR / "confusion_matrix_cnn.png", dpi=150, bbox_inches="tight")
plt.close()

print("✅ Reports generated")

# TRAINING CURVES (SAVE GRAPH)
history = {}

for k in history1.history.keys():
    history[k] = history1.history[k] + history2.history[k]

plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.tight_layout()
plt.savefig(REPORT_DIR / "training_curves.png", dpi=150)
plt.close()

print("✅ Training curves saved → reports/training_curves.png")