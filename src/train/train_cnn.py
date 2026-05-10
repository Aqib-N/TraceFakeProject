"""
train_cnn.py
TraceFake AI — Improved CNN Training Pipeline
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
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15

INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
THRESHOLD = 0.35

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("src/models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA PIPELINE
# =============================================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,
)

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

# =============================================================================
# MODEL
# =============================================================================
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="swish")(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(128, activation="swish")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

print(f"\nTotal params     : {model.count_params():,}")
print(f"Trainable params : {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

# =============================================================================
# LOSS FUNCTION
# =============================================================================
loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
    gamma=2,
    apply_class_balancing=True,
)

metrics = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

# =============================================================================
# PHASE 1
# =============================================================================
model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss=loss_fn,
    metrics=metrics,
)

print("\n" + "=" * 60)
print("PHASE 1: Training classifier head")
print("=" * 60)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    class_weight={0: 1.0, 1: 1.5},
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
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ],
)

# =============================================================================
# PHASE 2
# =============================================================================
base.trainable = True

fine_tune_at = len(base.layers) - 30

for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=loss_fn,
    metrics=metrics,
)

print("\n" + "=" * 60)
print(f"PHASE 2: Fine-tuning last {len(base.layers) - fine_tune_at} layers")
print("=" * 60)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=len(history1.history["accuracy"]),
    epochs=len(history1.history["accuracy"]) + EPOCHS_PHASE2,
    class_weight={0: 1.0, 1: 1.5},
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
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ],
)

# =============================================================================
# SAVE MODEL
# =============================================================================
model.save(str(MODEL_DIR / "cnn.keras"))
print("\n✅ Final model saved")

# =============================================================================
# LOAD BEST MODEL
# =============================================================================
model = tf.keras.models.load_model(
    MODEL_DIR / "cnn_best.keras",
    compile=False,
)

# =============================================================================
# FINAL EVALUATION
# =============================================================================
val_gen.reset()

y_true = val_gen.classes

y_prob = model.predict(val_gen, verbose=1).flatten()
y_pred = (y_prob > THRESHOLD).astype(int)

print("\n" + "=" * 60)
print("FINAL CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=["FAKE", "REAL"],
)

print(report)

# =============================================================================
# CONFUSION MATRIX
# =============================================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"],
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("✅ Reports generated")