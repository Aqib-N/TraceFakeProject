import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Silence TF noise ──────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import IMG_SIZE, BATCH_SIZE, DATA_DIR, MODEL_DIR, REPORT_DIR
except ImportError:
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32
    DATA_DIR   = Path("data/processed")
    MODEL_DIR  = Path("src/models")
    REPORT_DIR = Path("reports")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS      = 30   # with early stopping; typically stops at 15–20
INITIAL_LR  = 1e-5  # FIX: LOW lr so unfrozen backbone doesn't destroy ImageNet weights


# ── DIAGNOSTIC: Check dataset BEFORE training ─────────────────────────────────
print("\n" + "=" * 60)
print("DATASET DIAGNOSTIC")
print("=" * 60)

fake_dir = DATA_DIR / "fake"
real_dir = DATA_DIR / "real"

if not fake_dir.exists() or not real_dir.exists():
    raise FileNotFoundError(
        f"Expected {fake_dir} and {real_dir}\n"
        f"Found in {DATA_DIR}: {list(DATA_DIR.iterdir()) if DATA_DIR.exists() else 'DIRECTORY MISSING'}"
    )

fake_files = list(fake_dir.glob("*.[jp][pn][g]")) + list(fake_dir.glob("*.jpeg"))
real_files = list(real_dir.glob("*.[jp][pn][g]")) + list(real_dir.glob("*.jpeg"))

print(f"Fake images: {len(fake_files)}")
print(f"Real images: {len(real_files)}")

if len(fake_files) == 0 or len(real_files) == 0:
    raise ValueError("One or both class folders are empty!")

# Check for accidental duplicates between classes
fake_names = {f.name for f in fake_files}
real_names = {f.name for f in real_files}
overlap = fake_names & real_names
if overlap:
    print(f"⚠️  WARNING: {len(overlap)} filenames exist in BOTH fake/ and real/")
    print(f"   Examples: {list(overlap)[:5]}")
    print("   This causes label noise — rename duplicates before training.")

# Check class balance
ratio = len(fake_files) / max(len(real_files), 1)
print(f"Class ratio fake/real: {ratio:.2f}")
if ratio < 0.5 or ratio > 2.0:
    print("⚠️  Severe class imbalance detected — class_weight will compensate")

total = len(fake_files) + len(real_files)
cw_fake = total / (2 * len(fake_files))
cw_real = total / (2 * len(real_files))
class_weight = {0: cw_fake, 1: cw_real}
print(f"Class weights: FAKE={cw_fake:.3f}, REAL={cw_real:.3f}")
print("=" * 60 + "\n")


# ── Focal Loss (FIX: replaces binary_crossentropy) ────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for binary classification.
    gamma=2: focuses training on hard misclassified examples.
    alpha=0.25: slight up-weight for the minority class (fake).
    """
    def loss(y_true, y_pred):
        y_pred  = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce     = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        fl      = alpha_t * K.pow(1 - p_t, gamma) * bce
        return K.mean(fl)
    return loss


# ── DCT Frequency Branch ──────────────────────────────────────────────────────
def build_dct_branch(input_tensor):
    """
    Lightweight frequency-domain branch.
    Converts RGB → grayscale → DCT-like features via learned conv filters.
    GAN artifacts (grid patterns, frequency peaks) are detected here.
    """
    # Convert to grayscale via learned 1x1 conv
    x = layers.Conv2D(1, 1, padding="same", use_bias=False,
                      name="rgb_to_gray")(input_tensor)
    # High-frequency detail extractor (Laplacian-like)
    x = layers.Conv2D(16, 3, padding="same", activation="relu",
                      name="freq_conv1")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      name="freq_conv2")(x)
    x = layers.MaxPooling2D(4)(x)   # downsample aggressively
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      name="freq_conv3")(x)
    x = layers.GlobalAveragePooling2D(name="freq_gap")(x)
    return x


# ── Model Architecture ────────────────────────────────────────────────────────
inp = layers.Input(shape=(224, 224, 3), name="image_input")

# FIX: backbone FULLY UNFROZEN from the start with low LR
# ImageNet features alone cannot detect GAN artifacts
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inp,
)
base.trainable = True   # FIX: was False → that's why accuracy was stuck at 0.5

# Dual pooling (catches both smooth-region and edge artifacts)
avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
max_pool = layers.GlobalMaxPooling2D(name="max_pool")(base.output)
rgb_feat = layers.Concatenate(name="pool_concat")([avg_pool, max_pool])

# Frequency branch (NEW)
freq_feat = build_dct_branch(inp)

# Merge RGB + frequency features
merged = layers.Concatenate(name="merge")([rgb_feat, freq_feat])
x      = layers.BatchNormalization()(merged)
x      = layers.Dense(512, activation="swish")(x)
x      = layers.Dropout(0.5)(x)
x      = layers.Dense(256, activation="swish")(x)
x      = layers.Dropout(0.4)(x)
x      = layers.Dense(128, activation="swish")(x)
x      = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid", name="output")(x)

model = models.Model(inputs=inp, outputs=output)

print(f"Model parameters: {model.count_params():,}")
print(f"Trainable params : {sum(np.prod(w.shape) for w in model.trainable_weights):,}")


# ── Data Pipeline ─────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.20,
    shear_range=0.10,
    brightness_range=[0.75, 1.25],
    channel_shift_range=20.0,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="training",
    classes=["fake", "real"], shuffle=True, seed=42,
)
val_gen = val_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation",
    classes=["fake", "real"], shuffle=False, seed=42,
)

steps_per_epoch = int(np.ceil(train_gen.samples / BATCH_SIZE))
val_steps       = int(np.ceil(val_gen.samples   / BATCH_SIZE))

print(f"\nTraining: {train_gen.samples} | Validation: {val_gen.samples}")
print(f"Class map: {train_gen.class_indices}")   # should be {'fake':0, 'real':1}


# ── Compile & Train ───────────────────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss=focal_loss(gamma=2.0, alpha=0.25),   # FIX: focal loss
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("\n" + "=" * 60)
print("TRAINING (fully unfrozen backbone + focal loss)")
print("Watch: AUC should exceed 0.60 by epoch 5")
print("If AUC stays at 0.50 after 5 epochs → data problem, stop training")
print("=" * 60)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    class_weight=class_weight,
    callbacks=[
        # Primary monitor: AUC
        callbacks.EarlyStopping(
            monitor="val_auc", patience=8,
            restore_best_weights=True, mode="max",
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "cnn_best.keras"),
            monitor="val_auc", save_best_only=True, mode="max",
        ),
        # LR reduction on loss plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=3, min_lr=1e-7, verbose=1,
        ),
        # Early abort if model is clearly not learning after 5 epochs
        callbacks.EarlyStopping(
            monitor="val_auc", patience=5,
            baseline=0.55,       # must beat random by epoch 5
            restore_best_weights=True, mode="max",
        ),
    ],
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(str(MODEL_DIR / "cnn.keras"))
print("\n✅ Model saved")

# ── Evaluation ────────────────────────────────────────────────────────────────
val_gen.reset()
y_true       = val_gen.classes
y_pred_proba = model.predict(val_gen, steps=val_steps, verbose=0).flatten()
y_pred       = (y_pred_proba > 0.5).astype(int)[:len(y_true)]

val_auc_hist = history.history.get("val_auc", [])
val_acc_hist = history.history.get("val_accuracy", [])

print(f"\nBest Val AUC     : {max(val_auc_hist):.4f}")
print(f"Best Val Accuracy: {max(val_acc_hist):.4f}")
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=["FAKE", "REAL"])
print(report)

# Diagnosis
if max(val_auc_hist) < 0.60:
    print("\n" + "!" * 60)
    print("DIAGNOSIS: AUC still < 0.60 after full training.")
    print("This means your dataset has a structural problem.")
    print("Run this check on your data/processed folders:")
    print("  python3 -c \"")
    print("  import os; from pathlib import Path")
    print("  for cls in ['fake','real']:")
    print("    p = Path('data/processed') / cls")
    print("    files = list(p.glob('*'))")
    print("    sizes = [os.path.getsize(f) for f in files[:100]]")
    print("    print(cls, 'avg size:', sum(sizes)//len(sizes), 'bytes')\"")
    print("  If fake avg_size << real avg_size → images were corrupted/empty on download.")
    print("!" * 60)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("TraceFake CNN v2 — Training Report", fontsize=14, fontweight="bold")

epochs_range = range(1, len(val_auc_hist) + 1)

axes[0].plot(epochs_range, history.history["accuracy"],     "b-",  label="Train")
axes[0].plot(epochs_range, history.history["val_accuracy"], "r--", label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history["loss"],     "b-",  label="Train")
axes[1].plot(epochs_range, history.history["val_loss"], "r--", label="Val")
axes[1].set_title("Loss (Focal)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history.history["auc"],     "b-",  label="Train")
axes[2].plot(epochs_range, val_auc_hist,               "r--", label="Val")
axes[2].set_title("AUC"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0.6, color="orange", linestyle=":", alpha=0.7, label="0.6 target")

plt.tight_layout()
plt.savefig(REPORT_DIR / "training_graphs_v2.png", dpi=150, bbox_inches="tight")
plt.close()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.title("Confusion Matrix v2"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig(REPORT_DIR / "confusion_matrix_v2.png", dpi=150, bbox_inches="tight")
plt.close()

with open(REPORT_DIR / "cnn_report_v2.txt", "w") as f:
    f.write(report)

print(f"\n✅ Reports saved to {REPORT_DIR}")