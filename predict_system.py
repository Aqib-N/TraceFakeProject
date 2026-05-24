
import os
import sys
import logging
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path
import json

# ── Silence TF noise ──────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"]       = "ERROR"
os.environ["GLOG_minloglevel"]     = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

try:
    from config import (
        MODEL_DIR, DEFAULT_FUSION_WEIGHTS, DEFAULT_FUSION_BIAS,
        EXIF_FEATURE_COLS,
    )
except ImportError:
    MODEL_DIR              = BASE_DIR / "src" / "models"
    DEFAULT_FUSION_WEIGHTS = {"cnn": 0.60, "exif": 0.00, "forensic": 0.40}
    DEFAULT_FUSION_BIAS    = 0.0
    EXIF_FEATURE_COLS      = [
        "missing_count", "has_camera_info", "has_software",
        "software_suspicious", "has_timestamp", "timestamp_consistent",
        "timestamp_plausible", "timestamp_future",
        "exif_total_tags", "has_gps", "has_flash", "has_orientation",
    ]

# ── Safe imports ──────────────────────────────────────────────────────────────
try:
    from src.preprocess.exif_extractor import (
        extract as exif_extract,
        extract_features,
        build_feature_array,
    )
except ImportError:
    try:
        from exif_extractor import (
            extract as exif_extract,
            extract_features,
            build_feature_array,
        )
    except ImportError:
        def exif_extract(path):        return {}
        def extract_features(raw):     return {c: 0 for c in EXIF_FEATURE_COLS}
        def build_feature_array(feat): return np.zeros((1, len(EXIF_FEATURE_COLS)))

try:
    from src.preprocess.forensics import forensic_score
except ImportError:
    try:
        from forensics import forensic_score
    except ImportError:
        def forensic_score(path): return 0.5


# ── Mock classes ──────────────────────────────────────────────────────────────
class _MockCNN:
    def predict(self, x, verbose=0):
        return np.array([[0.5]])

class _MockEXIF:
    def predict_proba(self, x):
        return np.array([[0.5, 0.5]])


# ── Load CNN ──────────────────────────────────────────────────────────────────
# FIX: compile=False — skips loss deserialization entirely.
# focal_loss registration not needed for inference, only for training.
# Works on ALL Keras/TF versions.
_cnn_loaded = False
for _cnn_name in ["cnn_best.keras", "cnn.keras"]:
    _cnn_path = MODEL_DIR / _cnn_name
    if _cnn_path.exists():
        try:
            cnn = tf.keras.models.load_model(
                str(_cnn_path),
                compile=False,   # FIX: no loss deserialization needed
            )
            print(f"✅ CNN loaded from {_cnn_path}")
            _cnn_loaded = True
            break
        except Exception as e:
            print(f"⚠️  Failed to load {_cnn_name}: {e}")

if not _cnn_loaded:
    print("⚠️  No CNN model found — using neutral mock (0.5)")
    cnn = _MockCNN()


# ── Load EXIF model ───────────────────────────────────────────────────────────
exif_path = MODEL_DIR / "exif_xgb.pkl"
if exif_path.exists():
    try:
        exif_model = joblib.load(str(exif_path))
        print(f"✅ EXIF model loaded")
    except Exception as e:
        print(f"⚠️  EXIF model load failed: {e} — using mock")
        exif_model = _MockEXIF()
else:
    print("⚠️  EXIF model not found — using mock")
    exif_model = _MockEXIF()


# ── Load fusion weights ───────────────────────────────────────────────────────
fusion_path = MODEL_DIR / "fusion_weights.pkl"
if fusion_path.exists():
    try:
        fusion_data    = joblib.load(str(fusion_path))
        FUSION_WEIGHTS = fusion_data["weights"]
        FUSION_BIAS    = fusion_data.get("bias", 0.0)
        print(f"✅ Fusion weights: {FUSION_WEIGHTS}")
    except Exception as e:
        print(f"⚠️  Fusion weights failed: {e} — using defaults")
        FUSION_WEIGHTS = DEFAULT_FUSION_WEIGHTS
        FUSION_BIAS    = DEFAULT_FUSION_BIAS
else:
    FUSION_WEIGHTS = DEFAULT_FUSION_WEIGHTS
    FUSION_BIAS    = DEFAULT_FUSION_BIAS
    print(f"ℹ️  Default fusion weights: {FUSION_WEIGHTS}")


# ── Load decision threshold ───────────────────────────────────────────────────
THRESHOLD = 0.50
_threshold_path = MODEL_DIR / "threshold.json"
if _threshold_path.exists():
    try:
        _t        = json.loads(_threshold_path.read_text())
        THRESHOLD = float(_t.get("threshold", 0.50))
        print(f"✅ Decision threshold: {THRESHOLD}")
    except Exception:
        pass
else:
    print(f"ℹ️  Default threshold: {THRESHOLD}")


# ── Predictions ───────────────────────────────────────────────────────────────

def img_pred(path) -> float:
    try:
        img = keras_image.load_img(path, target_size=(224, 224))
        arr = keras_image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        return float(cnn.predict(arr, verbose=0)[0][0])
    except Exception as e:
        print(f"CNN error: {e}")
        return 0.5


def exif_pred(features: dict) -> float:
    try:
        meta_path = MODEL_DIR / "exif_model_meta.json"
        if meta_path.exists():
            _meta = json.loads(meta_path.read_text())
            _cols = _meta.get("feature_cols", EXIF_FEATURE_COLS)
            arr   = np.array([[features.get(c, 0) for c in _cols]])
        else:
            arr = build_feature_array(features)

        if hasattr(exif_model, "predict_proba"):
            return float(exif_model.predict_proba(arr)[0][1])
        return float(exif_model.predict(arr)[0])
    except Exception as e:
        print(f"EXIF prediction error: {e}")
        return 0.5


def forensic_pred(path) -> float:
    try:
        return forensic_score(path)
    except Exception as e:
        print(f"Forensic error: {e}")
        return 0.5


def final_predict(path, features: dict = None) -> dict:
    cnn_score = img_pred(path)

    if features is None:
        exif_raw = exif_extract(path)
        features = extract_features(exif_raw)

    exif_score = exif_pred(features)
    forensic   = forensic_pred(path)

    final_score = (
        FUSION_WEIGHTS["cnn"]      * cnn_score  +
        FUSION_WEIGHTS["exif"]     * exif_score +
        FUSION_WEIGHTS["forensic"] * forensic   +
        FUSION_BIAS
    )
    final_score = float(max(0.0, min(1.0, final_score)))

    uncertain_lo = max(0.10, THRESHOLD - 0.10)
    uncertain_hi = min(0.90, THRESHOLD + 0.10)

    if uncertain_lo < final_score < uncertain_hi:
        result     = "UNCERTAIN"
        confidence = round(abs(final_score - 0.5) * 2, 4)
    elif final_score >= THRESHOLD:
        result     = "REAL"
        confidence = round(final_score, 4)
    else:
        result     = "FAKE"
        confidence = round(1.0 - final_score, 4)

    return {
        "cnn_score":      round(cnn_score,   4),
        "exif_score":     round(exif_score,  4),
        "forensic_score": round(forensic,    4),
        "final_score":    round(final_score, 4),
        "result":         result,
        "confidence":     confidence,
    }


def batch_predict(image_paths) -> list:
    results = []
    for path in image_paths:
        try:
            result          = final_predict(path)
            result["file"]  = str(path)
            result["error"] = None
        except Exception as e:
            result = {"file": str(path), "error": str(e), "result": "ERROR"}
        results.append(result)
    return results
