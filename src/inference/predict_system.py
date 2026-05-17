import os
import sys
import logging
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path

# Silence TF noise 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"]        = "ERROR"
os.environ["GLOG_minloglevel"]      = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# Path setup 
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

try:
    from config import (
        MODEL_DIR, DEFAULT_FUSION_WEIGHTS, DEFAULT_FUSION_BIAS,
        EXIF_FEATURE_COLS,
    )
except ImportError:
    MODEL_DIR             = BASE_DIR / "src" / "models"
    DEFAULT_FUSION_WEIGHTS = {"cnn": 0.45, "exif": 0.15, "forensic": 0.40}
    DEFAULT_FUSION_BIAS    = 0.0
    EXIF_FEATURE_COLS      = [
        "missing_count", "has_camera_info", "has_software",
        "software_suspicious", "has_timestamp", "timestamp_consistent",
        "timestamp_plausible", "timestamp_future",
        "exif_total_tags", "has_gps", "has_flash", "has_orientation",
    ]

from exif_extractor import extract as exif_extract, extract_features, build_feature_array
from forensics import forensic_score

print(f"Looking for models in: {MODEL_DIR.absolute()}")

# Load CNN
cnn_path = MODEL_DIR / "cnn.keras"
if cnn_path.exists():
    cnn = tf.keras.models.load_model(str(cnn_path))
    print(f"[TraceFake] ✅ CNN loaded from {cnn_path}")
else:
    print(f"⚠️  CNN model not found at {cnn_path} — using neutral mock")
    class _MockCNN:
        def predict(self, x, verbose=0):
            return np.array([[0.5]])
    cnn = _MockCNN()

# Load EXIF XGBoost ─
exif_path = MODEL_DIR / "exif_xgb.pkl"
if exif_path.exists():
    exif_model = joblib.load(str(exif_path))
    print(f"✅ Loaded EXIF model from {exif_path}")
else:
    print(f"⚠️  EXIF model not found at {exif_path} — using neutral mock")
    class _MockEXIF:
        def predict_proba(self, x):
            return np.array([[0.5, 0.5]])
    exif_model = _MockEXIF()

# Load fusion weights
fusion_path = MODEL_DIR / "fusion_weights.pkl"
if fusion_path.exists():
    try:
        fusion_data    = joblib.load(str(fusion_path))
        FUSION_WEIGHTS = fusion_data["weights"]
        FUSION_BIAS    = fusion_data.get("bias", 0.0)
        print(f"✅ Loaded learned fusion weights: {FUSION_WEIGHTS}")
    except Exception as e:
        print(f"⚠️  Could not load fusion weights ({e}) — using defaults")
        FUSION_WEIGHTS = DEFAULT_FUSION_WEIGHTS
        FUSION_BIAS    = DEFAULT_FUSION_BIAS
else:
    FUSION_WEIGHTS = DEFAULT_FUSION_WEIGHTS
    FUSION_BIAS    = DEFAULT_FUSION_BIAS
    print(f"⚠️  Using default fusion weights: {FUSION_WEIGHTS}")


# CNN Prediction 

def img_pred(path) -> float:
    """CNN deep-feature prediction. Returns probability in [0, 1]."""
    try:
        img = keras_image.load_img(path, target_size=(224, 224))
        arr = keras_image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        score = float(cnn.predict(arr, verbose=0)[0][0])
        return score
    except Exception as e:
        print(f"[TraceFake] CNN error: {e}")
        return 0.5  



# EXIF Prediction 

def exif_pred(features: dict) -> float:
    """
    XGBoost EXIF metadata prediction.
    """
    try:
        arr = build_feature_array(features)   # shape (1, N), guaranteed order
        if hasattr(exif_model, "predict_proba"):
            return float(exif_model.predict_proba(arr)[0][1])
        return float(exif_model.predict(arr)[0])
    except Exception as e:
        print(f"EXIF prediction error: {e}")
        return 0.5


# Forensic Prediction

def forensic_pred(path) -> float:
    """
    Forensic signal prediction (ELA + Noise + JPEG + Chromatic + FFT).
    FIX: was 'from preprocess.forensics import forensic_score' — always failed.
    """
    try:
        return forensic_score(path)
    except Exception as e:
        print(f"[TraceFake] Forensic error: {e}")
        return 0.5   


# Final Fusion 

def final_predict(path, features: dict = None) -> dict:

    # CNN 
    cnn_score = img_pred(path)

    # EXIF 
  
    if features is None:
        exif_raw = exif_extract(path)
        features = extract_features(exif_raw)

    exif_score = exif_pred(features)

    # Forensics
    forensic = forensic_pred(path)

    # Weighted fusion
    final_score = (
        FUSION_WEIGHTS["cnn"]      * cnn_score  +
        FUSION_WEIGHTS["exif"]     * exif_score +
        FUSION_WEIGHTS["forensic"] * forensic   +
        FUSION_BIAS
    )
    final_score = float(max(0.0, min(1.0, final_score)))

    #  Uncertainty band 
    if 0.40 < final_score < 0.60:
        result     = "UNCERTAIN"
        confidence = round(abs(final_score - 0.5) * 2, 4)  # 0 = max uncertainty
    elif final_score >= 0.60:
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


# Batch Prediction 

def batch_predict(image_paths) -> list:
    """
    Process multiple images.
    """
    results = []
    for path in image_paths:
        try:
            result          = final_predict(path)
            result["file"]  = str(path)
            result["error"] = None
        except Exception as e:
            result = {
                "file":   str(path),
                "error":  str(e),
                "result": "ERROR",
            }
        results.append(result)
    return results