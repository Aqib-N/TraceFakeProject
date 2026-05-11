import os
import sys
import logging
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

#Silence TensorFlow noise 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# Add src/ to path so "from preprocess.X import Y" works
sys.path.insert(0, str(Path(__file__).parent.parent))

# FIX 1: models live in src/models/ — __file__ is src/inference/predict_system.py
#         parent       = src/inference/
#         parent.parent = src/
#         + "models"   = src/models/
MODEL_DIR = Path(__file__).parent.parent / "models"

print(f"[TraceFake] Looking for models in: {MODEL_DIR.absolute()}")

#Load CNN
cnn_path = MODEL_DIR / "cnn.keras"
if cnn_path.exists():
    cnn = tf.keras.models.load_model(str(cnn_path))
    print(f"[TraceFake] ✅ CNN loaded from {cnn_path}")
else:
    print(f"[TraceFake] ⚠️  CNN not found at {cnn_path} — using neutral fallback")
    class _MockCNN:
        def predict(self, x, verbose=0):
            return np.array([[0.5]])
    cnn = _MockCNN()

#Load EXIF model 
exif_path = MODEL_DIR / "exif_xgb.pkl"
if exif_path.exists():
    exif_model = joblib.load(str(exif_path))
    print(f"[TraceFake] ✅ EXIF model loaded from {exif_path}")
else:
    print(f"[TraceFake] ⚠️  EXIF model not found at {exif_path} — using neutral fallback")
    class _MockEXIF:
        def predict_proba(self, x):
            return np.array([[0.5, 0.5]]) 
    exif_model = _MockEXIF()

#Load fusion weights
try:
    fusion_path = MODEL_DIR / "fusion_weights.pkl"
    if not fusion_path.exists():
        raise FileNotFoundError
    fusion_data = joblib.load(str(fusion_path))
    FUSION_WEIGHTS = fusion_data["weights"]
    FUSION_BIAS = fusion_data.get("bias", 0.0)
    print(f"[TraceFake] ✅ Fusion weights loaded: {FUSION_WEIGHTS}")
except Exception:
    FUSION_WEIGHTS = {"cnn": 0.45, "exif": 0.15, "forensic": 0.40}
    FUSION_BIAS = 0.0
    print(f"[TraceFake] ⚠️  Using default fusion weights: {FUSION_WEIGHTS}")


# CNN PREDICTION
def img_pred(path: str) -> float:
    """EfficientNetB0 binary prediction. Returns P(real) in [0, 1]."""
    try:
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        score = float(cnn.predict(arr, verbose=0)[0][0])
        return score
    except Exception as e:
        print(f"[TraceFake] CNN error: {e}")
        return 0.5  


# EXIF PREDICTION
FEATURE_COLS = [
    "missing_count", "has_camera_info", "has_software",
    "software_suspicious", "has_timestamp", "timestamp_consistent",
    "exif_total_tags", "has_gps", "has_flash", "has_orientation",
]

def exif_pred(features: dict) -> float:
    """XGBoost EXIF metadata prediction. Returns P(real) in [0, 1]."""
    try:
        arr = np.array([[features.get(f, 0) for f in FEATURE_COLS]])
        if hasattr(exif_model, "predict_proba"):
            return float(exif_model.predict_proba(arr)[0][1])
        else:
            return float(exif_model.predict(arr)[0])
    except Exception as e:
        print(f"[TraceFake] EXIF error: {e}")
        return 0.5  

# FORENSIC PREDICTION
def forensic_pred(path: str) -> float:
    """ELA + noise + JPEG + chromatic fusion. Returns score in [0, 1]."""
    try:
        from preprocess.forensics import forensic_score
        return forensic_score(path)
    except Exception as e:
        print(f"[TraceFake] Forensic error: {e}")
        return 0.5   

# FINAL FUSION
def final_predict(path: str, features: dict = None) -> dict:
    """
    Multi-signal fusion.
    Returns dict with cnn_score, exif_score, forensic_score,
    final_score, result ('REAL'/'FAKE'), confidence.
    """
    cnn_score = img_pred(path)

    # Build EXIF feature vector
    if features is None:
        try:
            from preprocess.exif_extractor import extract, extract_features
            exif_raw = extract(path)
            features = extract_features(exif_raw)
        except Exception as e:
            print(f"[TraceFake] EXIF feature extraction error: {e}")
            features = {f: 0 for f in FEATURE_COLS}
            features["missing_count"] = 4

    exif_present = features.get("exif_total_tags", 0) > 0

    exif_score = exif_pred(features)
    forensic   = forensic_pred(path)

    if exif_present:
        w_cnn, w_exif, w_for = (
            FUSION_WEIGHTS.get("cnn",      0.45),
            FUSION_WEIGHTS.get("exif",     0.15),
            FUSION_WEIGHTS.get("forensic", 0.40),
        )
    else:
        w_cnn, w_exif, w_for = 0.55, 0.00, 0.45

    final_score = w_cnn * cnn_score + w_exif * exif_score + w_for * forensic + FUSION_BIAS
    final_score = float(np.clip(final_score, 0.0, 1.0))

    return {
        "cnn_score":      round(cnn_score,   4),
        "exif_score":     round(exif_score,  4),
        "forensic_score": round(forensic,    4),
        "final_score":    round(final_score, 4),
        "result":         "REAL" if final_score > 0.5 else "FAKE",
        "confidence":     round(max(final_score, 1.0 - final_score), 4),
        "exif_present":   exif_present,  
    }

# BATCH PREDICTION
def batch_predict(image_paths):
    results = []
    for path in image_paths:
        result = final_predict(str(path))
        result["file"] = str(path)
        results.append(result)
    return results