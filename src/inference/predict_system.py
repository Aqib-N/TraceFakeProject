import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
import sys
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load models - FIXED PATH
MODEL_DIR = Path(__file__).parent.parent / "models" # Changed from src/models to models
print(f"Looking for models in: {MODEL_DIR.absolute()}")

# Load CNN model
cnn_path = MODEL_DIR / "cnn.keras"
if cnn_path.exists():
    cnn = tf.keras.models.load_model(str(cnn_path))
    print(f"✅ Loaded CNN model from {cnn_path}")
else:
    print(f"❌ CNN model not found at {cnn_path}")
    # Fallback mock
    class MockCNN:
        def predict(self, x, verbose=0):
            return np.array([[0.5]])
    cnn = MockCNN()

# Load EXIF model
exif_path = MODEL_DIR / "exif_xgb.pkl"
if exif_path.exists():
    exif = joblib.load(str(exif_path))
    print(f"✅ Loaded EXIF model from {exif_path}")
else:
    print(f"❌ EXIF model not found at {exif_path}")
    class MockEXIF:
        def predict_proba(self, x):
            return np.array([[0.3, 0.7]])
    exif = MockEXIF()

# Try to load learned fusion weights
try:
    fusion_path = MODEL_DIR / "fusion_weights.pkl"
    if fusion_path.exists():
        fusion_data = joblib.load(str(fusion_path))
        FUSION_WEIGHTS = fusion_data['weights']
        FUSION_BIAS = fusion_data.get('bias', 0.0)
        print(f"✅ Loaded learned fusion weights: {FUSION_WEIGHTS}")
    else:
        raise FileNotFoundError
except:
    # Optimized default weights
    FUSION_WEIGHTS = {'cnn': 0.45, 'exif': 0.15, 'forensic': 0.40}
    FUSION_BIAS = 0.0
    print(f"⚠️ Using default fusion weights: {FUSION_WEIGHTS}")

# =========================
# CNN PREDICTION
# =========================
def img_pred(path):
    """CNN deep feature prediction"""
    try:
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        return float(cnn.predict(arr, verbose=0)[0][0])
    except Exception as e:
        print(f"CNN prediction error: {e}")
        return 0.5

# =========================
# EXIF PREDICTION
# =========================
def exif_pred(features):
    """XGBoost EXIF metadata prediction"""
    try:
        feature_cols = [
            "missing_count", "has_camera_info", "has_software",
            "software_suspicious", "has_timestamp", "timestamp_consistent",
            "exif_total_tags", "has_gps", "has_flash", "has_orientation"
        ]
        
        arr = np.array([[features.get(f, 0) for f in feature_cols]])
        
        if hasattr(exif, 'predict_proba'):
            return float(exif.predict_proba(arr)[0][1])
        else:
            return float(exif.predict(arr)[0])
    except Exception as e:
        print(f"EXIF prediction error: {e}")
        return 0.5

# =========================
# FORENSIC PREDICTION
# =========================
def forensic_pred(path):
    """Forensic signal prediction"""
    try:
        from preprocess.forensics import forensic_score
        return forensic_score(path)
    except Exception as e:
        print(f"Forensic prediction error: {e}")
        return 0.5

# =========================
# FINAL FUSION MODEL
# =========================
def final_predict(path, features=None):
    """
    Multi-signal fusion with optimized weights
    """
    # Get individual signals
    cnn_score = img_pred(path)
    
    if features is None:
        try:
            from preprocess.exif_extractor import extract
            exif_raw = extract(path)
            features = {
                "missing_count": len([f for f in ["Make", "Model", "Software", "DateTime"] if f not in exif_raw]),
                "has_camera_info": 1 if ("Make" in exif_raw and "Model" in exif_raw) else 0,
                "has_software": 1 if "Software" in exif_raw else 0,
                "software_suspicious": 0,
                "has_timestamp": 1 if ("DateTime" in exif_raw or "DateTimeOriginal" in exif_raw) else 0,
                "timestamp_consistent": 0,
                "exif_total_tags": len(exif_raw),
                "has_gps": 1 if "GPSInfo" in exif_raw else 0,
                "has_flash": 1 if "Flash" in exif_raw else 0,
                "has_orientation": 1 if "Orientation" in exif_raw else 0
            }
        except:
            features = {
                "missing_count": 4, "has_camera_info": 0, "has_software": 0,
                "software_suspicious": 0, "has_timestamp": 0, "timestamp_consistent": 0,
                "exif_total_tags": 0, "has_gps": 0, "has_flash": 0, "has_orientation": 0
            }
    
    exif_score = exif_pred(features)
    forensic = forensic_pred(path)
    
    # Weighted fusion
    final_score = (
        FUSION_WEIGHTS['cnn'] * cnn_score +
        FUSION_WEIGHTS['exif'] * exif_score +
        FUSION_WEIGHTS['forensic'] * forensic +
        FUSION_BIAS
    )
    
    final_score = max(0.0, min(1.0, final_score))
    
    return {
        "cnn_score": round(cnn_score, 4),
        "exif_score": round(exif_score, 4),
        "forensic_score": round(forensic, 4),
        "final_score": round(final_score, 4),
        "result": "REAL" if final_score > 0.5 else "FAKE",
        "confidence": round(max(final_score, 1 - final_score), 4)
    }

# =========================
# BATCH PREDICTION
# =========================
def batch_predict(image_paths):
    """Process multiple images"""
    results = []
    for path in image_paths:
        result = final_predict(path)
        result['file'] = str(path)
        results.append(result)
    return results