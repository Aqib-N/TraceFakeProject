"""
TraceFake AI — Multi-Signal Fusion Prediction System
Uses learned meta-weights instead of fixed weights for optimal accuracy
"""

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Load models
MODEL_DIR = Path("src/models")
cnn = tf.keras.models.load_model(str(MODEL_DIR / "cnn.keras"))
exif = joblib.load(str(MODEL_DIR / "exif_xgb.pkl"))

# Try to load learned fusion weights, fallback to optimized defaults
try:
    fusion_data = joblib.load(str(MODEL_DIR / "fusion_weights.pkl"))
    FUSION_WEIGHTS = fusion_data['weights']
    FUSION_BIAS = fusion_data.get('bias', 0.0)
    print(f"Loaded learned fusion weights: {FUSION_WEIGHTS}")
except:
    # Optimized default weights based on research
    # CNN (EfficientNet): 97.16% accuracy [^3^]
    # ELA forensic: 98% accuracy [^22^]
    # EXIF: ~85% accuracy (weaker signal)
    FUSION_WEIGHTS = {'cnn': 0.45, 'exif': 0.15, 'forensic': 0.40}
    FUSION_BIAS = 0.0
    print(f"Using default fusion weights: {FUSION_WEIGHTS}")


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
        # Ensure feature order matches training
        feature_cols = [
            "missing_count", "has_camera_info", "has_software",
            "software_suspicious", "has_timestamp", "timestamp_consistent",
            "exif_total_tags", "has_gps", "has_flash", "has_orientation"
        ]
        
        # Build feature vector, default to 0 for missing
        arr = np.array([[features.get(f, 0) for f in feature_cols]])
        return float(exif.predict_proba(arr)[0][1])
    except Exception as e:
        print(f"EXIF prediction error: {e}")
        return 0.5


# =========================
# FORENSIC PREDICTION
# =========================
def forensic_pred(path):
    """Forensic signal prediction"""
    from src.preprocess.forensics import forensic_score
    return forensic_score(path)


# =========================
# FINAL FUSION MODEL
# =========================
def final_predict(path, features=None):
    """
    Multi-signal fusion with optimized weights
    Returns detailed breakdown for transparency
    """
    # Get individual signals
    cnn_score = img_pred(path)
    
    if features is None:
        # Extract features on-the-fly if not provided
        from src.preprocess.exif_extractor import extract
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
    
    exif_score = exif_pred(features)
    forensic = forensic_pred(path)
    
    # Weighted fusion
    final_score = (
        FUSION_WEIGHTS['cnn'] * cnn_score +
        FUSION_WEIGHTS['exif'] * exif_score +
        FUSION_WEIGHTS['forensic'] * forensic +
        FUSION_BIAS
    )
    
    # Clamp to valid probability range
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