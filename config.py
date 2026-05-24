"""
TraceFake AI — Centralized Configuration
Single source of truth for all paths and constants.
Fixes: MODEL_DIR inconsistency between train_cnn.py and predict_system.py
"""

from pathlib import Path

# Root & Directories 
ROOT_DIR    = Path(__file__).parent
MODEL_DIR   = ROOT_DIR / "src" / "models"
DATA_DIR    = ROOT_DIR / "data" / "processed"
REPORT_DIR  = ROOT_DIR / "reports"
METADATA_CSV = ROOT_DIR / "data" / "metadata.csv"

# Auto-create directories on import
for _d in [MODEL_DIR, DATA_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Image Settings 
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# Training Settings 
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15
INITIAL_LR    = 1e-3

# Upload Validation 
MAX_FILE_SIZE_MB  = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}

# Fusion Weights (defaults; overridden by fusion_weights.pkl if present) 
DEFAULT_FUSION_WEIGHTS = {"cnn": 0.45, "exif": 0.15, "forensic": 0.40}
DEFAULT_FUSION_BIAS    = 0.0

# EXIF 
SUSPICIOUS_SOFTWARE_TERMS = [
    "photoshop", "gimp", "paint", "fake", "gan", "deepfake",
    "stable diffusion", "midjourney", "dall-e", "generator",
    "firefly", "imagen", "runway", "pika", "leonardo",
]

EXIF_FEATURE_COLS = [
    "missing_count", "has_camera_info", "has_software",
    "software_suspicious", "has_timestamp", "timestamp_consistent",
    "timestamp_plausible", "timestamp_future",
    "exif_total_tags", "has_gps", "has_flash", "has_orientation",
]