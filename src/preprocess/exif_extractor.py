from pathlib import Path
from PIL import Image, ExifTags
from datetime import datetime
import numpy as np

# Import canonical feature list and suspicious-software terms from config
try:
    from config import SUSPICIOUS_SOFTWARE_TERMS, EXIF_FEATURE_COLS
except ImportError:
    # Fallback if config.py not on path
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


FIELDS = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "GPSInfo", "Flash", "ExifVersion",
    "Orientation", "ImageWidth", "ImageLength",
}


# Raw EXIF Extraction 


def extract(image_path):
    """
    Extract EXIF metadata from image using PIL only.
    Returns dict of tag_name → string_value for fields in FIELDS.
    Never raises; returns empty dict on any failure.
    """
    exif_data = {}
    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()
            if exif_raw:
                for tag_id, value in exif_raw.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag in FIELDS:
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="ignore")
                            except Exception:
                                value = str(value)
                        exif_data[tag] = str(value)[:100]
    except Exception as e:
        # Non-fatal — downstream callers handle missing EXIF gracefully
        print(f"EXIF extraction warning for {image_path}: {e}")
    return exif_data


# Timestamp Helpers 

def _parse_exif_datetime(dt_string: str):
    """Parse EXIF datetime string; returns datetime or None."""
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_string.strip(), fmt)
        except ValueError:
            continue
    return None


def timestamp_is_plausible(dt_string: str) -> bool:
    """
    Return True if the timestamp is within a reasonable photographic range.
    Rejects epoch defaults (1970/1980), pre-photography dates, and future dates.
    """
    dt = _parse_exif_datetime(dt_string)
    if dt is None:
        return False
    now = datetime.now()
    return 1990 < dt.year <= now.year and dt <= now


# Feature Extraction 

def extract_features(exif_data: dict) -> dict:
    """
    Convert raw EXIF dict → feature dict.
    Keys are exactly EXIF_FEATURE_COLS — callers must not assume ordering;
    use build_feature_array() to get a guaranteed-order numpy array.

    New features vs original:
      - timestamp_plausible : 1 if DateTime passes sanity checks
      - timestamp_future    : 1 if DateTime is in the future (strong fake signal)
    """
    features = {col: 0 for col in EXIF_FEATURE_COLS}

    # Camera make / model
    if "Make" in exif_data or "Model" in exif_data:
        features["has_camera_info"] = 1

    # Software 
    if "Software" in exif_data:
        features["has_software"] = 1
        software_lower = exif_data["Software"].lower()
        # FIX: was always 0 in predict_system.py inline fallback
        if any(term in software_lower for term in SUSPICIOUS_SOFTWARE_TERMS):
            features["software_suspicious"] = 1

    # Timestamps ──
    has_dt     = "DateTime"         in exif_data
    has_dt_orig = "DateTimeOriginal" in exif_data
    features["has_timestamp"] = int(has_dt or has_dt_orig)

    if has_dt and has_dt_orig:
        features["timestamp_consistent"] = int(
            exif_data["DateTime"] == exif_data["DateTimeOriginal"]
        )
    # else remains 0 (incomplete timestamps = not consistent)

    # NEW: plausibility checks
    primary_ts = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime", "")
    if primary_ts:
        plausible = timestamp_is_plausible(primary_ts)
        features["timestamp_plausible"] = int(plausible)
        # Future date is a strong manipulation signal
        dt = _parse_exif_datetime(primary_ts)
        if dt and dt > datetime.now():
            features["timestamp_future"] = 1

    # Other fields 
    features["has_gps"]         = int("GPSInfo"     in exif_data)
    features["has_flash"]       = int("Flash"        in exif_data)
    features["has_orientation"] = int("Orientation"  in exif_data)
    features["exif_total_tags"] = len(exif_data)

    # Missing critical fields count 
    critical = ["Make", "Model", "DateTime", "Software"]
    features["missing_count"] = sum(1 for f in critical if f not in exif_data)

    return features


def build_feature_array(features: dict) -> "np.ndarray":
    """
    Return a (1, N) numpy array in the canonical EXIF_FEATURE_COLS order.
    This guarantees the XGBoost model always receives features in the same
    order it was trained on, regardless of dict iteration order.
    """
    return np.array([[features.get(col, 0) for col in EXIF_FEATURE_COLS]])


# Convenience wrapper 

def build_single_image(image_path):
    """
    Extract features for a single image.
    Returns (features_dict, exif_raw_dict).
    """
    exif_data = extract(image_path)
    features  = extract_features(exif_data)
    return features, exif_data