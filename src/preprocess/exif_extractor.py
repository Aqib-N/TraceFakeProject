"""
exif_extractor.py
TraceFake AI — Enhanced EXIF Metadata Extractor
"""

from pathlib import Path
from PIL import Image, ExifTags
import numpy as np

FIELDS = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "GPSInfo", "Flash", "ExifVersion",
    "Orientation", "ImageWidth", "ImageLength"
}

# FIX 1: Expanded to match training data (preprocess_images.py FAKE_SOFTWARE list)
# FIX 2: Removed "photoshop", "gimp", "paint" — real photographers use these
SUSPICIOUS_SOFTWARE = [
    "deepfake", "deepfacelab", "faceswap", "roop",
    "midjourney", "dall-e", "dalle", "stable diffusion",
    "stylegan", "gan", "firefly", "imagen",
    "ai generated", "synthetic", "generator", "swapper",
    "face generator", "deepfake creator"
]


def extract(image_path):
    """
    Extract EXIF metadata from image.
    Handles JPEG and PNG safely (PNG has no _getexif).
    """
    exif_data = {}

    try:
        with Image.open(image_path) as img:
            # FIX 3: PNG files don't have _getexif — use getexif() which works for both
            try:
                exif_raw = img.getexif()          # Pillow ≥ 6.0, works on JPEG & PNG
            except AttributeError:
                exif_raw = img._getexif() or {}   # Fallback for older Pillow

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
        # Don't crash — just return empty dict
        pass

    return exif_data


def extract_features(exif_data):
    """
    Convert EXIF data to feature vector for XGBoost prediction.
    Feature names must exactly match training columns in train_exif_model.py.
    """
    features = {
        "missing_count": 0,
        "has_camera_info": 0,
        "has_software": 0,
        "software_suspicious": 0,
        "has_timestamp": 0,
        "timestamp_consistent": 0,
        "exif_total_tags": len(exif_data),
        "has_gps": 0,
        "has_flash": 0,
        "has_orientation": 0,
    }

    # Camera info
    if "Make" in exif_data and "Model" in exif_data:
        features["has_camera_info"] = 1  # FIX 4: require BOTH, not just one
    elif "Make" in exif_data or "Model" in exif_data:
        features["has_camera_info"] = 1  # partial — still flag as present

    # Software
    if "Software" in exif_data:
        features["has_software"] = 1
        software_lower = exif_data["Software"].lower()
        if any(term in software_lower for term in SUSPICIOUS_SOFTWARE):
            features["software_suspicious"] = 1

    # Timestamps
    has_dt = "DateTime" in exif_data
    has_dto = "DateTimeOriginal" in exif_data
    features["has_timestamp"] = 1 if (has_dt or has_dto) else 0

    if has_dt and has_dto:
        # Consistent if both timestamps agree
        features["timestamp_consistent"] = (
            1 if exif_data["DateTime"] == exif_data["DateTimeOriginal"] else 0
        )
    else:
        features["timestamp_consistent"] = 0

    # GPS
    if "GPSInfo" in exif_data:
        features["has_gps"] = 1

    # Flash
    if "Flash" in exif_data:
        features["has_flash"] = 1

    # Orientation
    if "Orientation" in exif_data:
        features["has_orientation"] = 1

    # Missing critical fields
    critical_fields = ["Make", "Model", "DateTime", "Software"]
    features["missing_count"] = sum(1 for f in critical_fields if f not in exif_data)

    return features


def build_single_image(image_path):
    """Extract features for a single image. Returns (features_dict, exif_dict)."""
    exif_data = extract(image_path)
    features = extract_features(exif_data)
    return features, exif_data