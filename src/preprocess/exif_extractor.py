"""
exif_extractor.py
TraceFake AI — Enhanced EXIF Metadata Extractor
Simplified for cloud deployment
"""

from pathlib import Path
from PIL import Image, ExifTags
import numpy as np

# Simplified field set for cloud deployment
FIELDS = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "GPSInfo", "Flash", "ExifVersion",
    "Orientation", "ImageWidth", "ImageLength"
}

SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "paint", "fake", "gan", "deepfake",
    "stable diffusion", "midjourney", "dall-e", "generator"
]

def extract(image_path):
    """
    Extract EXIF metadata from image
    Works with PIL only (no exifread dependency)
    """
    exif_data = {}
    
    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()
            
            if exif_raw:
                for tag_id, value in exif_raw.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag in FIELDS:
                        # Convert bytes to string
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = str(value)
                        exif_data[tag] = str(value)[:100]
                        
    except Exception as e:
        print(f"EXIF extraction warning: {e}")
    
    return exif_data

def extract_features(exif_data):
    """
    Convert EXIF data to feature vector for prediction
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
        "has_orientation": 0
    }
    
    # Check for camera make/model
    if "Make" in exif_data or "Model" in exif_data:
        features["has_camera_info"] = 1
    
    # Check for software
    if "Software" in exif_data:
        features["has_software"] = 1
        software = exif_data["Software"].lower()
        if any(term in software for term in SUSPICIOUS_SOFTWARE):
            features["software_suspicious"] = 1
    
    # Check timestamps
    has_date = "DateTime" in exif_data or "DateTimeOriginal" in exif_data
    features["has_timestamp"] = 1 if has_date else 0
    
    # Timestamp consistency (simplified)
    if "DateTime" in exif_data and "DateTimeOriginal" in exif_data:
        if exif_data["DateTime"] == exif_data["DateTimeOriginal"]:
            features["timestamp_consistent"] = 1
    elif "DateTime" in exif_data or "DateTimeOriginal" in exif_data:
        features["timestamp_consistent"] = 0  # Incomplete
    else:
        features["timestamp_consistent"] = 0
    
    # Check GPS
    if "GPSInfo" in exif_data:
        features["has_gps"] = 1
    
    # Check flash
    if "Flash" in exif_data:
        features["has_flash"] = 1
    
    # Check orientation
    if "Orientation" in exif_data:
        features["has_orientation"] = 1
    
    # Count missing critical fields
    critical_fields = ["Make", "Model", "DateTime", "Software"]
    missing = sum(1 for f in critical_fields if f not in exif_data)
    features["missing_count"] = missing
    
    return features

# Simplified version of build function (for cloud)
def build_single_image(image_path):
    """Extract features for a single image"""
    exif_data = extract(image_path)
    features = extract_features(exif_data)
    return features, exif_data