"""
TraceFake AI — Enhanced EXIF Metadata Extractor
Extracts 10+ forensic features for XGBoost classifier
"""

from pathlib import Path
from PIL import Image, ExifTags
import exifread
import pandas as pd
import os

REAL_DIR = Path("data/processed/real")
FAKE_DIR = Path("data/processed/fake")
OUTPUT = Path("data/metadata.csv")

# Extended field set for better detection
FIELDS = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "GPSInfo", "Flash", "ExifVersion",
    "Orientation", "ImageWidth", "ImageLength", "XResolution",
    "YResolution", "ResolutionUnit", "YCbCrPositioning"
}

SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "paint", "fake", "gan", "deepfake",
    "stable diffusion", "midjourney", "dall-e", "generator"
]

def exif_read(fp):
    """Read EXIF using exifread (more thorough than PIL)"""
    try:
        with open(fp, "rb") as f:
            tags = exifread.process_file(f, details=False)
        
        out = {}
        for k, v in tags.items():
            for f in FIELDS:
                if f.lower() in k.lower():
                    out[f] = str(v)
        return out
    except Exception as e:
        return {}

def pillow_exif(fp):
    """Fallback EXIF reading with PIL"""
    try:
        img = Image.open(fp)
        exif = img._getexif()
        if not exif:
            return {}
        
        out = {}
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, k)
            if tag in FIELDS:
                out[tag] = str(v)
        return out
    except Exception:
        return {}

def extract(fp):
    """Combined EXIF extraction"""
    exif = exif_read(fp)
    if not exif:
        exif = pillow_exif(fp)
    return exif

def check_suspicious_software(software_str):
    """Check if software field contains suspicious terms"""
    if not software_str:
        return 0
    software_lower = str(software_str).lower()
    return 1 if any(term in software_lower for term in SUSPICIOUS_SOFTWARE) else 0

def check_timestamp_consistency(exif):
    """Check if DateTime fields are consistent"""
    dt = exif.get("DateTime", "")
    dto = exif.get("DateTimeOriginal", "")
    dtd = exif.get("DateTimeDigitized", "")
    
    if not dt or not dto:
        return 0  # Can't verify
    
    return 1 if dt == dto == dtd or (dt == dto) else 0

def build(dir_path, label):
    """Build enriched feature set for each image"""
    rows = []
    
    for fp in dir_path.glob("*.jpg"):
        exif = extract(fp)
        
        # Basic counts
        missing_count = sum(1 for f in FIELDS if f not in exif)
        has_exif = 1 if missing_count < len(FIELDS) else 0
        
        # Enhanced features
        has_camera_info = 1 if ("Make" in exif and "Model" in exif) else 0
        has_software = 1 if "Software" in exif else 0
        software_suspicious = check_suspicious_software(exif.get("Software", ""))
        has_timestamp = 1 if ("DateTime" in exif or "DateTimeOriginal" in exif) else 0
        timestamp_consistent = check_timestamp_consistency(exif)
        has_gps = 1 if "GPSInfo" in exif else 0
        has_flash = 1 if "Flash" in exif else 0
        has_orientation = 1 if "Orientation" in exif else 0
        
        rows.append({
            "file": fp.name,
            "label": label,
            # Legacy features (backward compatible)
            "missing": missing_count,
            "has_exif": has_exif,
            "suspicious": 1 if missing_count >= 3 else 0,
            # Enhanced features
            "missing_count": missing_count,
            "has_camera_info": has_camera_info,
            "has_software": has_software,
            "software_suspicious": software_suspicious,
            "has_timestamp": has_timestamp,
            "timestamp_consistent": timestamp_consistent,
            "exif_total_tags": len(exif),
            "has_gps": has_gps,
            "has_flash": has_flash,
            "has_orientation": has_orientation,
            # Raw metadata for reference
            "camera_make": exif.get("Make", ""),
            "camera_model": exif.get("Model", ""),
            "software": exif.get("Software", ""),
            "datetime": exif.get("DateTime", "")
        })
    
    return rows

def generate_summary_table(df):
    """Generate real vs fake metadata summary (Assignment requirement)"""
    summary = df.groupby("label").agg({
        "missing_count": ["mean", "sum", "count"],
        "has_exif": "mean",
        "has_camera_info": "mean",
        "has_software": "mean",
        "software_suspicious": "mean",
        "has_timestamp": "mean",
        "timestamp_consistent": "mean",
        "has_gps": "mean",
        "exif_total_tags": "mean"
    }).round(3)
    
    summary.index = ["FAKE (0)", "REAL (1)"]
    
    print("\n" + "=" * 70)
    print("EXIF METADATA SUMMARY — REAL vs FAKE IMAGES")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)
    
    # Save summary
    Path("reports").mkdir(exist_ok=True)
    summary.to_csv("reports/metadata_summary.csv")
    print("\n📊 Summary saved to reports/metadata_summary.csv")
    
    return summary

if __name__ == "__main__":
    print("Extracting EXIF metadata from real images...")
    real_data = build(REAL_DIR, 1)
    
    print("Extracting EXIF metadata from fake images...")
    fake_data = build(FAKE_DIR, 0)
    
    df = pd.DataFrame(real_data + fake_data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    
    print(f"\n✅ Saved metadata.csv with {len(df)} records")
    print(f"   Real: {len(real_data)} | Fake: {len(fake_data)}")
    
    # Generate summary table
    generate_summary_table(df)