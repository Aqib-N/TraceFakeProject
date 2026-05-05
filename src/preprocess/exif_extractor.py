from pathlib import Path
from PIL import Image, ExifTags
import exifread
import pandas as pd
import os

REAL_DIR = Path("data/processed/real")
FAKE_DIR = Path("data/processed/fake")
OUTPUT = Path("data/metadata.csv")

FIELDS = {"Make","Model","Software","DateTime","DateTimeOriginal"}

def exif_read(fp):
    try:
        with open(fp, "rb") as f:
            tags = exifread.process_file(f, details=False)

        out = {}
        for k,v in tags.items():
            for f in FIELDS:
                if f.lower() in k.lower():
                    out[f] = str(v)
        return out
    except:
        return {}

def pillow_exif(fp):
    try:
        img = Image.open(fp)
        exif = img._getexif()
        if not exif:
            return {}

        out = {}
        for k,v in exif.items():
            tag = ExifTags.TAGS.get(k,k)
            if tag in FIELDS:
                out[tag] = str(v)
        return out
    except:
        return {}

def extract(fp):
    exif = exif_read(fp)
    if not exif:
        exif = pillow_exif(fp)
    return exif

def build(dir_path, label):
    rows = []

    for fp in dir_path.glob("*.jpg"):
        exif = extract(fp)

        missing = sum(1 for f in FIELDS if f not in exif)

        rows.append({
            "file": fp.name,
            "label": label,
            "missing": missing,
            "has_exif": 1 if missing < len(FIELDS) else 0,
            "suspicious": 1 if missing >= 3 else 0
        })

    return rows

if __name__ == "__main__":
    data = build(REAL_DIR, 1) + build(FAKE_DIR, 0)

    df = pd.DataFrame(data)
    OUTPUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("Saved metadata.csv")