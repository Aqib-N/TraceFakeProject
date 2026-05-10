"""
preprocess_images.py
TraceFake AI — Dataset Preprocessing & EXIF Injection
Supports: Kaggle, Google Colab, local
"""

import random
import hashlib
import piexif
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# SOURCE PATHS — uncomment the block that matches your environment
# =============================================================================

# ── Kaggle ────────────────────────────────────────────────────────────────────
SRC_REAL = Path("/kaggle/input/datasets/aqibnawaz7/real-vs-fake-image-dataset/real_image_processed/real_image_processed")
SRC_FAKE = Path("/kaggle/input/datasets/aqibnawaz7/real-vs-fake-image-dataset/fake_image_processed/fake_image_processed")

# ── Google Colab (uncomment to use) ──────────────────────────────────────────
# SRC_REAL = Path("/content/drive/MyDrive/real_image_processed")
# SRC_FAKE = Path("/content/drive/MyDrive/fake_image_processed")

# ── Local (uncomment to use) ─────────────────────────────────────────────────
# SRC_REAL = Path("raw_data/real")
# SRC_FAKE = Path("raw_data/fake")

# =============================================================================
# OUTPUT PATHS
# =============================================================================
OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")
OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = (224, 224)
MAX_IMAGES = 21300      # per class
# FIX 1: added NUM_WORKERS for parallel processing on Kaggle/Colab
NUM_WORKERS = 4

# =============================================================================
# CAMERA / SOFTWARE LISTS
# =============================================================================
REAL_CAMERAS = [
    # Canon
    (b'Canon', b'EOS R5'), (b'Canon', b'EOS R6 Mark II'), (b'Canon', b'EOS R8'),
    (b'Canon', b'EOS R50'), (b'Canon', b'EOS 5D Mark IV'), (b'Canon', b'EOS R3'),
    (b'Canon', b'EOS R7'), (b'Canon', b'EOS 90D'),
    # Nikon
    (b'Nikon', b'Z9'), (b'Nikon', b'Z8'), (b'Nikon', b'Z7 II'),
    (b'Nikon', b'Z6 II'), (b'Nikon', b'Z5'), (b'Nikon', b'Zf'),
    (b'Nikon', b'D850'), (b'Nikon', b'D780'),
    # Sony
    (b'Sony', b'ILCE-1'), (b'Sony', b'ILCE-7RM5'), (b'Sony', b'ILCE-7M4'),
    (b'Sony', b'ILCE-7CM2'), (b'Sony', b'ILCE-6700'), (b'Sony', b'ILCE-9M3'),
    (b'Sony', b'ILCE-7SM3'), (b'Sony', b'ZV-E1'),
    # Fujifilm
    (b'FUJIFILM', b'X-H2S'), (b'FUJIFILM', b'X-H2'), (b'FUJIFILM', b'X-T5'),
    (b'FUJIFILM', b'X-S20'), (b'FUJIFILM', b'X100VI'), (b'FUJIFILM', b'GFX 100 II'),
    (b'FUJIFILM', b'X-T30 II'),
    # Panasonic
    (b'Panasonic', b'DC-S5M2'), (b'Panasonic', b'DC-S5M2X'),
    (b'Panasonic', b'DC-G9M2'), (b'Panasonic', b'DC-S1R'),
    (b'Panasonic', b'DC-S5'), (b'Panasonic', b'Lumix GH6'),
    # Specialty
    (b'Leica', b'M11'), (b'Leica', b'Q3'), (b'Leica', b'SL3'),
    (b'OM System', b'OM-1 Mark II'), (b'OM SYSTEM', b'OM-5'),
    (b'Pentax', b'K-3 Mark III'), (b'RICOH', b'GR IIIx'),
    (b'Hasselblad', b'X2D 100C'), (b'Hasselblad', b'907X'),
    # Smartphones
    (b'Apple', b'iPhone 15 Pro'), (b'Apple', b'iPhone 15 Pro Max'),
    (b'Apple', b'iPhone 15'), (b'Apple', b'iPhone 14 Pro Max'),
    (b'Samsung', b'Galaxy S24 Ultra'), (b'Samsung', b'Galaxy S24+'),
    (b'Samsung', b'Galaxy S23 Ultra'), (b'Samsung', b'Galaxy Z Fold 5'),
    (b'Google', b'Pixel 8 Pro'), (b'Google', b'Pixel 8'), (b'Google', b'Pixel 7 Pro'),
    (b'Xiaomi', b'13 Ultra'), (b'Xiaomi', b'14 Pro'),
    (b'OnePlus', b'12'), (b'OnePlus', b'Open'),
    (b'Huawei', b'P60 Pro'), (b'Huawei', b'Mate 60 Pro'),
    (b'Nothing', b'Phone 2'), (b'Vivo', b'X100 Pro'), (b'Asus', b'Zenfone 10'),
]

FAKE_CAMERAS = [
    (b'Midjourney', b'v6.1'), (b'Midjourney', b'v6'), (b'Midjourney', b'v5.2'),
    (b'OpenAI', b'DALL-E 3'), (b'OpenAI', b'DALL-E 2'),
    (b'Stability AI', b'Stable Diffusion 3.5'),
    (b'Stability AI', b'Stable Diffusion XL'),
    (b'Google', b'Imagen 2'), (b'Meta', b'Imagine'),
    (b'Adobe', b'Firefly 2'), (b'Canva', b'AI Generator'),
    (b'DeepFaceLab', b'3.0.1'), (b'Roop', b'Face Swapper'),
    (b'StyleGAN', b'StyleGAN3'), (b'Unknown', b'AI Generated'),
]

REAL_SOFTWARE = [
    b'Adobe Photoshop 2025', b'Adobe Photoshop 2024',
    b'Adobe Lightroom Classic 14.0', b'Adobe Lightroom Classic 13.0',
    b'Adobe Camera Raw 17.0', b'Capture One Pro 16.5',
    b'DxO PhotoLab 8', b'Luminar Neo 1.20',
    b'Affinity Photo 2.5', b'Darktable 4.8',
    b'GIMP 2.10.38', b'GIMP 3.0',
    b'Snapseed 2.0', b'Lightroom Mobile 9.0',
]

FAKE_SOFTWARE = [
    b'DeepFaceLab 3.0', b'Roop Face Swapper v1.3',
    b'Midjourney AI v6.1', b'DALL-E 3 Generator',
    b'Stable Diffusion 3.5', b'Adobe Firefly 2',
    b'StyleGAN3 Generator', b'AI Generated Content',
    b'Deepfake Creator Pro', b'Face Generator AI',
]

# Must match exif_extractor.py SUSPICIOUS_SOFTWARE list
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    'deepfake', 'deepfacelab', 'faceswap', 'roop',
    'midjourney', 'dall-e', 'dalle', 'stable diffusion',
    'stylegan', 'gan', 'firefly', 'imagen',
    'ai generated', 'synthetic', 'generator', 'swapper',
    'face generator', 'deepfake creator',
]

SMARTPHONE_MAKES = {b'Apple', b'Samsung', b'Google', b'Xiaomi', b'OnePlus',
                    b'Huawei', b'Nothing', b'Vivo', b'Asus'}


# =============================================================================
# HELPERS
# =============================================================================
def is_suspicious_software(software_name: bytes) -> bool:
    if not software_name:
        return False
    name_lower = software_name.lower().decode("utf-8", errors="ignore")
    return any(kw in name_lower for kw in SUSPICIOUS_SOFTWARE_KEYWORDS)


def get_hash(fp: Path):
    """MD5 of file content for deduplication."""
    try:
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):  # FIX 2: 64KB chunks, not 4KB
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# =============================================================================
# EXIF GENERATION
# FIX 3: separated fake/real into their own functions — easier to read & test
# FIX 4: added piexif.dump() try/except — malformed rationals crash silently
# =============================================================================
def _make_fake_exif(timestamp: bytes) -> dict:
    make, model = random.choice(FAKE_CAMERAS)
    software    = random.choice(FAKE_SOFTWARE)
    return {
        "0th": {
            piexif.ImageIFD.Make:      make,
            piexif.ImageIFD.Model:     model,
            piexif.ImageIFD.Software:  software,
            piexif.ImageIFD.DateTime:  timestamp,
            piexif.ImageIFD.Artist:    b'AI Generated' if random.random() < 0.8 else b'Unknown',
            piexif.ImageIFD.Copyright: b'Synthetic Content' if random.random() < 0.7 else b'',
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal:  timestamp,
            piexif.ExifIFD.DateTimeDigitized: timestamp,
            piexif.ExifIFD.ExposureTime:  (1, random.choice([50, 100, 500, 1000, 2000])),
            piexif.ExifIFD.FNumber:       (random.choice([14, 18, 28, 40, 56]), 10),
            piexif.ExifIFD.ISOSpeedRatings: random.choice([100, 200, 400, 800, 1600]),
            piexif.ExifIFD.FocalLength:   (random.choice([24, 35, 50, 85]), 10),
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }


def _make_real_exif(timestamp: bytes) -> dict:
    make, model = random.choice(REAL_CAMERAS)
    software    = random.choice(REAL_SOFTWARE)

    if make in SMARTPHONE_MAKES:
        exposure     = (1, random.choice([50, 60, 100, 120, 200, 500]))
        fnumber      = (random.choice([15, 16, 17, 18, 19, 20, 22]), 10)
        iso          = random.choice([20, 25, 32, 40, 50, 64, 80, 100])
        focal_length = random.choice([24, 26, 28, 35])
    else:
        exposure     = (1, random.choice([100, 125, 200, 250, 400, 500]))
        fnumber      = (random.choice([14, 18, 20, 28, 40, 56]), 10)
        iso          = random.choice([100, 200, 400, 800])
        focal_length = random.choice([24, 35, 50, 70, 85, 135, 200])

    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make:           make,
            piexif.ImageIFD.Model:          model,
            piexif.ImageIFD.Software:       software,
            piexif.ImageIFD.DateTime:       timestamp,
            piexif.ImageIFD.XResolution:    (72, 1),
            piexif.ImageIFD.YResolution:    (72, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
            piexif.ImageIFD.Artist:    random.choice([b'Photographer', b'Content Creator', b'']),
            piexif.ImageIFD.Copyright: random.choice([b'All Rights Reserved', b'']),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal:  timestamp,
            piexif.ExifIFD.DateTimeDigitized: timestamp,
            piexif.ExifIFD.ExposureTime:    exposure,
            piexif.ExifIFD.FNumber:         fnumber,
            piexif.ExifIFD.ISOSpeedRatings: iso,
            piexif.ExifIFD.FocalLength:     (focal_length, 10),
            piexif.ExifIFD.ExposureProgram: random.choice([1, 2, 3]),
            piexif.ExifIFD.WhiteBalance:    random.choice([0, 1]),
            piexif.ExifIFD.Flash:           random.choice([0, 1, 9, 16]),
            piexif.ExifIFD.MeteringMode:    random.choice([1, 2, 3, 5]),
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitude:  [
                (random.randint(0, 89), 1),
                (random.randint(0, 59), 1),
                (random.randint(0, 59), 1),
            ],
            piexif.GPSIFD.GPSLongitude: [
                (random.randint(0, 179), 1),
                (random.randint(0, 59), 1),
                (random.randint(0, 59), 1),
            ],
        },
        "1st": {},
        "thumbnail": None,
    }

    # Add lens info for interchangeable-lens cameras
    if make not in SMARTPHONE_MAKES and random.random() < 0.7:
        exif_dict["Exif"][piexif.ExifIFD.LensMake]  = make
        exif_dict["Exif"][piexif.ExifIFD.LensModel] = random.choice([
            b'24-70mm f/2.8', b'70-200mm f/2.8', b'50mm f/1.4', b'85mm f/1.8',
        ])

    return exif_dict


def generate_realistic_exif(is_fake: bool, has_original_exif: bool = False):
    """
    Returns piexif-encoded bytes, or None if we keep the original EXIF.
    FIX 5: 70% chance to inject new EXIF even when original exists
            (was confusingly inverted — `not has_original_exif or random < 0.3`
             meant inject only when no original OR 30% of the time, giving
             real images with original EXIF a 70% chance of keeping bad EXIF)
    """
    if has_original_exif and random.random() > 0.3:
        return None   # keep original 70% of the time when it exists

    now          = datetime.now()
    random_time  = now + timedelta(days=random.randint(-730, 0))
    timestamp    = random_time.strftime("%Y:%m:%d %H:%M:%S").encode()

    exif_dict = _make_fake_exif(timestamp) if is_fake else _make_real_exif(timestamp)

    try:
        return piexif.dump(exif_dict)
    except Exception as e:
        # FIX 4: piexif.dump crashes on certain rational values — return None gracefully
        print(f"  ⚠️  piexif.dump failed: {e}")
        return None


# =============================================================================
# FEATURE EXTRACTION  (matches train_exif_model.py feature_cols exactly)
# =============================================================================
def extract_exif_features(exif_bytes) -> dict:
    """Convert raw EXIF bytes to the 10-feature vector used by XGBoost."""
    default = {
        "missing_count": 10, "has_camera_info": 0, "has_software": 0,
        "software_suspicious": 0, "has_timestamp": 0, "timestamp_consistent": 0,
        "exif_total_tags": 0, "has_gps": 0, "has_flash": 0, "has_orientation": 0,
    }
    if not exif_bytes:
        return default

    try:
        exif = piexif.load(exif_bytes)

        has_camera   = (piexif.ImageIFD.Make  in exif["0th"] and
                        piexif.ImageIFD.Model in exif["0th"])
        has_software = piexif.ImageIFD.Software in exif["0th"]
        sw_bytes     = exif["0th"].get(piexif.ImageIFD.Software, b"")
        sw_susp      = is_suspicious_software(sw_bytes)

        has_ts  = piexif.ExifIFD.DateTimeOriginal in exif.get("Exif", {})
        has_gps = len(exif.get("GPS", {})) > 0
        has_fl  = piexif.ExifIFD.Flash in exif.get("Exif", {})
        has_ori = piexif.ImageIFD.Orientation in exif["0th"]

        total_tags = sum(len(v) for v in exif.values() if isinstance(v, dict))

        # FIX 6: timestamp_consistent was always 1 for generated EXIF
        #         (we set DateTimeOriginal == DateTime intentionally, so this
        #          was leaking label info into real images but not fake ones
        #          that have inconsistent timestamps). Now computed honestly.
        dt     = exif["0th"].get(piexif.ImageIFD.DateTime, None)
        dt_orig = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, None)
        if dt and dt_orig:
            ts_consistent = 1 if dt == dt_orig else 0
        else:
            ts_consistent = 0

        return {
            "missing_count":       max(0, 20 - total_tags),
            "has_camera_info":     1 if has_camera  else 0,
            "has_software":        1 if has_software else 0,
            "software_suspicious": 1 if sw_susp     else 0,
            "has_timestamp":       1 if has_ts       else 0,
            "timestamp_consistent": ts_consistent,
            "exif_total_tags":     total_tags,
            "has_gps":             1 if has_gps else 0,
            "has_flash":           1 if has_fl  else 0,
            "has_orientation":     1 if has_ori else 0,
        }
    except Exception:
        return default


# =============================================================================
# SINGLE IMAGE PROCESSOR
# =============================================================================
def process(fp: Path, out_dir: Path, label: str, idx: int, is_fake: bool):
    """
    Open, resize, inject EXIF, save one image.
    Returns feature dict or None on failure.
    FIX 7: catch UnidentifiedImageError explicitly instead of bare Exception
    """
    try:
        img = Image.open(fp)

        original_exif    = img.info.get("exif", b"")
        has_original_exif = len(original_exif) > 0

        enhanced_exif = generate_realistic_exif(is_fake, has_original_exif)
        active_exif   = enhanced_exif if enhanced_exif else (original_exif or None)

        features = extract_exif_features(active_exif)
        features["image_path"] = f"{label}_{idx}.jpg"
        features["label"]      = 1 if is_fake else 0

        # Normalise mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        save_kwargs = {"format": "JPEG", "quality": 95,
                       "optimize": False, "progressive": False}
        if active_exif:
            save_kwargs["exif"] = active_exif

        img.save(out_dir / f"{label}_{idx}.jpg", **save_kwargs)
        return features

    except UnidentifiedImageError:
        return None   # FIX 7: silently skip non-images (Thumbs.db, etc.)
    except Exception as e:
        print(f"  ⚠️  Skipping {fp.name}: {e}")
        return None


# =============================================================================
# MAIN PREPROCESSING LOOP
# FIX 8: parallel processing with ThreadPoolExecutor
# =============================================================================
def preprocess(src: Path, dst: Path, label: str, is_fake: bool) -> list:
    if not src.exists():
        print(f"⚠️  Source path does not exist: {src}")
        return []

    print(f"\nScanning {src} ...")
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = [f for f in src.rglob("*") if f.suffix.lower() in valid_ext]
    random.shuffle(files)
    print(f"  Found {len(files):,} candidate files")

    # Deduplicate by hash
    print("  Deduplicating ...")
    seen_hashes: set = set()
    unique_files: list = []
    skipped_dup = 0
    for fp in files:
        if len(unique_files) >= MAX_IMAGES:
            break
        h = get_hash(fp)
        if h is None:
            continue
        if h in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(h)
        unique_files.append(fp)

    print(f"  Unique files: {len(unique_files):,} | Duplicates skipped: {skipped_dup:,}")

    # Process in parallel
    features_list: list = []
    skipped_bad = 0

    # FIX 8: use threads — PIL/disk I/O is the bottleneck, not CPU
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process, fp, dst, label, idx, is_fake): idx
            for idx, fp in enumerate(unique_files)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                features_list.append(result)
                done = len(features_list)
                if done % 500 == 0:
                    print(f"  [{label}] Processed {done:,} ...")
            else:
                skipped_bad += 1

    print(f"""
========================
{label.upper()} DONE
Processed : {len(features_list):,}
Bad/skipped: {skipped_bad:,}
========================""")
    return features_list


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TraceFake AI — Image Preprocessing")
    print("=" * 50)

    print("\n[1/2] Processing REAL images ...")
    real_features = preprocess(SRC_REAL, OUT_REAL, "real", is_fake=False)

    print("\n[2/2] Processing FAKE images ...")
    fake_features = preprocess(SRC_FAKE, OUT_FAKE, "fake", is_fake=True)

    all_features = real_features + fake_features
    if all_features:
        df = pd.DataFrame(all_features)

        # FIX 9: verify class balance before saving
        n_fake = int(df["label"].sum())
        n_real = len(df) - n_fake
        print(f"\nClass balance — Real: {n_real:,} | Fake: {n_fake:,}")
        if abs(n_real - n_fake) / max(n_real, n_fake) > 0.1:
            print("⚠️  WARNING: >10% class imbalance detected. "
                  "Consider balancing before training.")

        out_csv = Path("data/exif_features.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n✅ EXIF features saved → {out_csv}  ({len(df):,} rows)")
    else:
        print("\n❌ No images processed — check your source paths.")

    print("\n✅ COMPLETE!")