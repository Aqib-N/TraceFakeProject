import random
import hashlib
import piexif
import pandas as pd
import requests
import time

from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Google Colab / Drive:
# SRC_REAL = Path("/content/drive/MyDrive/real_image_processed")
# SRC_FAKE = Path("/content/drive/MyDrive/fake_image_processed")

# Kaggle REAL dataset
SRC_REAL = Path("/kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq")

# Download FAKE images from ThisPersonDoesNotExist
SRC_FAKE = Path("/kaggle/working/fake_image_processed")
SRC_FAKE.mkdir(parents=True, exist_ok=True)

TOTAL_FAKE_IMAGES = 25000

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("Downloading fake AI faces...")

for i in range(TOTAL_FAKE_IMAGES):

    save_path = SRC_FAKE / f"fake_face_{i:05d}.jpg"

    # Skip existing files
    if save_path.exists():
        continue

    try:
        response = requests.get(
            "https://thispersondoesnotexist.com/image",
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            continue

        img = Image.open(BytesIO(response.content)).convert("RGB")

        img.save(save_path, "JPEG", quality=95)

        if i % 100 == 0:
            print(f"Downloaded {i} fake images")

        time.sleep(0.2)

    except Exception as e:
        print(f"Error {i}: {e}")

print("Fake dataset download complete.")

# Google Colab (uncomment to use)
# SRC_REAL = Path("/content/drive/MyDrive/real_image_processed")
# SRC_FAKE = Path("/content/drive/MyDrive/fake_image_processed")


# OUTPUT PATHS
OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")
OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = (224, 224)
MAX_IMAGES = 21300
NUM_WORKERS = 4


# Camera & software lists

REAL_CAMERAS = [
    (b'Canon',      b'EOS R5'),         (b'Canon',      b'EOS R6 Mark II'),
    (b'Canon',      b'EOS R8'),         (b'Canon',      b'EOS R50'),
    (b'Canon',      b'EOS 5D Mark IV'), (b'Canon',      b'EOS R3'),
    (b'Canon',      b'EOS R7'),         (b'Canon',      b'EOS 90D'),
    (b'Nikon',      b'Z9'),             (b'Nikon',      b'Z8'),
    (b'Nikon',      b'Z7 II'),          (b'Nikon',      b'Z6 II'),
    (b'Nikon',      b'Z5'),             (b'Nikon',      b'Zf'),
    (b'Nikon',      b'D850'),           (b'Nikon',      b'D780'),
    (b'Sony',       b'ILCE-1'),         (b'Sony',       b'ILCE-7RM5'),
    (b'Sony',       b'ILCE-7M4'),       (b'Sony',       b'ILCE-7CM2'),
    (b'Sony',       b'ILCE-6700'),      (b'Sony',       b'ILCE-9M3'),
    (b'Sony',       b'ILCE-7SM3'),      (b'Sony',       b'ZV-E1'),
    (b'FUJIFILM',   b'X-H2S'),          (b'FUJIFILM',   b'X-H2'),
    (b'FUJIFILM',   b'X-T5'),           (b'FUJIFILM',   b'X-S20'),
    (b'FUJIFILM',   b'X100VI'),         (b'FUJIFILM',   b'GFX 100 II'),
    (b'FUJIFILM',   b'X-T30 II'),
    (b'Panasonic',  b'DC-S5M2'),        (b'Panasonic',  b'DC-S5M2X'),
    (b'Panasonic',  b'DC-G9M2'),        (b'Panasonic',  b'DC-S1R'),
    (b'Panasonic',  b'DC-S5'),          (b'Panasonic',  b'Lumix GH6'),
    (b'Leica',      b'M11'),            (b'Leica',      b'Q3'),
    (b'Leica',      b'SL3'),
    (b'OM System',  b'OM-1 Mark II'),   (b'OM SYSTEM',  b'OM-5'),
    (b'Pentax',     b'K-3 Mark III'),   (b'RICOH',      b'GR IIIx'),
    (b'Hasselblad', b'X2D 100C'),       (b'Hasselblad', b'907X'),
    (b'Apple',      b'iPhone 15 Pro'),  (b'Apple',      b'iPhone 15 Pro Max'),
    (b'Apple',      b'iPhone 15'),      (b'Apple',      b'iPhone 14 Pro Max'),
    (b'Samsung',    b'Galaxy S24 Ultra'),(b'Samsung',   b'Galaxy S24+'),
    (b'Samsung',    b'Galaxy S23 Ultra'),(b'Samsung',   b'Galaxy Z Fold 5'),
    (b'Google',     b'Pixel 8 Pro'),    (b'Google',     b'Pixel 8'),
    (b'Google',     b'Pixel 7 Pro'),    (b'Xiaomi',     b'13 Ultra'),
    (b'Xiaomi',     b'14 Pro'),         (b'OnePlus',    b'12'),
    (b'OnePlus',    b'Open'),           (b'Huawei',     b'P60 Pro'),
    (b'Huawei',     b'Mate 60 Pro'),    (b'Nothing',    b'Phone 2'),
    (b'Vivo',       b'X100 Pro'),       (b'Asus',       b'Zenfone 10'),
]

FAKE_CAMERAS = [
    (b'Midjourney',    b'v6.1'),
    (b'Midjourney',    b'v6'),
    (b'Midjourney',    b'v5.2'),
    (b'OpenAI',        b'DALL-E 3'),
    (b'OpenAI',        b'DALL-E 2'),
    (b'Stability AI',  b'Stable Diffusion 3.5'),
    (b'Stability AI',  b'Stable Diffusion XL'),
    (b'Google',        b'Imagen 2'),
    (b'Meta',          b'Imagine'),
    (b'Adobe',         b'Firefly 2'),
    (b'Canva',         b'AI Generator'),
    (b'DeepFaceLab',   b'3.0.1'),
    (b'Roop',          b'Face Swapper'),
    (b'StyleGAN',      b'StyleGAN3'),
    (b'Unknown',       b'AI Generated'),
]

REAL_SOFTWARE = [
    b'Adobe Photoshop 2025',
    b'Adobe Photoshop 2024',
    b'Adobe Lightroom Classic 14.0',
    b'Adobe Lightroom Classic 13.0',
    b'Adobe Camera Raw 17.0',
    b'Capture One Pro 16.5',
    b'DxO PhotoLab 8',
    b'Luminar Neo 1.20',
    b'Affinity Photo 2.5',
    b'Darktable 4.8',
    b'GIMP 2.10.38',
    b'GIMP 3.0',
    b'Snapseed 2.0',
    b'Lightroom Mobile 9.0',
]

FAKE_SOFTWARE = [
    b'DeepFaceLab 3.0',
    b'Roop Face Swapper v1.3',
    b'Midjourney AI v6.1',
    b'DALL-E 3 Generator',
    b'Stable Diffusion 3.5',
    b'Adobe Firefly 2',
    b'StyleGAN3 Generator',
    b'AI Generated Content',
    b'Deepfake Creator Pro',
    b'Face Generator AI',
]

SUSPICIOUS_SOFTWARE_KEYWORDS = [
    'deepfake', 'deepface', 'faceswap', 'roop', 'midjourney',
    'dall-e', 'dalle', 'stable diffusion', 'stylegan', 'gan',
    'firefly', 'imagen', 'synthetic', 'generated', 'ai',
    'swapper', 'creator', 'generator', 'runway', 'pika',
    'leonardo',
]

SMARTPHONE_MAKES = {b'Apple', b'Samsung', b'Google', b'Xiaomi',
                    b'OnePlus', b'Huawei', b'Nothing', b'Vivo', b'Asus'}


# Helpers

def is_suspicious_software(software_name) -> bool:
    if not software_name:
        return False
    if isinstance(software_name, bytes):
        name_lower = software_name.decode('utf-8', errors='ignore').lower()
    else:
        name_lower = str(software_name).lower()
    return any(kw in name_lower for kw in SUSPICIOUS_SOFTWARE_KEYWORDS)


def is_plausible_timestamp(dt_string: str) -> bool:
    if not dt_string:
        return False
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_string.strip(), fmt)
            now = datetime.now()
            return 1990 < dt.year <= now.year and dt <= now
        except ValueError:
            continue
    return False


def _safe_ifd_len(ifd_value) -> int:
    """Return len() of an IFD dict, returning 0 if it is None or not a dict."""
    if ifd_value is None:
        return 0
    try:
        return len(ifd_value)
    except TypeError:
        return 0


# EXIF generation

def generate_realistic_exif(is_fake: bool, has_original_exif: bool = False):
    """Generate realistic EXIF bytes with modern device patterns."""

    if has_original_exif and random.random() >= 0.3:
        return None  # keep original EXIF

    now = datetime.now()
    random_days = random.randint(-730, 0)
    ts = (now + timedelta(days=random_days)).strftime("%Y:%m:%d %H:%M:%S").encode()

    if is_fake:
        make, model = random.choice(FAKE_CAMERAS)
        software    = random.choice(FAKE_SOFTWARE)
        exif_dict = {
            '0th': {
                piexif.ImageIFD.Make:      make,
                piexif.ImageIFD.Model:     model,
                piexif.ImageIFD.Software:  software,
                piexif.ImageIFD.DateTime:  ts,
                piexif.ImageIFD.Artist:    b'AI Generated' if random.random() < 0.8 else b'Unknown',
                piexif.ImageIFD.Copyright: b'Synthetic Content' if random.random() < 0.7 else b'',
            },
            'Exif': {
                piexif.ExifIFD.DateTimeOriginal:  ts,
                piexif.ExifIFD.DateTimeDigitized: ts,
                piexif.ExifIFD.ExposureTime:      (random.choice([1, 10, 100, 500, 1000]),
                                                   random.choice([50, 100, 500, 1000, 2000])),
                piexif.ExifIFD.FNumber:           (random.choice([1, 14, 2, 28, 4, 56, 8]), 10),
                piexif.ExifIFD.ISOSpeedRatings:   random.choice([100, 200, 400, 800, 1600, 3200, 6400]),
                piexif.ExifIFD.FocalLength:       (random.choice([24, 35, 50, 85]), 10),
            },
            'GPS': {},
        }
    else:
        make, model = random.choice(REAL_CAMERAS)
        software    = random.choice(REAL_SOFTWARE)

        if make in SMARTPHONE_MAKES:
            exposure     = (1, random.choice([50, 60, 100, 120, 200, 500]))
            fnumber      = (random.choice([15, 16, 17, 18, 19, 20, 22]), 10)
            iso          = random.choice([20, 25, 32, 40, 50, 64, 80, 100])
            focal_length = random.choice([24, 26, 28, 35])
        else:
            exposure     = (1, random.choice([100, 125, 160, 200, 250, 320, 400, 500]))
            fnumber      = (random.choice([14, 18, 2, 28, 4, 56, 8, 11, 16]), 10)
            iso          = random.choice([100, 125, 160, 200, 250, 320, 400, 500, 640, 800])
            focal_length = random.choice([24, 35, 50, 70, 85, 105, 135, 200])

        exif_dict = {
            '0th': {
                piexif.ImageIFD.Make:           make,
                piexif.ImageIFD.Model:          model,
                piexif.ImageIFD.Software:       software,
                piexif.ImageIFD.DateTime:       ts,
                piexif.ImageIFD.Artist:         random.choice([b'Professional Photographer', b'Content Creator', b'']),
                piexif.ImageIFD.Copyright:      random.choice([b'All Rights Reserved', b'Creative Commons', b'']),
                piexif.ImageIFD.XResolution:    (72, 1),
                piexif.ImageIFD.YResolution:    (72, 1),
                piexif.ImageIFD.ResolutionUnit: 2,
            },
            'Exif': {
                piexif.ExifIFD.DateTimeOriginal:  ts,
                piexif.ExifIFD.DateTimeDigitized: ts,
                piexif.ExifIFD.ExposureTime:      exposure,
                piexif.ExifIFD.FNumber:           fnumber,
                piexif.ExifIFD.ISOSpeedRatings:   iso,
                piexif.ExifIFD.FocalLength:       (focal_length, 10),
                piexif.ExifIFD.ExposureProgram:   random.choice([1, 2, 3]),
                piexif.ExifIFD.WhiteBalance:      random.choice([0, 1]),
                piexif.ExifIFD.Flash:             random.choice([0, 1, 9, 16]),
                piexif.ExifIFD.MeteringMode:      random.choice([1, 2, 3, 4, 5]),
            },
            'GPS': {
                piexif.GPSIFD.GPSLatitude:  [(random.randint(0, 90), 1),  (random.randint(0, 59), 1), (random.randint(0, 59), 1)],
                piexif.GPSIFD.GPSLongitude: [(random.randint(0, 180), 1), (random.randint(0, 59), 1), (random.randint(0, 59), 1)],
            },
        }

        if make not in SMARTPHONE_MAKES and random.random() < 0.7:
            exif_dict['Exif'][piexif.ExifIFD.LensMake]  = make
            exif_dict['Exif'][piexif.ExifIFD.LensModel] = random.choice([
                b'24-70mm f/2.8', b'70-200mm f/2.8', b'50mm f/1.4', b'85mm f/1.8',
            ])

    try:
        return piexif.dump(exif_dict)
    except Exception as e:
        print(f"piexif.dump error: {e}")
        return None


# Feature extraction (synced with exif_extractor.py)

_NULL_FEATURES = {
    'missing_count':        10,
    'has_camera_info':      0,
    'has_software':         0,
    'software_suspicious':  0,
    'has_timestamp':        0,
    'timestamp_consistent': 0,
    'timestamp_plausible':  0,
    'timestamp_future':     0,
    'exif_total_tags':      0,
    'has_gps':              0,
    'has_flash':            0,
    'has_orientation':      0,
}


def extract_exif_features(exif_bytes) -> dict:
    """
    Extract features from raw EXIF bytes.
    Output columns match EXIF_FEATURE_COLS in config.py exactly (12 columns).

    FIX: piexif.load() can return None for any IFD (e.g. GPS, Interop).
         All IFD accesses now go through dict.get() with a fallback of {},
         and tag counting uses _safe_ifd_len() which handles None gracefully.
    """
    if not exif_bytes:
        return dict(_NULL_FEATURES)

    try:
        exif = piexif.load(exif_bytes)

        # ── Safely retrieve each IFD, defaulting to {} when None ────────────
        ifd_0th   = exif.get('0th')   or {}
        ifd_exif  = exif.get('Exif')  or {}
        ifd_gps   = exif.get('GPS')   or {}

        # ── Camera / software ────────────────────────────────────────────────
        has_camera  = (piexif.ImageIFD.Make  in ifd_0th and
                       piexif.ImageIFD.Model in ifd_0th)
        has_software = piexif.ImageIFD.Software in ifd_0th

        software_val = ifd_0th.get(piexif.ImageIFD.Software, b'')
        if isinstance(software_val, bytes):
            software_str = software_val.decode('utf-8', errors='ignore')
        else:
            software_str = str(software_val)
        suspicious = is_suspicious_software(software_str)

        # ── Presence flags ───────────────────────────────────────────────────
        has_ts     = piexif.ExifIFD.DateTimeOriginal in ifd_exif
        has_gps    = len(ifd_gps) > 0
        has_flash  = piexif.ExifIFD.Flash       in ifd_exif
        has_orient = piexif.ImageIFD.Orientation in ifd_0th

        # ── Total tag count (FIX: _safe_ifd_len handles None IFDs) ──────────
        total_tags = sum(_safe_ifd_len(v) for v in exif.values())

        # ── Timestamp plausibility ───────────────────────────────────────────
        raw_ts = ifd_0th.get(piexif.ImageIFD.DateTime, b'')
        if isinstance(raw_ts, bytes):
            ts_str = raw_ts.decode('utf-8', errors='ignore')
        else:
            ts_str = str(raw_ts)

        plausible = is_plausible_timestamp(ts_str)
        future    = False
        if ts_str:
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt     = datetime.strptime(ts_str.strip(), fmt)
                    future = dt > datetime.now()
                    break
                except ValueError:
                    continue

        return {
            'missing_count':        max(0, 20 - total_tags),
            'has_camera_info':      int(has_camera),
            'has_software':         int(has_software),
            'software_suspicious':  int(suspicious),
            'has_timestamp':        int(has_ts),
            'timestamp_consistent': 1,          # placeholder; multi-tag check omitted
            'timestamp_plausible':  int(plausible),
            'timestamp_future':     int(future),
            'exif_total_tags':      total_tags,
            'has_gps':              int(has_gps),
            'has_flash':            int(has_flash),
            'has_orientation':      int(has_orient),
        }

    except Exception as e:
        print(f"extract_exif_features error: {e}")
        return dict(_NULL_FEATURES)


# File hashing

def get_hash(fp) -> str | None:
    try:
        h = hashlib.md5()
        with open(fp, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# Single image processing

def process(fp, out_dir, label, idx, is_fake) -> dict | None:
    try:
        img = Image.open(fp)

        original_exif     = img.info.get('exif', b'')
        has_original_exif = len(original_exif) > 0

        enhanced_exif = generate_realistic_exif(is_fake, has_original_exif)
        active_exif   = enhanced_exif if enhanced_exif else original_exif

        features = extract_exif_features(active_exif)
        features['image_path'] = f"{label}_{idx}.jpg"
        features['label']      = 1 if is_fake else 0  # 1=FAKE, 0=REAL

        if img.mode not in ('RGB',):
            img = img.convert('RGB')

        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        save_kwargs = {'format': 'JPEG', 'quality': 95,
                       'optimize': False, 'progressive': False}
        if active_exif:
            save_kwargs['exif'] = active_exif

        img.save(out_dir / f"{label}_{idx}.jpg", **save_kwargs)
        return features

    except Exception:
        return None


# Batch preprocessing

def preprocess(src: Path, dst: Path, label: str, is_fake: bool) -> list:
    if not src.exists():
        print(f"⚠️  Source path does not exist: {src}")
        return []

    print(f"\nScanning {src} for images...")
    files = list(src.rglob("*"))
    random.shuffle(files)

    valid_ext     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    seen_hashes   = set()
    features_list = []
    count = skipped_dup = skipped_bad = 0

    for fp in files:
        if len(seen_hashes) >= MAX_IMAGES:
            break
        if fp.suffix.lower() not in valid_ext:
            continue

        h = get_hash(fp)
        if h is None:
            skipped_bad += 1
            continue
        if h in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(h)

        result = process(fp, dst, label, count, is_fake)
        if result:
            features_list.append(result)
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count} {label} images...")

    print(f"""
========================
{label.upper()} DONE
Processed         : {count}
Duplicates skipped: {skipped_dup}
Bad files skipped : {skipped_bad}
========================""")
    return features_list


# Entry point

if __name__ == "__main__":
    print("=" * 50)
    print("Processing REAL images...")
    print("=" * 50)
    real_features = preprocess(SRC_REAL, OUT_REAL, "real", is_fake=False)

    print("\n" + "=" * 50)
    print("Processing FAKE images...")
    print("=" * 50)
    fake_features = preprocess(SRC_FAKE, OUT_FAKE, "fake", is_fake=True)

    all_features = real_features + fake_features
    if all_features:
        df = pd.DataFrame(all_features)

        # Verify all expected columns are present
        from config import EXIF_FEATURE_COLS
        expected = EXIF_FEATURE_COLS + ['image_path', 'label']
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")

        df.to_csv("data/metadata.csv", index=False)
        print(f"\n✅ Metadata saved to data/metadata.csv")
        print(f"   Total samples : {len(df)}")
        print(f"   Fake (label=1): {df['label'].sum()}")
        print(f"   Real (label=0): {len(df) - df['label'].sum()}")
        print(f"   Columns       : {list(df.columns)}")

    print("\n✅ COMPLETE!")