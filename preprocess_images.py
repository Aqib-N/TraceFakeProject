import random
import hashlib
import piexif
import pandas as pd

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # FIX: was parents[2]

# ── Source paths (edit for your environment) ──────────────────────────────────
# Kaggle:
SRC_REAL = Path("/kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
SRC_FAKE = Path("/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train/fake")

# Google Colab (uncomment):
# SRC_REAL = Path("/content/drive/MyDrive/real_image_processed")
# SRC_FAKE = Path("/content/drive/MyDrive/fake_image_processed")

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")
OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = (224, 224)
MAX_IMAGES = 21300

# ── EXIF strategy ─────────────────────────────────────────────────────────────
# "strip"  → remove all EXIF from both classes (safest, no leakage)
# "inject" → inject realistic camera EXIF into fake images (harder task)
EXIF_STRATEGY = "strip"


# ── Camera lists ──────────────────────────────────────────────────────────────
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

# FIX: removed GIMP — it's in SUSPICIOUS_SOFTWARE_TERMS in config.py,
# so real images labelled with GIMP software would be flagged as fake signal.
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
    b'Snapseed 2.0',
    b'Lightroom Mobile 9.0',
]

SMARTPHONE_MAKES = {b'Apple', b'Samsung', b'Google', b'Xiaomi',
                    b'OnePlus', b'Huawei', b'Nothing', b'Vivo', b'Asus'}


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_plausible_timestamp(dt_string: str) -> bool:
    if not dt_string:
        return False
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt  = datetime.strptime(dt_string.strip(), fmt)
            now = datetime.now()
            return 1990 < dt.year <= now.year and dt <= now
        except ValueError:
            continue
    return False


def _safe_ifd_len(ifd_value) -> int:
    if ifd_value is None:
        return 0
    try:
        return len(ifd_value)
    except TypeError:
        return 0


# ── EXIF generation ───────────────────────────────────────────────────────────
def generate_realistic_exif(has_original_exif: bool = False):
    """
    Generate realistic camera EXIF (real-camera data only — no fake keywords).
    Used for BOTH real and fake images when EXIF_STRATEGY == "inject",
    so the EXIF model cannot cheat by detecting generator software names.
    """
    if has_original_exif and random.random() >= 0.3:
        return None  # keep original EXIF 70% of the time

    now         = datetime.now()
    random_days = random.randint(-730, 0)
    ts          = (now + timedelta(days=random_days)).strftime("%Y:%m:%d %H:%M:%S").encode()

    make, model = random.choice(REAL_CAMERAS)
    software    = random.choice(REAL_SOFTWARE)

    if make in SMARTPHONE_MAKES:
        exposure     = (1, random.choice([50, 60, 100, 120, 200, 500]))
        fnumber      = (random.choice([15, 16, 17, 18, 19, 20, 22]), 10)
        iso          = random.choice([20, 25, 32, 40, 50, 64, 80, 100])
        focal_length = random.choice([24, 26, 28, 35])
    else:
        exposure     = (1, random.choice([100, 125, 160, 200, 250, 320, 400, 500]))
        fnumber      = (random.choice([14, 18, 20, 28, 40, 56, 80, 110, 160]), 10)
        iso          = random.choice([100, 125, 160, 200, 250, 320, 400, 500, 640, 800])
        focal_length = random.choice([24, 35, 50, 70, 85, 105, 135, 200])

    exif_dict = {
        '0th': {
            piexif.ImageIFD.Make:           make,
            piexif.ImageIFD.Model:          model,
            piexif.ImageIFD.Software:       software,
            piexif.ImageIFD.DateTime:       ts,
            piexif.ImageIFD.Artist:         random.choice(
                [b'Professional Photographer', b'Content Creator', b'']
            ),
            piexif.ImageIFD.Copyright:      random.choice(
                [b'All Rights Reserved', b'Creative Commons', b'']
            ),
            piexif.ImageIFD.XResolution:    (72, 1),
            piexif.ImageIFD.YResolution:    (72, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
            piexif.ImageIFD.Orientation:    random.choice([1, 3, 6, 8]),
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
            piexif.GPSIFD.GPSLatitude:  [
                (random.randint(0, 90),  1),
                (random.randint(0, 59), 1),
                (random.randint(0, 59), 1),
            ],
            piexif.GPSIFD.GPSLongitude: [
                (random.randint(0, 180), 1),
                (random.randint(0, 59),  1),
                (random.randint(0, 59),  1),
            ],
        },
    }

    if make not in SMARTPHONE_MAKES and random.random() < 0.7:
        exif_dict['Exif'][piexif.ExifIFD.LensMake]  = make
        exif_dict['Exif'][piexif.ExifIFD.LensModel] = random.choice([
            b'24-70mm f/2.8', b'70-200mm f/2.8',
            b'50mm f/1.4',    b'85mm f/1.8',
        ])

    try:
        return piexif.dump(exif_dict)
    except Exception as e:
        print(f"piexif.dump error: {e}")
        return None


# ── Feature extraction ────────────────────────────────────────────────────────
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

SUSPICIOUS_SOFTWARE_KEYWORDS = [
    'deepfake', 'deepface', 'faceswap', 'roop', 'midjourney',
    'dall-e', 'dalle', 'stable diffusion', 'stylegan', 'gan',
    'firefly', 'imagen', 'synthetic', 'generated', 'ai',
    'swapper', 'creator', 'generator', 'runway', 'pika', 'leonardo',
]


def _is_suspicious(software_bytes) -> bool:
    if not software_bytes:
        return False
    name = (software_bytes.decode('utf-8', errors='ignore')
            if isinstance(software_bytes, bytes) else str(software_bytes)).lower()
    return any(kw in name for kw in SUSPICIOUS_SOFTWARE_KEYWORDS)


def extract_exif_features(exif_bytes) -> dict:
    """
    Extract the 12 canonical EXIF features matching EXIF_FEATURE_COLS.
    FIX: timestamp_consistent now properly compared (was hardcoded 1).
    """
    if not exif_bytes:
        return dict(_NULL_FEATURES)

    try:
        exif     = piexif.load(exif_bytes)
        ifd_0th  = exif.get('0th')  or {}
        ifd_exif = exif.get('Exif') or {}
        ifd_gps  = exif.get('GPS')  or {}

        has_camera  = (piexif.ImageIFD.Make  in ifd_0th and
                       piexif.ImageIFD.Model in ifd_0th)
        has_software = piexif.ImageIFD.Software in ifd_0th
        software_val = ifd_0th.get(piexif.ImageIFD.Software, b'')
        suspicious   = _is_suspicious(software_val)

        has_ts      = piexif.ExifIFD.DateTimeOriginal in ifd_exif
        has_gps     = len(ifd_gps) > 0
        has_flash   = piexif.ExifIFD.Flash       in ifd_exif
        has_orient  = piexif.ImageIFD.Orientation in ifd_0th
        total_tags  = sum(_safe_ifd_len(v) for v in exif.values())

        # Timestamp fields
        raw_dt  = ifd_0th.get(piexif.ImageIFD.DateTime, b'')
        raw_dto = ifd_exif.get(piexif.ExifIFD.DateTimeOriginal, b'')

        def _dec(b):
            return b.decode('utf-8', errors='ignore') if isinstance(b, bytes) else str(b)

        ts_dt  = _dec(raw_dt)
        ts_dto = _dec(raw_dto)

        # FIX: actually compare the two timestamp fields
        ts_consistent = int(bool(ts_dt and ts_dto and ts_dt == ts_dto))
        ts_plausible  = int(is_plausible_timestamp(ts_dto or ts_dt))

        future = False
        primary_ts = ts_dto or ts_dt
        if primary_ts:
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    future = datetime.strptime(primary_ts.strip(), fmt) > datetime.now()
                    break
                except ValueError:
                    continue

        return {
            'missing_count':        max(0, 20 - total_tags),
            'has_camera_info':      int(has_camera),
            'has_software':         int(has_software),
            'software_suspicious':  int(suspicious),
            'has_timestamp':        int(has_ts),
            'timestamp_consistent': ts_consistent,
            'timestamp_plausible':  ts_plausible,
            'timestamp_future':     int(future),
            'exif_total_tags':      total_tags,
            'has_gps':              int(has_gps),
            'has_flash':            int(has_flash),
            'has_orientation':      int(has_orient),
        }

    except Exception as e:
        print(f"extract_exif_features error: {e}")
        return dict(_NULL_FEATURES)


# ── File hashing ──────────────────────────────────────────────────────────────
def get_hash(fp) -> str | None:
    try:
        h = hashlib.md5()
        with open(fp, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ── Single image processing ───────────────────────────────────────────────────
def process(fp, out_dir, label, idx, is_fake) -> dict | None:
    try:
        img = Image.open(fp)

        if EXIF_STRATEGY == "strip":
            # FIX option A: strip all EXIF → no leakage possible
            active_exif = b''
        else:
            # FIX option B: inject real-camera EXIF into both classes
            original_exif     = img.info.get('exif', b'')
            has_original_exif = len(original_exif) > 0
            generated         = generate_realistic_exif(has_original_exif)
            active_exif       = generated if generated else original_exif

        features               = extract_exif_features(active_exif)
        features['image_path'] = f"{label}_{idx}.jpg"

        # FIX: label convention matches Keras class_indices
        # classes=["fake","real"] → fake=0, real=1
        features['label'] = 0 if is_fake else 1   # FIX: was inverted (1=FAKE)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        save_kwargs = {'format': 'JPEG', 'quality': 95,
                       'optimize': False, 'progressive': False}
        if active_exif:
            save_kwargs['exif'] = active_exif

        img.save(out_dir / f"{label}_{idx}.jpg", **save_kwargs)
        return features

    except UnidentifiedImageError:
        print(f"  Unreadable image: {fp.name}")
        return None
    except Exception as e:
        # FIX: was silent (bare except: return None) — now visible
        print(f"  process() error on {fp.name}: {e}")
        return None


# ── Batch preprocessing ───────────────────────────────────────────────────────
def preprocess(src: Path, dst: Path, label: str, is_fake: bool) -> list:
    if not src.exists():
        print(f"⚠️  Source path does not exist: {src}")
        return []

    print(f"\nScanning {src}...")
    files     = list(src.rglob("*"))
    random.shuffle(files)

    valid_ext     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    seen_hashes   = set()
    features_list = []
    count = skipped_dup = skipped_bad = 0

    for fp in files:
        if count >= MAX_IMAGES:
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
            if count % 500 == 0:
                print(f"  {count}/{MAX_IMAGES} {label} images processed...")
        else:
            skipped_bad += 1

    print(f"""
========================
{label.upper()} DONE
Processed         : {count}
Duplicates skipped: {skipped_dup}
Bad files skipped : {skipped_bad}
========================""")
    return features_list


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(f"EXIF strategy: {EXIF_STRATEGY}")
    print("=" * 55)

    print("\nProcessing REAL images...")
    real_features = preprocess(SRC_REAL, OUT_REAL, "real", is_fake=False)

    print("\nProcessing FAKE images...")
    fake_features = preprocess(SRC_FAKE, OUT_FAKE, "fake", is_fake=True)

    all_features = real_features + fake_features
    if all_features:
        df = pd.DataFrame(all_features)

        # FIX: wrapped config import so a missing config.py doesn't kill the run
        try:
            from config import EXIF_FEATURE_COLS
            expected     = EXIF_FEATURE_COLS + ['image_path', 'label']
            missing_cols = [c for c in expected if c not in df.columns]
            if missing_cols:
                print(f"⚠️  Missing columns in output: {missing_cols}")
        except ImportError:
            print("⚠️  config.py not found — skipping column validation")

        df.to_csv("data/metadata.csv", index=False)
        print(f"\n✅ Metadata saved → data/metadata.csv")
        print(f"   Total samples : {len(df)}")
        print(f"   FAKE (label=0): {(df['label'] == 0).sum()}")
        print(f"   REAL (label=1): {(df['label'] == 1).sum()}")
        print(f"   Columns       : {list(df.columns)}")

        # Leakage check
        print("\nLeakage check — feature correlation with label:")
        try:
            from config import EXIF_FEATURE_COLS as COLS
        except ImportError:
            COLS = [c for c in df.columns if c not in ('image_path', 'label')]
        for col in COLS:
            corr = df[col].corr(df['label'])
            flag = " ← ⚠️ LEAKY" if abs(corr) > 0.8 else ""
            print(f"  {col:30s}: {corr:+.3f}{flag}")

    print("\n✅ COMPLETE!")