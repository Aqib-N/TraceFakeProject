import os
import sys
import random
import piexif
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import DATA_DIR, METADATA_CSV, EXIF_FEATURE_COLS
except ImportError:
    DATA_DIR      = Path("data/processed")
    METADATA_CSV  = Path("data/metadata.csv")
    EXIF_FEATURE_COLS = [
        "missing_count", "has_camera_info", "has_software",
        "software_suspicious", "has_timestamp", "timestamp_consistent",
        "timestamp_plausible", "timestamp_future",
        "exif_total_tags", "has_gps", "has_flash", "has_orientation",
    ]

FAKE_DIR = DATA_DIR / "fake"
REAL_DIR = DATA_DIR / "real"


# ── Camera / software pools ───────────────────────────────────────────────────
REAL_CAMERAS = [
    (b'Canon',     b'EOS R5'),          (b'Canon',     b'EOS R6 Mark II'),
    (b'Canon',     b'EOS 5D Mark IV'),  (b'Canon',     b'EOS R8'),
    (b'Nikon',     b'Z8'),              (b'Nikon',     b'Z6 II'),
    (b'Nikon',     b'D850'),            (b'Nikon',     b'Zf'),
    (b'Sony',      b'ILCE-7RM5'),       (b'Sony',      b'ILCE-7M4'),
    (b'Sony',      b'ILCE-6700'),       (b'Sony',      b'ZV-E1'),
    (b'FUJIFILM',  b'X-T5'),            (b'FUJIFILM',  b'X-H2'),
    (b'FUJIFILM',  b'X100VI'),          (b'Panasonic',  b'DC-S5M2'),
    (b'Apple',     b'iPhone 15 Pro'),   (b'Apple',     b'iPhone 15'),
    (b'Apple',     b'iPhone 14 Pro'),   (b'Samsung',   b'Galaxy S24 Ultra'),
    (b'Samsung',   b'Galaxy S23 Ultra'),(b'Google',    b'Pixel 8 Pro'),
    (b'Google',    b'Pixel 7 Pro'),     (b'Xiaomi',    b'13 Ultra'),
]

# No suspicious keywords — Lightroom/CaptureOne are editing tools,
# NOT in SUSPICIOUS_SOFTWARE_TERMS in config.py
REAL_SOFTWARE = [
    b'Adobe Lightroom Classic 14.0',
    b'Adobe Lightroom Classic 13.0',
    b'Adobe Camera Raw 17.0',
    b'Capture One Pro 16.5',
    b'DxO PhotoLab 8',
    b'Affinity Photo 2.5',
    b'Darktable 4.8',
    b'Snapseed 2.0',
]

SMARTPHONE_MAKES = {b'Apple', b'Samsung', b'Google', b'Xiaomi'}


def _random_ts(days_back_min=0, days_back_max=2190) -> bytes:
    """Random plausible timestamp between now and 6 years ago."""
    offset = random.randint(days_back_min, days_back_max)
    dt     = datetime.now() - timedelta(days=offset)
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode()


def _future_ts() -> bytes:
    """Timestamp 1–365 days in the future (fake signal)."""
    dt = datetime.now() + timedelta(days=random.randint(1, 365))
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode()


# ── EXIF builders ─────────────────────────────────────────────────────────────

def build_real_exif() -> bytes:
    """
    Full realistic camera EXIF for real images.
    DateTime == DateTimeOriginal (consistent camera behavior).
    """
    make, model = random.choice(REAL_CAMERAS)
    software    = random.choice(REAL_SOFTWARE)
    ts          = _random_ts(0, 2190)

    is_phone = make in SMARTPHONE_MAKES
    if is_phone:
        exposure = (1, random.choice([50, 60, 100, 125, 250, 500]))
        fnumber  = (random.choice([17, 18, 20, 22, 24]), 10)
        iso      = random.choice([20, 25, 32, 50, 64, 100, 125])
        focal    = random.choice([13, 16, 24, 28])
    else:
        exposure = (1, random.choice([100, 125, 160, 200, 250, 400, 500]))
        fnumber  = (random.choice([14, 18, 20, 28, 40, 56, 80, 110]), 10)
        iso      = random.choice([100, 125, 200, 320, 400, 640, 800])
        focal    = random.choice([24, 35, 50, 70, 85, 105, 135, 200])

    exif = {
        '0th': {
            piexif.ImageIFD.Make:           make,
            piexif.ImageIFD.Model:          model,
            piexif.ImageIFD.Software:       software,
            piexif.ImageIFD.DateTime:       ts,      # consistent with Exif.DTO
            piexif.ImageIFD.Orientation:    random.choice([1, 3, 6, 8]),
            piexif.ImageIFD.XResolution:    (72, 1),
            piexif.ImageIFD.YResolution:    (72, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        },
        'Exif': {
            piexif.ExifIFD.DateTimeOriginal:  ts,    # same as DateTime = consistent
            piexif.ExifIFD.DateTimeDigitized: ts,
            piexif.ExifIFD.ExposureTime:      exposure,
            piexif.ExifIFD.FNumber:           fnumber,
            piexif.ExifIFD.ISOSpeedRatings:   iso,
            piexif.ExifIFD.FocalLength:       (focal, 1),
            piexif.ExifIFD.Flash:             random.choice([0, 1, 9, 16]),
            piexif.ExifIFD.WhiteBalance:      random.choice([0, 1]),
            piexif.ExifIFD.MeteringMode:      random.choice([1, 2, 3, 5]),
            piexif.ExifIFD.ExposureProgram:   random.choice([1, 2, 3]),
        },
        'GPS': {} if random.random() > 0.60 else {   # GPS 60% of the time
            piexif.GPSIFD.GPSLatitude:  [(random.randint(0, 89), 1),
                                         (random.randint(0, 59), 1),
                                         (random.randint(0, 59), 1)],
            piexif.GPSIFD.GPSLongitude: [(random.randint(0, 179), 1),
                                         (random.randint(0, 59),  1),
                                         (random.randint(0, 59),  1)],
            piexif.GPSIFD.GPSLatitudeRef:  b'N',
            piexif.GPSIFD.GPSLongitudeRef: b'E',
        },
    }
    try:
        return piexif.dump(exif)
    except Exception:
        return b''


def build_fake_exif() -> bytes:
    """
    Sparse / inconsistent EXIF for fake images.
    Mimics how AI image generators save files (no camera hardware).

    4 scenarios, weighted:
      40% → completely empty (no EXIF at all)
      30% → minimal fields only, no Make/Model
      20% → inconsistent timestamps
      10% → future timestamp
    """
    roll = random.random()

    if roll < 0.40:
        # Completely empty — GAN output straight from generator
        return b''

    elif roll < 0.70:
        # Minimal — only generic software tag, no camera info
        try:
            return piexif.dump({
                '0th': {
                    piexif.ImageIFD.Software: random.choice([
                        b'Adobe Photoshop 2024',   # editing software, not camera
                        b'GIMP 2.10',
                        b'Paint.NET 5.0',
                        b'IrfanView 4.62',
                    ]),
                    piexif.ImageIFD.Orientation: 1,
                },
                'Exif': {},
                'GPS':  {},
            })
        except Exception:
            return b''

    elif roll < 0.90:
        # Inconsistent timestamps — DateTime != DateTimeOriginal
        ts1 = _random_ts(0,    365)   # recent
        ts2 = _random_ts(365, 1825)   # much older
        try:
            return piexif.dump({
                '0th': {
                    piexif.ImageIFD.DateTime:    ts1,   # different from DTO
                    piexif.ImageIFD.Orientation: 1,
                },
                'Exif': {
                    piexif.ExifIFD.DateTimeOriginal: ts2,  # inconsistent
                    piexif.ExifIFD.Flash: 0,
                },
                'GPS': {},
            })
        except Exception:
            return b''

    else:
        # Future timestamp — strong fake signal
        ts = _future_ts()
        try:
            return piexif.dump({
                '0th': {
                    piexif.ImageIFD.DateTime:    ts,
                    piexif.ImageIFD.Orientation: 1,
                },
                'Exif': {
                    piexif.ExifIFD.DateTimeOriginal: ts,
                    piexif.ExifIFD.Flash: 0,
                },
                'GPS': {},
            })
        except Exception:
            return b''


# ── Feature extraction (matches EXIF_FEATURE_COLS exactly) ────────────────────

def _safe_len(v) -> int:
    try:
        return len(v) if v else 0
    except TypeError:
        return 0


def extract_features(exif_bytes: bytes) -> dict:
    null = {c: 0 for c in EXIF_FEATURE_COLS}
    null['missing_count'] = 10

    if not exif_bytes:
        return null

    try:
        exif     = piexif.load(exif_bytes)
        ifd_0th  = exif.get('0th')  or {}
        ifd_exif = exif.get('Exif') or {}
        ifd_gps  = exif.get('GPS')  or {}

        total_tags = sum(_safe_len(v) for v in exif.values())

        raw_dt  = ifd_0th.get(piexif.ImageIFD.DateTime, b'')
        raw_dto = ifd_exif.get(piexif.ExifIFD.DateTimeOriginal, b'')

        def dec(b):
            return b.decode('utf-8', errors='ignore') if isinstance(b, bytes) else str(b)

        ts_dt  = dec(raw_dt)
        ts_dto = dec(raw_dto)

        # Plausibility check
        def plausible(s):
            if not s:
                return False
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s.strip(), fmt)
                    return 1990 < dt.year <= datetime.now().year and dt <= datetime.now()
                except ValueError:
                    continue
            return False

        # Future check
        future = False
        primary = ts_dto or ts_dt
        if primary:
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    future = datetime.strptime(primary.strip(), fmt) > datetime.now()
                    break
                except ValueError:
                    continue

        sw  = ifd_0th.get(piexif.ImageIFD.Software, b'')
        swl = (sw.decode('utf-8', errors='ignore') if isinstance(sw, bytes)
               else str(sw)).lower()
        suspicious_kws = [
            'photoshop', 'gimp', 'paint', 'fake', 'gan', 'deepfake',
            'stable diffusion', 'midjourney', 'dall-e', 'generator',
            'firefly', 'imagen', 'runway', 'pika', 'leonardo',
        ]

        return {
            'missing_count':        max(0, 20 - total_tags),
            'has_camera_info':      int(piexif.ImageIFD.Make  in ifd_0th and
                                        piexif.ImageIFD.Model in ifd_0th),
            'has_software':         int(piexif.ImageIFD.Software in ifd_0th),
            'software_suspicious':  int(any(k in swl for k in suspicious_kws)),
            'has_timestamp':        int(bool(ts_dt or ts_dto)),
            'timestamp_consistent': int(bool(ts_dt and ts_dto and ts_dt == ts_dto)),
            'timestamp_plausible':  int(plausible(primary)),
            'timestamp_future':     int(future),
            'exif_total_tags':      total_tags,
            'has_gps':              int(len(ifd_gps) > 0),
            'has_flash':            int(piexif.ExifIFD.Flash in ifd_exif),
            'has_orientation':      int(piexif.ImageIFD.Orientation in ifd_0th),
        }
    except Exception as e:
        print(f"  extract_features error: {e}")
        return null


# ── Inject into one folder ────────────────────────────────────────────────────

def inject_folder(folder: Path, is_fake: bool, label: int) -> list:
    files  = sorted(folder.glob("*.jpg"))
    cls    = "FAKE" if is_fake else "REAL"
    total  = len(files)
    rows   = []
    errors = 0

    print(f"\n[{cls}] Injecting EXIF into {total:,} images...")

    for i, fp in enumerate(files):
        try:
            img = Image.open(fp).convert("RGB")

            exif_bytes = build_fake_exif() if is_fake else build_real_exif()
            features   = extract_features(exif_bytes)

            save_kwargs = {'format': 'JPEG', 'quality': 95, 'optimize': False}
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes

            img.save(fp, **save_kwargs)   # overwrite in-place

            features['image_path'] = fp.name
            features['label']      = label
            rows.append(features)

            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{total:,} done...")

        except Exception as e:
            print(f"  Error on {fp.name}: {e}")
            errors += 1

    print(f"  ✅ {total - errors:,} injected | {errors} errors")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("EXIF INJECTION — TraceFake AI")
    print("Injecting into existing data/processed/fake & real")
    print("=" * 58)

    if not FAKE_DIR.exists() or not REAL_DIR.exists():
        print(f"❌ data/processed/fake or real not found.")
        print(f"   Run preprocess_images.py first.")
        sys.exit(1)

    fake_rows = inject_folder(FAKE_DIR, is_fake=True,  label=0)
    real_rows = inject_folder(REAL_DIR, is_fake=False, label=1)

    all_rows = fake_rows + real_rows
    if not all_rows:
        print("❌ No rows generated.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_CSV, index=False)

    print(f"\n✅ metadata.csv saved → {METADATA_CSV}")
    print(f"   Total   : {len(df):,}")
    print(f"   FAKE (0): {(df['label'] == 0).sum():,}")
    print(f"   REAL (1): {(df['label'] == 1).sum():,}")

    # ── Leakage + signal report ───────────────────────────────────────────────
    print("\nFeature stats (fake vs real mean):")
    print(f"  {'Feature':<28} {'Fake mean':>10} {'Real mean':>10} {'Corr':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")

    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]

    for col in EXIF_FEATURE_COLS:
        fake_mean = fake_df[col].mean()
        real_mean = real_df[col].mean()
        corr      = df[col].corr(df['label'])
        flag = " ← ⚠️ LEAKY" if abs(corr) > 0.95 else (
               " ← ✅ signal" if abs(corr) > 0.3 else "")
        print(f"  {col:<28} {fake_mean:>10.3f} {real_mean:>10.3f} {corr:>+8.3f}{flag}")
