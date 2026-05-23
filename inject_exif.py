"""
TraceFake AI — EXIF Injection v2

Root cause of AUC=1.0:
  missing_count dominates because 40% of fake images had completely
  empty EXIF (missing_count=20) while real images had full EXIF
  (missing_count~0). The model learned "no EXIF = fake" trivially.

Fix:
  ALL fake images now get SOME EXIF fields — never completely empty.
  The difference is now in QUALITY and CONSISTENCY of EXIF, not
  presence vs absence. This forces the model to learn real patterns:

  REAL: full camera EXIF, consistent timestamps, GPS, flash
  FAKE: partial EXIF, inconsistent/missing timestamps, no GPS,
        suspicious software, no camera make/model

Also removes missing_count and exif_total_tags from TRAIN_COLS
since both directly encode "how much EXIF" which is now less
of a giveaway but still too dominant.
"""

import sys, random, piexif, json
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta

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

REAL_CAMERAS = [
    (b'Canon',    b'EOS R5'),         (b'Canon',    b'EOS R6 Mark II'),
    (b'Canon',    b'EOS 5D Mark IV'), (b'Nikon',    b'Z8'),
    (b'Nikon',    b'Z6 II'),          (b'Nikon',    b'D850'),
    (b'Sony',     b'ILCE-7RM5'),      (b'Sony',     b'ILCE-7M4'),
    (b'FUJIFILM', b'X-T5'),           (b'FUJIFILM', b'X-H2'),
    (b'Apple',    b'iPhone 15 Pro'),  (b'Apple',    b'iPhone 14 Pro'),
    (b'Samsung',  b'Galaxy S24 Ultra'),(b'Google',  b'Pixel 8 Pro'),
]

REAL_SOFTWARE = [
    b'Adobe Lightroom Classic 14.0',
    b'Adobe Lightroom Classic 13.0',
    b'Capture One Pro 16.5',
    b'DxO PhotoLab 8',
    b'Darktable 4.8',
]

# Software that IS in SUSPICIOUS_SOFTWARE_TERMS → software_suspicious=1
FAKE_SOFTWARE = [
    b'Adobe Photoshop 2024',   # "photoshop" is suspicious
    b'GIMP 2.10',              # "gimp" is suspicious
    b'Paint.NET 5.0',          # "paint" is suspicious
]

SMARTPHONE_MAKES = {b'Apple', b'Samsung', b'Google'}


def _ts(days_back_min=0, days_back_max=2190) -> bytes:
    dt = datetime.now() - timedelta(days=random.randint(days_back_min, days_back_max))
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode()

def _future_ts() -> bytes:
    dt = datetime.now() + timedelta(days=random.randint(1, 365))
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode()


# ── Real EXIF: full, consistent, camera hardware ──────────────────────────────
def build_real_exif() -> bytes:
    make, model = random.choice(REAL_CAMERAS)
    software    = random.choice(REAL_SOFTWARE)
    ts          = _ts(0, 2190)
    is_phone    = make in SMARTPHONE_MAKES

    exif = {
        '0th': {
            piexif.ImageIFD.Make:           make,
            piexif.ImageIFD.Model:          model,
            piexif.ImageIFD.Software:       software,
            piexif.ImageIFD.DateTime:       ts,
            piexif.ImageIFD.Orientation:    random.choice([1, 3, 6, 8]),
            piexif.ImageIFD.XResolution:    (72, 1),
            piexif.ImageIFD.YResolution:    (72, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        },
        'Exif': {
            piexif.ExifIFD.DateTimeOriginal:  ts,   # consistent
            piexif.ExifIFD.DateTimeDigitized: ts,
            piexif.ExifIFD.ExposureTime:  (1, random.choice([100,125,200,250,400])),
            piexif.ExifIFD.FNumber:       (random.choice([18,20,28,40,56]), 10),
            piexif.ExifIFD.ISOSpeedRatings: random.choice([100,200,400,800]),
            piexif.ExifIFD.FocalLength:   (random.choice([24,35,50,85,135]), 1),
            piexif.ExifIFD.Flash:         random.choice([0, 1, 9, 16]),
            piexif.ExifIFD.WhiteBalance:  random.choice([0, 1]),
            piexif.ExifIFD.MeteringMode:  random.choice([1, 2, 3, 5]),
        },
        'GPS': {} if random.random() > 0.60 else {
            piexif.GPSIFD.GPSLatitude:  [(random.randint(0,89),1),(random.randint(0,59),1),(0,1)],
            piexif.GPSIFD.GPSLongitude: [(random.randint(0,179),1),(random.randint(0,59),1),(0,1)],
            piexif.GPSIFD.GPSLatitudeRef:  b'N',
            piexif.GPSIFD.GPSLongitudeRef: b'E',
        },
    }
    try:
        return piexif.dump(exif)
    except Exception:
        return b''


# ── Fake EXIF: always HAS some fields, but wrong/suspicious ones ──────────────
def build_fake_exif() -> bytes:
    """
    FIX: fake images ALWAYS get EXIF now — never empty.
    Signal comes from WHAT the EXIF contains, not whether it exists.

    3 scenarios:
      40% → editing software only (no camera hardware, suspicious software)
      35% → inconsistent timestamps (DateTime != DateTimeOriginal)
      25% → future or implausible timestamp + no GPS
    """
    roll = random.random()

    if roll < 0.40:
        # Editing software, no Make/Model → has_software=1, has_camera_info=0
        # software_suspicious=1 (photoshop/gimp/paint are in suspicious list)
        ts = _ts(0, 730)
        try:
            return piexif.dump({
                '0th': {
                    piexif.ImageIFD.Software:    random.choice(FAKE_SOFTWARE),
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

    elif roll < 0.75:
        # Inconsistent timestamps → timestamp_consistent=0
        ts1 = _ts(0,   365)    # recent modification time
        ts2 = _ts(730, 2190)   # old "original" time — inconsistent
        try:
            return piexif.dump({
                '0th': {
                    piexif.ImageIFD.Software:    random.choice(FAKE_SOFTWARE),
                    piexif.ImageIFD.DateTime:    ts1,   # different → inconsistent
                    piexif.ImageIFD.Orientation: 1,
                },
                'Exif': {
                    piexif.ExifIFD.DateTimeOriginal: ts2,  # mismatch
                    piexif.ExifIFD.Flash: 0,
                },
                'GPS': {},
            })
        except Exception:
            return b''

    else:
        # Future timestamp → timestamp_future=1, timestamp_plausible=0
        ts = _future_ts()
        try:
            return piexif.dump({
                '0th': {
                    piexif.ImageIFD.Software:    random.choice(FAKE_SOFTWARE),
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


# ── Feature extraction ────────────────────────────────────────────────────────
SUSPICIOUS_KWS = [
    'photoshop', 'gimp', 'paint', 'fake', 'gan', 'deepfake',
    'stable diffusion', 'midjourney', 'dall-e', 'generator',
    'firefly', 'imagen', 'runway', 'pika', 'leonardo',
]

def extract_features(exif_bytes: bytes) -> dict:
    null = {c: 0 for c in EXIF_FEATURE_COLS}
    null['missing_count'] = 10
    if not exif_bytes:
        return null
    try:
        exif     = piexif.load(exif_bytes)
        ifd_0    = exif.get('0th')  or {}
        ifd_e    = exif.get('Exif') or {}
        ifd_g    = exif.get('GPS')  or {}

        def dec(b):
            return b.decode('utf-8', errors='ignore') if isinstance(b, bytes) else str(b)

        ts_dt  = dec(ifd_0.get(piexif.ImageIFD.DateTime, b''))
        ts_dto = dec(ifd_e.get(piexif.ExifIFD.DateTimeOriginal, b''))
        primary = ts_dto or ts_dt

        def plausible(s):
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s.strip(), fmt)
                    return 1990 < dt.year <= datetime.now().year and dt <= datetime.now()
                except ValueError:
                    continue
            return False

        future = False
        if primary:
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    future = datetime.strptime(primary.strip(), fmt) > datetime.now()
                    break
                except ValueError:
                    continue

        sw  = ifd_0.get(piexif.ImageIFD.Software, b'')
        swl = dec(sw).lower()
        total = sum(len(v) if hasattr(v,'__len__') else 1
                    for v in exif.values() if v)

        return {
            'missing_count':        max(0, 20 - total),
            'has_camera_info':      int(piexif.ImageIFD.Make  in ifd_0 and
                                        piexif.ImageIFD.Model in ifd_0),
            'has_software':         int(piexif.ImageIFD.Software in ifd_0),
            'software_suspicious':  int(any(k in swl for k in SUSPICIOUS_KWS)),
            'has_timestamp':        int(bool(ts_dt or ts_dto)),
            'timestamp_consistent': int(bool(ts_dt and ts_dto and ts_dt == ts_dto)),
            'timestamp_plausible':  int(plausible(primary) if primary else False),
            'timestamp_future':     int(future),
            'exif_total_tags':      total,
            'has_gps':              int(len(ifd_g) > 0),
            'has_flash':            int(piexif.ExifIFD.Flash in ifd_e),
            'has_orientation':      int(piexif.ImageIFD.Orientation in ifd_0),
        }
    except Exception as e:
        print(f"  extract error: {e}")
        return null


# ── Inject one folder ─────────────────────────────────────────────────────────
def inject_folder(folder: Path, is_fake: bool, label: int) -> list:
    files  = sorted(folder.glob("*.jpg"))
    cls    = "FAKE" if is_fake else "REAL"
    rows   = []
    errors = 0

    print(f"\n[{cls}] Injecting into {len(files):,} images...")

    for i, fp in enumerate(files):
        try:
            img        = Image.open(fp).convert("RGB")
            exif_bytes = build_fake_exif() if is_fake else build_real_exif()
            features   = extract_features(exif_bytes)
            features['image_path'] = fp.name
            features['label']      = label

            kw = {'format': 'JPEG', 'quality': 95, 'optimize': False}
            if exif_bytes:
                kw['exif'] = exif_bytes
            img.save(fp, **kw)
            rows.append(features)

            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(files):,}...")
        except Exception as e:
            print(f"  Error {fp.name}: {e}")
            errors += 1

    print(f"  ✅ {len(files)-errors:,} done | {errors} errors")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("EXIF INJECTION v2 — fake images always get EXIF")
    print("Signal = quality/consistency, not presence/absence")
    print("=" * 55)

    fake_rows = inject_folder(FAKE_DIR, is_fake=True,  label=0)
    real_rows = inject_folder(REAL_DIR, is_fake=False, label=1)

    df = pd.DataFrame(fake_rows + real_rows)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_CSV, index=False)

    print(f"\n✅ metadata.csv → {METADATA_CSV}")
    print(f"   Total: {len(df):,} | FAKE: {(df['label']==0).sum():,} | REAL: {(df['label']==1).sum():,}")

    # ── Report ────────────────────────────────────────────────────────────────
    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]

    # Features to remove before training
    LEAKY_THRESHOLD = 0.95
    leaky = []

    print(f"\n{'Feature':<28} {'Fake':>8} {'Real':>8} {'Corr':>8}  Status")
    print("-" * 65)
    for col in EXIF_FEATURE_COLS:
        fm   = fake_df[col].mean()
        rm   = real_df[col].mean()
        corr = df[col].corr(df['label'])
        if abs(corr) > LEAKY_THRESHOLD:
            status = "⚠️  REMOVE — still leaky"
            leaky.append(col)
        elif abs(corr) > 0.60:
            status = "✅ strong signal"
        elif abs(corr) > 0.30:
            status = "✅ medium signal"
        else:
            status = "〰 weak signal"
        print(f"{col:<28} {fm:>8.3f} {rm:>8.3f} {corr:>+8.3f}  {status}")

    if leaky:
        print(f"\n⚠️  Still leaky after v2: {leaky}")
        print("   Add these to LEAKY set in train_exif_model.py")
    else:
        print("\n✅ No leaky features — ready to train!")

    print("\nRun: python train_exif_model.py")