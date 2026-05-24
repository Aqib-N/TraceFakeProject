"""
TraceFake AI — EXIF Injection v3

Problems in v2:
  - software_suspicious = 1.0 for ALL fakes → -1.000 corr → leaky
  - has_software/timestamp/flash/orientation = 1.0 for both → nan (useless)
  - missing_count/exif_total_tags still perfectly separated

Core principle this version:
  NO feature should be 0.0 or 1.0 for either class.
  Target correlation range: 0.30 – 0.75 for each feature.

  REAL: high probability of each "good" signal (~85-95%)
  FAKE: lower but non-zero probability (~15-45%)

  This means:
    - Some real images are missing fields (camera sometimes stripped)
    - Some fake images have good-looking fields (uploader added metadata)
    - The MODEL must learn the overall PATTERN, not any single field
"""

import sys, random, piexif
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import DATA_DIR, METADATA_CSV, EXIF_FEATURE_COLS
except ImportError:
    DATA_DIR     = Path("data/processed")
    METADATA_CSV = Path("data/metadata.csv")
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
    (b'Nikon',    b'Z6 II'),          (b'Sony',     b'ILCE-7RM5'),
    (b'Sony',     b'ILCE-7M4'),       (b'FUJIFILM', b'X-T5'),
    (b'Apple',    b'iPhone 15 Pro'),  (b'Samsung',  b'Galaxy S24 Ultra'),
    (b'Google',   b'Pixel 8 Pro'),
]
REAL_SOFTWARE = [
    b'Adobe Lightroom Classic 14.0',
    b'Capture One Pro 16.5',
    b'DxO PhotoLab 8',
    b'Darktable 4.8',
]
# Suspicious software (in config.py SUSPICIOUS_SOFTWARE_TERMS)
SUSPICIOUS_SW = [b'Adobe Photoshop 2024', b'GIMP 2.10', b'Paint.NET 5.0']
# Non-suspicious generic software
GENERIC_SW    = [b'Windows Photo Viewer', b'Preview 11.0', b'Photos 1.0']

SMARTPHONE = {b'Apple', b'Samsung', b'Google'}

def _ts(min_days=0, max_days=2190) -> bytes:
    dt = datetime.now() - timedelta(days=random.randint(min_days, max_days))
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode()

def _future_ts() -> bytes:
    dt = datetime.now() + timedelta(days=random.randint(1, 365))
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode()

SUSPICIOUS_KWS = [
    'photoshop','gimp','paint','fake','gan','deepfake',
    'stable diffusion','midjourney','dall-e','generator',
    'firefly','imagen','runway','pika','leonardo',
]


# ── REAL EXIF builder ─────────────────────────────────────────────────────────
def build_real_exif() -> bytes:
    """
    Real images: mostly good EXIF with small random omissions.
    Target per-feature presence rates:
      has_camera_info      ~90%
      has_software         ~85%
      software_suspicious  ~5%   (occasionally edited in Photoshop)
      has_timestamp        ~90%
      timestamp_consistent ~85%
      timestamp_plausible  ~88%
      timestamp_future      ~1%
      has_gps              ~55%
      has_flash            ~80%
      has_orientation      ~90%
    """
    ifd_0 = {piexif.ImageIFD.Orientation: 1}
    ifd_e = {}
    ifd_g = {}

    # Camera make/model — present 90%
    if random.random() < 0.90:
        make, model = random.choice(REAL_CAMERAS)
        ifd_0[piexif.ImageIFD.Make]  = make
        ifd_0[piexif.ImageIFD.Model] = model
    else:
        make = b''

    # Software — present 85%
    if random.random() < 0.85:
        # 5% chance it's suspicious (Photoshop edit)
        if random.random() < 0.05:
            ifd_0[piexif.ImageIFD.Software] = random.choice(SUSPICIOUS_SW)
        else:
            ifd_0[piexif.ImageIFD.Software] = random.choice(REAL_SOFTWARE)

    # Timestamps — present 90%
    if random.random() < 0.90:
        ts = _ts(0, 2190)
        ifd_0[piexif.ImageIFD.DateTime] = ts
        # Consistent 85% of the time
        if random.random() < 0.85:
            ifd_e[piexif.ExifIFD.DateTimeOriginal] = ts   # matches DateTime
        else:
            ifd_e[piexif.ExifIFD.DateTimeOriginal] = _ts(365, 2190)  # older

    # Flash — present 80%
    if random.random() < 0.80:
        ifd_e[piexif.ExifIFD.Flash] = random.choice([0, 1, 9, 16])

    # GPS — present 55%
    if random.random() < 0.55:
        ifd_g = {
            piexif.GPSIFD.GPSLatitude:     [(random.randint(0,89),1),(random.randint(0,59),1),(0,1)],
            piexif.GPSIFD.GPSLongitude:    [(random.randint(0,179),1),(random.randint(0,59),1),(0,1)],
            piexif.GPSIFD.GPSLatitudeRef:  b'N',
            piexif.GPSIFD.GPSLongitudeRef: b'E',
        }

    # Extra Exif fields
    ifd_e[piexif.ExifIFD.ISOSpeedRatings] = random.choice([100,200,400,800])
    ifd_e[piexif.ExifIFD.FocalLength]     = (random.choice([24,35,50,85]),1)

    try:
        return piexif.dump({'0th': ifd_0, 'Exif': ifd_e, 'GPS': ifd_g})
    except Exception:
        return b''


# ── FAKE EXIF builder ─────────────────────────────────────────────────────────
def build_fake_exif() -> bytes:
    """
    Fake images: partial/suspicious EXIF with lower quality signals.
    Target per-feature presence rates:
      has_camera_info      ~20%  (occasionally someone adds camera info)
      has_software         ~70%  (editors often leave software tag)
      software_suspicious  ~50%  (Photoshop/GIMP present ~50% of fakes)
      has_timestamp        ~65%
      timestamp_consistent ~25%  (often inconsistent)
      timestamp_plausible  ~40%
      timestamp_future     ~20%
      has_gps              ~10%  (rarely have GPS)
      has_flash            ~30%
      has_orientation      ~70%
    """
    ifd_0 = {}
    ifd_e = {}
    ifd_g = {}

    # Camera — only 20% of fakes have it
    if random.random() < 0.20:
        make, model = random.choice(REAL_CAMERAS)
        ifd_0[piexif.ImageIFD.Make]  = make
        ifd_0[piexif.ImageIFD.Model] = model

    # Software — 70% have it, 50% of those are suspicious
    if random.random() < 0.70:
        if random.random() < 0.50:
            ifd_0[piexif.ImageIFD.Software] = random.choice(SUSPICIOUS_SW)
        else:
            ifd_0[piexif.ImageIFD.Software] = random.choice(GENERIC_SW)

    # Orientation — 70%
    if random.random() < 0.70:
        ifd_0[piexif.ImageIFD.Orientation] = 1

    # Timestamps — 65% have them
    if random.random() < 0.65:
        r = random.random()
        if r < 0.20:
            # Future timestamp
            ts = _future_ts()
            ifd_0[piexif.ImageIFD.DateTime]          = ts
            ifd_e[piexif.ExifIFD.DateTimeOriginal]   = ts
        elif r < 0.55:
            # Inconsistent timestamps
            ts1 = _ts(0,   365)
            ts2 = _ts(730, 2190)
            ifd_0[piexif.ImageIFD.DateTime]          = ts1
            ifd_e[piexif.ExifIFD.DateTimeOriginal]   = ts2
        else:
            # Plausible + consistent (~30% of fakes with timestamps)
            ts = _ts(0, 2190)
            ifd_0[piexif.ImageIFD.DateTime]          = ts
            ifd_e[piexif.ExifIFD.DateTimeOriginal]   = ts

    # Flash — 30%
    if random.random() < 0.30:
        ifd_e[piexif.ExifIFD.Flash] = 0

    # GPS — only 10%
    if random.random() < 0.10:
        ifd_g = {
            piexif.GPSIFD.GPSLatitude:     [(random.randint(0,89),1),(random.randint(0,59),1),(0,1)],
            piexif.GPSIFD.GPSLongitude:    [(random.randint(0,179),1),(random.randint(0,59),1),(0,1)],
            piexif.GPSIFD.GPSLatitudeRef:  b'N',
            piexif.GPSIFD.GPSLongitudeRef: b'E',
        }

    if not ifd_0 and not ifd_e:
        ifd_0[piexif.ImageIFD.Orientation] = 1

    try:
        return piexif.dump({'0th': ifd_0, 'Exif': ifd_e, 'GPS': ifd_g})
    except Exception:
        return b''


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(exif_bytes: bytes) -> dict:
    null = {c: 0 for c in EXIF_FEATURE_COLS}
    null['missing_count'] = 10
    if not exif_bytes:
        return null
    try:
        exif  = piexif.load(exif_bytes)
        ifd_0 = exif.get('0th')  or {}
        ifd_e = exif.get('Exif') or {}
        ifd_g = exif.get('GPS')  or {}

        def dec(b):
            return b.decode('utf-8', errors='ignore') if isinstance(b, bytes) else str(b)

        ts_dt  = dec(ifd_0.get(piexif.ImageIFD.DateTime, b''))
        ts_dto = dec(ifd_e.get(piexif.ExifIFD.DateTimeOriginal, b''))
        primary = ts_dto or ts_dt

        def plausible(s):
            if not s: return False
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s.strip(), fmt)
                    return 1990 < dt.year <= datetime.now().year and dt <= datetime.now()
                except ValueError: continue
            return False

        future = False
        if primary:
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    future = datetime.strptime(primary.strip(), fmt) > datetime.now()
                    break
                except ValueError: continue

        sw  = ifd_0.get(piexif.ImageIFD.Software, b'')
        swl = dec(sw).lower()

        def safe_len(v):
            try: return len(v)
            except: return 1 if v else 0

        total = sum(safe_len(v) for ifd in exif.values() if ifd
                    for v in (ifd.values() if isinstance(ifd, dict) else []))

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


# ── Inject folder ─────────────────────────────────────────────────────────────
def inject_folder(folder: Path, is_fake: bool, label: int) -> list:
    files  = sorted(folder.glob("*.jpg"))
    cls    = "FAKE" if is_fake else "REAL"
    rows   = []
    errors = 0
    print(f"\n[{cls}] Injecting into {len(files):,} images...")
    for i, fp in enumerate(files):
        try:
            img  = Image.open(fp).convert("RGB")
            exif = build_fake_exif() if is_fake else build_real_exif()
            feat = extract_features(exif)
            feat['image_path'] = fp.name
            feat['label']      = label
            kw = {'format': 'JPEG', 'quality': 95, 'optimize': False}
            if exif:
                kw['exif'] = exif
            img.save(fp, **kw)
            rows.append(feat)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(files):,}...")
        except Exception as e:
            print(f"  Error {fp.name}: {e}")
            errors += 1
    print(f"  ✅ {len(files)-errors:,} done | {errors} errors")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("EXIF INJECTION v3 — probabilistic, no absolute 0/1 rates")
    print("=" * 58)

    fake_rows = inject_folder(FAKE_DIR, is_fake=True,  label=0)
    real_rows = inject_folder(REAL_DIR, is_fake=False, label=1)

    df = pd.DataFrame(fake_rows + real_rows)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_CSV, index=False)

    print(f"\n✅ metadata.csv saved | Total: {len(df):,}")

    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]

    leaky = []
    useless = []

    print(f"\n{'Feature':<28} {'Fake%':>7} {'Real%':>7} {'Corr':>8}  Status")
    print("-" * 68)
    for col in EXIF_FEATURE_COLS:
        fm   = fake_df[col].mean()
        rm   = real_df[col].mean()
        std  = df[col].std()
        if std == 0:
            corr   = float('nan')
            status = "❌ USELESS — zero variance (remove)"
            useless.append(col)
        else:
            corr = df[col].corr(df['label'])
            if abs(corr) > 0.95:
                status = "⚠️  LEAKY — remove from training"
                leaky.append(col)
            elif abs(corr) > 0.60:
                status = "✅ strong signal"
            elif abs(corr) > 0.30:
                status = "✅ medium signal"
            else:
                status = "〰  weak signal"
        print(f"{col:<28} {fm:>7.3f} {rm:>7.3f} {corr:>+8.3f}  {status}")

    remove = leaky + useless
    keep   = [c for c in EXIF_FEATURE_COLS if c not in remove]

    print(f"\n{'='*58}")
    print(f"Remove from training : {remove}")
    print(f"Train on ({len(keep)}) features: {keep}")
    print(f"{'='*58}")
    print("\nNext: python train_exif_model.py")