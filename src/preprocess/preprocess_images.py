import random
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import kagglehub

# Download dataset
dataset_path = Path(
    kagglehub.dataset_download("aqibnawaz7/real-vs-fake-image-dataset")
)

print("Downloaded to:", dataset_path)

# Paths 

SRC_REAL = dataset_path / "real"
SRC_FAKE = dataset_path / "fake"

OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")

OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
MAX_IMAGES = 10000

# Fast safe hash function
def get_hash(fp):
    try:
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None


# Image processing

def process(fp, out_dir, label, idx):
    try:
        img = Image.open(fp).convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img.save(out_dir / f"{label}_{idx}.jpg", quality=95)
        return True
    except UnidentifiedImageError:
        return False
    except Exception:
        return False

#  preprocessing
def preprocess(src, dst, label):
    files = list(Path(src).rglob("*"))
    random.shuffle(files)

    seen_hashes = set()   

    count = 0
    skipped_dup = 0
    skipped_bad = 0

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

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

        if process(fp, dst, label, count):
            count += 1

    print(f"""
========================
{label.upper()} DONE
Processed: {count}
Duplicates skipped: {skipped_dup}
Bad files skipped: {skipped_bad}
========================
""")

if __name__ == "__main__":
    preprocess(SRC_REAL, OUT_REAL, "real")
    preprocess(SRC_FAKE, OUT_FAKE, "fake")