import random
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import kagglehub

# Download dataset
dataset_path = Path(kagglehub.dataset_download("aqibnawaz7/real-vs-fake-image-dataset"))

print("Downloaded to:", dataset_path)

# FIX: point to actual dataset folders
SRC_REAL = dataset_path / "real"
SRC_FAKE = dataset_path / "fake"

OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")

IMG_SIZE = (224, 224)
MAX_IMAGES = 20000

OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

seen_hashes = set()

def get_hash(fp):
    try:
        return hashlib.md5(open(fp, "rb").read()).hexdigest()
    except:
        return None

def process(fp, out_dir, label, idx):
    try:
        img = Image.open(fp).convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img.save(out_dir / f"{label}_{idx}.jpg", quality=95)
        return True
    except Exception as e:
        return False

def preprocess(src, dst, label):
    files = list(Path(src).rglob("*"))
    random.shuffle(files)

    count = 0
    for fp in files:
        if count >= MAX_IMAGES:
            break

        if fp.suffix.lower() not in [".jpg", ".png", ".jpeg", ".webp"]:
            continue

        h = get_hash(fp)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        if process(fp, dst, label, count):
            count += 1

    print(f"{label} done: {count}")

if __name__ == "__main__":
    preprocess(SRC_REAL, OUT_REAL, "real")
    preprocess(SRC_FAKE, OUT_FAKE, "fake")