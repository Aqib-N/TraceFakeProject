import random
import hashlib
import piexif
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


#  Kaggle 
SRC_REAL = Path("/kaggle/input/datasets/aqibnawaz7/real-vs-fake-image-dataset/real_image_processed/real_image_processed")
SRC_FAKE = Path("/kaggle/input/datasets/aqibnawaz7/real-vs-fake-image-dataset/fake_image_processed/fake_image_processed")

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

# HELPERS
def get_hash(fp: Path):
    try:
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def process_image(fp: Path, out_dir: Path, label: str, idx: int):
    try:
        img = Image.open(fp)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        save_path = out_dir / f"{label}_{idx}.jpg"
        img.save(save_path, format="JPEG", quality=95)

        return {
            "image_path": f"{label}_{idx}.jpg",
            "label": 1 if label == "fake" else 0
        }

    except UnidentifiedImageError:
        return None
    except Exception:
        return None

# PIPELINE

def preprocess(src: Path, dst: Path, label: str):
    print(f"\nScanning {src}")

    files = [f for f in src.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    random.shuffle(files)

    seen = set()
    unique = []

    for fp in files:
        if len(unique) >= MAX_IMAGES:
            break
        h = get_hash(fp)
        if h and h not in seen:
            seen.add(h)
            unique.append(fp)

    print(f"Unique images: {len(unique)}")

    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {
            ex.submit(process_image, fp, dst, label, i): i
            for i, fp in enumerate(unique)
        }

        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    print(f"{label.upper()} done: {len(results)}")
    return results

# MAIN

if __name__ == "__main__":
    real = preprocess(SRC_REAL, OUT_REAL, "real")
    fake = preprocess(SRC_FAKE, OUT_FAKE, "fake")

    df = pd.DataFrame(real + fake)
    df.to_csv("data/images_only.csv", index=False)
