
import os
from pathlib import Path
from PIL import Image
import hashlib
import numpy as np

# pip install datasets
from datasets import load_dataset

DATA_DIR = Path("data/processed")
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

def get_hash(img):
    """Perceptual hash for deduplication"""
    img_gray = img.convert('L').resize((32, 32))
    pixels = np.array(img_gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()

def download_faces(target_real=50000, target_fake=50000):
    """Download 50K real + 50K fake faces — FIXED version"""
    # https://huggingface.co/datasets/Hemg/deepfake-and-real-images
    print("📥 Loading HuggingFace 190K face dataset...")
    ds = load_dataset("Hemg/deepfake-and-real-images", split="train", streaming=True)
    
    real_hashes = set()
    fake_hashes = set()
    real_count = 0
    fake_count = 0
    skipped_blur = 0
    skipped_dup = 0
    skipped_small = 0
    
    print("\n⏳ Processing (takes ~20-30 minutes)...")
    
    for item in ds:
        if real_count >= target_real and fake_count >= target_fake:
            break
        
        img = item['image']
        label = item['label']  # 0 = FAKE, 1 = REAL
        
        # FIX 1: Check minimum size BEFORE resize
        w, h = img.size
        if w < 128 or h < 128:
            skipped_small += 1
            continue
        
        # FIX 2: Resize FIRST, then check quality
        img = img.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img.convert('L'))
        
        # FIX 3: Better blur detection (Laplacian variance)
        import cv2
        lap_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        if lap_var < 50:  # Too blurry
            skipped_blur += 1
            continue
        
        # Deduplication on resized image
        img_hash = get_hash(img)
        
        if label == 1 and real_count < target_real:
            if img_hash in real_hashes:
                skipped_dup += 1
                continue
            real_hashes.add(img_hash)
            img.save(REAL_DIR / f"real_{real_count:05d}.jpg", quality=95)
            real_count += 1
            
        elif label == 0 and fake_count < target_fake:
            if img_hash in fake_hashes:
                skipped_dup += 1
                continue
            fake_hashes.add(img_hash)
            img.save(FAKE_DIR / f"fake_{fake_count:05d}.jpg", quality=95)
            fake_count += 1
        
        total = real_count + fake_count
        if total % 1000 == 0:
            print(f"  Progress: {real_count} real + {fake_count} fake = {total} total")
    
    print(f"\n{'='*60}")
    print(f"✅ DONE!")
    print(f"   Real: {real_count} | Fake: {fake_count}")
    print(f"   Skipped small: {skipped_small} | Blur: {skipped_blur} | Dupes: {skipped_dup}")
    print(f"{'='*60}")



if __name__ == "__main__":
    download_faces(target_real=50000, target_fake=50000)