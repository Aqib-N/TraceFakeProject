import random
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError


SRC_REAL = Path("/content/drive/MyDrive/real_image_processed")  
SRC_FAKE = Path("/content/drive/MyDrive/fake_image_processed")  

# Output paths
OUT_REAL = Path("data/processed/real")
OUT_FAKE = Path("data/processed/fake")

# Create output directories
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

# Image processing function - PRESERVES EXIF and forensic metadata
def process(fp, out_dir, label, idx):
    try:
        # Open image
        img = Image.open(fp)
        
        # Extract EXIF data and other metadata BEFORE any modifications
        exif_data = img.info.get('exif', b'')
        icc_profile = img.info.get('icc_profile', b'')
        comment = img.info.get('comment', b'')
        
        # Convert to RGB if needed (for PNG with alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        save_kwargs = {
            'format': 'JPEG',
            'quality': 95,
            'optimize': False,      
            'progressive': False   
        }
        
        # Add back all preserved metadata
        if exif_data:
            save_kwargs['exif'] = exif_data
        if icc_profile:
            save_kwargs['icc_profile'] = icc_profile
        if comment:
            save_kwargs['comment'] = comment
        
        img.save(out_dir / f"{label}_{idx}.jpg", **save_kwargs)
        return True
    except UnidentifiedImageError:
        return False
    except Exception as e:
        print(f"Error processing {fp}: {e}")
        return False

# Preprocessing function
def preprocess(src, dst, label):
    if not src.exists():
        print(f"⚠️ Warning: Source path does not exist: {src}")
        print(f"Please update SRC_REAL and SRC_FAKE paths to your Google Drive directories")
        return
    
    print(f"\nScanning {src} for images...")
    files = list(src.rglob("*"))
    random.shuffle(files)

    seen_hashes = set()   
    count = 0
    skipped_dup = 0
    skipped_bad = 0

    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

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

        # Progress indicator
        if count % 500 == 0 and count > 0:
            print(f"Processed {count}/{MAX_IMAGES} {label} images...")

    print(f"""
========================
{label.upper()} DONE
Processed: {count}
Duplicates skipped: {skipped_dup}
Bad files skipped: {skipped_bad}
EXIF/Forensic metadata: PRESERVED ✓
========================
""")

if __name__ == "__main__":
    print("=" * 50)
    print("Processing REAL images from Google Drive...")
    print("EXIF, GPS, and forensic metadata will be PRESERVED")
    print("=" * 50)
    preprocess(SRC_REAL, OUT_REAL, "real")
    
    print("\n" + "=" * 50)
    print("Processing FAKE images from Google Drive...")
    print("EXIF, GPS, and forensic metadata will be PRESERVED")
    print("=" * 50)
    preprocess(SRC_FAKE, OUT_FAKE, "fake")
    
    print("\n" + "=" * 50)
    print("✅ COMPLETE! Processed images saved to:")
    print(f"   Real images: {OUT_REAL.absolute()}")
    print(f"   Fake images: {OUT_FAKE.absolute()}")
    print("   EXIF/Forensic metadata: PRESERVED ✓")
    print("=" * 50)