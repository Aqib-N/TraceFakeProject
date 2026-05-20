import os
import sys
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = Path("data/processed")

print("=" * 60)
print("TRACEFAKE DATASET DIAGNOSTIC")
print("=" * 60)

CLASSES = ["fake", "real"]
all_hashes = {}
issues = []

for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    if not cls_dir.exists():
        issues.append(f"MISSING: {cls_dir}")
        continue

    files = sorted(
        list(cls_dir.glob("*.jpg")) +
        list(cls_dir.glob("*.jpeg")) +
        list(cls_dir.glob("*.png"))
    )

    print(f"\n[{cls.upper()}] {len(files)} images in {cls_dir}")

    if len(files) == 0:
        issues.append(f"EMPTY folder: {cls_dir}")
        continue

    # File size distribution
    sizes = [os.path.getsize(f) for f in files[:500]]
    print(f"  File size — min: {min(sizes)//1024}KB  "
          f"avg: {sum(sizes)//len(sizes)//1024}KB  "
          f"max: {max(sizes)//1024}KB")
    tiny = sum(1 for s in sizes if s < 5000)
    if tiny > 0:
        issues.append(f"{cls}: {tiny} files < 5KB (likely corrupted/empty)")

    # Image dimension check
    dims = []
    corrupt = 0
    sample = files[:200]
    for f in sample:
        try:
            with Image.open(f) as img:
                dims.append(img.size)
        except Exception:
            corrupt += 1
    if corrupt > 0:
        issues.append(f"{cls}: {corrupt}/{len(sample)} images failed to open")
    if dims:
        unique_dims = Counter(dims)
        print(f"  Dimensions — most common: {unique_dims.most_common(3)}")

    # Pixel statistics (detect blank/uniform images)
    pixel_means = []
    pixel_stds  = []
    for f in files[:100]:
        try:
            with Image.open(f) as img:
                arr = np.array(img.convert("RGB").resize((64, 64)))
                pixel_means.append(arr.mean())
                pixel_stds.append(arr.std())
        except Exception:
            pass
    if pixel_means:
        avg_mean = np.mean(pixel_means)
        avg_std  = np.mean(pixel_stds)
        print(f"  Pixel stats — mean: {avg_mean:.1f}  std: {avg_std:.1f}")
        if avg_std < 10:
            issues.append(
                f"{cls}: avg pixel std={avg_std:.1f} — images may be blank/uniform"
            )

    # Hash duplicates (within class)
    hashes = []
    for f in files[:300]:
        try:
            h = hashlib.md5(open(f, "rb").read()).hexdigest()
            hashes.append(h)
            all_hashes.setdefault(h, []).append((cls, f))
        except Exception:
            pass
    dup_count = len(hashes) - len(set(hashes))
    if dup_count > 0:
        print(f"  ⚠️  {dup_count} duplicate images within {cls}/")

    # EXIF presence
    has_exif = 0
    for f in files[:100]:
        try:
            with Image.open(f) as img:
                if img._getexif():
                    has_exif += 1
        except Exception:
            pass
    print(f"  EXIF present: {has_exif}/100 sampled images")
    if cls == "fake" and has_exif == 0:
        issues.append(
            "fake: 0/100 images have EXIF — EXIF model will have data leakage"
        )
    if cls == "real" and has_exif < 20:
        issues.append(
            f"real: only {has_exif}/100 images have EXIF — unexpected for camera photos"
        )

# Cross-class duplicate check
cross_dups = {h: paths for h, paths in all_hashes.items()
              if len(paths) > 1 and len({p[0] for p in paths}) > 1}
if cross_dups:
    issues.append(
        f"{len(cross_dups)} IDENTICAL images appear in BOTH fake/ and real/ "
        f"(label noise — model cannot learn)"
    )
    for h, paths in list(cross_dups.items())[:3]:
        print(f"  Cross-dup: {[str(p[1].name) for p in paths]}")

# Summary
print("\n" + "=" * 60)
if issues:
    print(f"ISSUES FOUND ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. ⚠️  {issue}")
    print("\nFix these before training — they explain the 0.50 accuracy.")
else:
    print("✅ No major data issues found.")
    print("Dataset looks healthy — the training fixes should work.")
print("=" * 60)