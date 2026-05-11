import uuid
import tempfile
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance


# ERROR LEVEL ANALYSIS (ELA)
def ela_analysis(image_path, quality: int = 90) -> float:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = Path(tmp.name)
    tmp.close()

    try:
        original = Image.open(image_path).convert("RGB")
        original.save(str(temp_path), "JPEG", quality=quality)
        compressed = Image.open(str(temp_path))

        diff = ImageChops.difference(original, compressed)
        extrema = diff.getextrema()
        max_diff = max(ex[1] for ex in extrema)

        if max_diff == 0:
            return 0.0

        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        diff_np = np.array(diff)
        ela_score = np.mean(diff_np) / 255.0
        return float(min(ela_score * 2.0, 1.0))

    except Exception:
        return 0.5  
    finally:
        temp_path.unlink(missing_ok=True)


# NOISE / TEXTURE ANALYSIS
def noise_analysis(image_path) -> float:
    """
    Laplacian variance — detects unnatural GAN smoothness.
    FIX 2: adaptive normalization based on image resolution.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()

        h, w = gray.shape
        area = h * w
        scale = 500.0 * (area / (224 * 224))
        normalized = float(min(noise / scale, 1.0))
        return normalized

    except Exception:
        return 0.5

# JPEG QUALITY ESTIMATION
def estimate_jpeg_quality(image_path) -> float:
    """
    Estimate JPEG quality — fakes often have inconsistent compression.
    Returns higher value for higher-quality (more likely real) images.
    """
    try:
        img = Image.open(image_path)
        if img.format != "JPEG":
            return 0.75   

        img_array = np.array(img)
        file_size = Path(image_path).stat().st_size
        pixels = img_array.shape[0] * img_array.shape[1]
        if pixels == 0:
            return 0.5

        bpp = (file_size * 8) / pixels  
        quality_estimate = float(min(bpp / 1.5, 1.0))
        return quality_estimate

    except Exception:
        return 0.5


# CHROMATIC ABERRATION ANALYSIS
def chromatic_analysis(image_path) -> float:
    """
    Detect chromatic aberration — real lenses have it, fakes often don't.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5

        b, g, r = cv2.split(img)
        rg_diff = cv2.absdiff(r, g)
        gb_diff = cv2.absdiff(g, b)
        ca_score = (np.mean(rg_diff) + np.mean(gb_diff)) / 255.0
        return float(min(ca_score, 1.0))

    except Exception:
        return 0.5


# COMBINED FORENSIC SCORE
def forensic_score(image_path) -> float:
    """
    Weighted combination of forensic signals.
    Each sub-function returns a value in [0, 1] where higher = more real.
    Falls back to 0.5 (neutral) if any signal fails.
    """
    ela          = ela_analysis(image_path)
    noise        = noise_analysis(image_path)
    jpeg_quality = estimate_jpeg_quality(image_path)
    chromatic    = chromatic_analysis(image_path)

    score = (
        ela          * 0.40 +
        noise        * 0.25 +
        jpeg_quality * 0.20 +
        chromatic    * 0.15
    )
    return float(min(score, 1.0))