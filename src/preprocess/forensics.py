from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2

# =========================
# ERROR LEVEL ANALYSIS (ELA)
# =========================
def ela_analysis(image_path, quality=90):
    try:
        original = Image.open(image_path).convert("RGB")

        temp_path = "temp_ela.jpg"
        original.save(temp_path, "JPEG", quality=quality)

        compressed = Image.open(temp_path)

        diff = ImageChops.difference(original, compressed)

        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])

        scale = 255.0 / max_diff if max_diff != 0 else 1
        diff = ImageEnhance.Brightness(diff).enhance(scale)

        # convert to score
        diff_np = np.array(diff)
        ela_score = np.mean(diff_np) / 255.0

        return ela_score

    except:
        return 0.0


# =========================
# NOISE / TEXTURE ANALYSIS
# =========================
def noise_analysis(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        noise = cv2.Laplacian(gray, cv2.CV_64F).var()

        # normalize
        return min(noise / 500.0, 1.0)

    except:
        return 0.0


# =========================
# COMBINED FORENSIC SCORE
# =========================
def forensic_score(image_path):
    ela = ela_analysis(image_path)
    noise = noise_analysis(image_path)

    # weighted forensic score
    return (ela * 0.6) + (noise * 0.4)