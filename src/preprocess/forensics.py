import tempfile
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance


# Error Level Analysis (ELA)

def ela_analysis(image_path, quality: int = 90):
    """
    Error Level Analysis — detects recompression artifacts.
    Fake images often show different compression patterns across regions.
    """
    temp_path = None
    try:
        original = Image.open(image_path).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
        original.save(temp_path, "JPEG", quality=quality)

        compressed = Image.open(temp_path)
        diff = ImageChops.difference(original, compressed)

        extrema = diff.getextrema()
        max_diff = max(ex[1] for ex in extrema)

        if max_diff == 0:
            return None

        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)

        diff_np = np.array(diff)
        ela_score = np.mean(diff_np) / 255.0
        return float(min(ela_score * 2, 1.0))

    except Exception as e:
        print(f"ELA analysis error for {image_path}: {e}")
        return 0.5  

    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# Noise / Texture Analysis

def noise_analysis(image_path):
    """
    Laplacian variance — detects unnatural smoothness in fakes.
    Real photos: typically 100–500; GAN outputs are often much smoother.
    Returns float in [0, 1].  0.5 on error (neutral).
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(noise / 500.0, 1.0))
    except Exception:
        return 0.5  


# JPEG Quality via Quantization Tables 

def estimate_jpeg_quality(image_path):
    """
    Estimate JPEG quality using DCT quantization tables from PIL.
    This is accurate and dimension-independent.
    """
    try:
        img = Image.open(image_path)

        if img.format != "JPEG":
            return 0.75  

        if not hasattr(img, "quantization") or not img.quantization:
            return 0.5  

        # Luminance quantization table (key 0)
        luma_table = img.quantization.get(0, [])
        if not luma_table:
            return 0.5

        q_sum = sum(luma_table)
        # Empirical range: ~64 (quality 100) → ~5000 (quality ~1)
        quality_score = 1.0 - min(q_sum / 5000.0, 1.0)
        return float(quality_score)

    except Exception:
        return 0.5


# Chromatic Aberration Analysis 

def chromatic_analysis(image_path):
    """
    Detect chromatic aberration — real lenses produce it; many fakes don't.
    Returns float in [0, 1].  0.5 on error (neutral).
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


# FFT Artifact Detection (NEW) 

def fft_artifact_score(image_path):
    """
    Frequency-domain analysis for GAN/diffusion grid artifacts.

    GANs and some diffusion models leave characteristic periodic peaks in the
    frequency domain that spatial methods (ELA, Laplacian) cannot detect.
    This function measures the ratio of peripheral to central frequency energy;
    a high ratio indicates suspicious periodic patterns.

    Returns float in [0, 1]:  higher = more likely synthetic.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.5

        # Compute 2-D FFT and shift zero frequency to center
        f_shift = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        magnitude = 20 * np.log(np.abs(f_shift) + 1)

        cy, cx = np.array(magnitude.shape) // 2
        r = 10  # central region radius

        center_energy = float(np.mean(
            magnitude[cy - r: cy + r, cx - r: cx + r]
        ))

        # Zero out center, measure periphery
        peripheral = magnitude.copy()
        peripheral[cy - r: cy + r, cx - r: cx + r] = 0
        peripheral_energy = float(np.mean(peripheral))

        # High peripheral/center ratio → suspicious periodic pattern
        ratio = peripheral_energy / (center_energy + 1e-6)
        # Typical real-photo ratios ~0.3–0.6; GAN artifacts push this higher
        score = min(ratio / 1.0, 1.0)
        return float(score)

    except Exception:
        return 0.5


# Combined Forensic Score 

def forensic_score(image_path):
    """
    Weighted combination of forensic signals.

    Weights (updated):
      ELA      0.35  — primary recompression signal
      Noise    0.20  — GAN smoothness detection
      JPEG     0.15  — quantization table analysis (FIX: now accurate)
      Chromatic 0.15 — optical aberration
      FFT      0.15  — frequency-domain GAN artifact (NEW)

    None from ela_analysis (degenerate uniform image) is treated as 0.5.
    """
    ela       = ela_analysis(image_path)
    ela       = 0.5 if ela is None else ela  
    noise     = noise_analysis(image_path)
    jpeg_q    = estimate_jpeg_quality(image_path)
    chromatic = chromatic_analysis(image_path)
    fft       = fft_artifact_score(image_path)

    final = (
        ela       * 0.35 +
        noise     * 0.20 +
        jpeg_q    * 0.15 +
        chromatic * 0.15 +
        fft       * 0.15
    )
    return float(min(final, 1.0))
