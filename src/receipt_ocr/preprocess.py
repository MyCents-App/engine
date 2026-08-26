"""Image loading and preprocessing for receipt photos.

Handles the messiness of real-world input: iPhone HEIC photos, wrong EXIF
orientation, thermal-print low contrast, and wildly varying resolutions.
Deliberately does NOT binarize the image -- Surya is a deep-learning OCR
model, not a classical one, and binarization tends to hurt its accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from receipt_ocr.config import settings

register_heif_opener()


@dataclass
class PreprocessReport:
    """Record of what was done to an image, kept for raw.json traceability."""

    original_size: tuple[int, int]
    final_size: tuple[int, int]
    exif_transposed: bool
    resized: bool
    upscaled: bool
    contrast_enhanced: bool
    rotation_applied_degrees: int = 0
    notes: list[str] = field(default_factory=list)


def load_image(path: str | Path) -> Image.Image:
    """Open a receipt photo in any supported format as an RGB PIL image."""
    image = Image.open(path)
    image.load()  # force decode now, so a corrupt file raises here, not later
    return image


def _resize_to_bounds(image: Image.Image) -> tuple[Image.Image, bool, bool]:
    w, h = image.size
    long_side = max(w, h)

    if long_side > settings.max_long_side:
        scale = settings.max_long_side / long_side
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return image.resize(new_size, Image.LANCZOS), True, False

    if long_side < settings.min_long_side:
        scale = settings.min_long_side / long_side
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return image.resize(new_size, Image.BICUBIC), False, True

    return image, False, False


def _enhance_contrast(image: Image.Image) -> Image.Image:
    """Mild CLAHE on the luminance channel -- helps faded thermal prints."""
    array = np.asarray(image)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=settings.clahe_clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def preprocess_image(path: str | Path) -> tuple[Image.Image, PreprocessReport]:
    """Load and prepare one receipt photo. Rotation-fallback (for scans with
    missing/wrong EXIF) is applied separately in the engine, since it needs a
    detection pass to judge which orientation is correct.
    """
    raw = load_image(path)
    original_size = raw.size

    # ImageOps.exif_transpose() always returns a new image object (even a
    # no-op copy), so we check the EXIF orientation tag directly to report
    # accurately whether a rotation actually happened.
    orientation_tag = raw.getexif().get(0x0112, 1)
    was_transposed = orientation_tag != 1
    image = ImageOps.exif_transpose(raw).convert("RGB")

    image, resized, upscaled = _resize_to_bounds(image)
    image = _enhance_contrast(image)

    report = PreprocessReport(
        original_size=original_size,
        final_size=image.size,
        exif_transposed=was_transposed,
        resized=resized,
        upscaled=upscaled,
        contrast_enhanced=True,
    )
    return image, report
