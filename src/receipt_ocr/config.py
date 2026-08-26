"""Central configuration for the receipt OCR pipeline.

Important: this module must be imported before any ``surya.*`` module, because
it sets environment-variable defaults (batch sizes) that Surya's own
``pydantic-settings`` object reads once, at first import. Every other module
in this package imports ``receipt_ocr.config`` first to guarantee that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a local .env file if present. This does NOT override variables already
# set in the real environment, so explicit `export RECOGNITION_BATCH_SIZE=...`
# always wins over .env, which in turn wins over the defaults below.
load_dotenv()

# --- Surya's own env-var-driven settings -----------------------------------
# Surya's defaults (RECOGNITION_BATCH_SIZE=512, DETECTOR_BATCH_SIZE=36) target
# ~20GB / ~16GB of VRAM and will OOM a 12GB card. These defaults are sized to
# comfortably fit an RTX 5070 (12GB) alongside the foundation model weights.
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "96")  # ~40MB/item -> ~3.8GB
os.environ.setdefault("DETECTOR_BATCH_SIZE", "8")  # ~440MB/item -> ~3.5GB


@dataclass(frozen=True)
class Settings:
    # --- preprocessing ---
    max_long_side: int = int(os.environ.get("RECEIPT_OCR_MAX_LONG_SIDE", 2560))
    min_long_side: int = int(os.environ.get("RECEIPT_OCR_MIN_LONG_SIDE", 1000))
    clahe_clip_limit: float = float(os.environ.get("RECEIPT_OCR_CLAHE_CLIP", 2.0))

    # --- orientation fallback ---
    # Off by default. EXIF-based correction (see preprocess.py) handles the
    # vast majority of real phone photos and is fully reliable. This fallback
    # -- guessing rotation from Surya's own recognition confidence when EXIF
    # is silent -- was tested and found unreliable: the confidence gap
    # between the correct orientation and a wrong one is often only a few
    # hundredths, so it can (and during testing, did) confidently flip an
    # already-upright image. Enable it only if you've confirmed it helps on
    # your actual data, and treat the `low_confidence` manifest status as
    # the honest signal for "this photo may need a retake" instead.
    orientation_fallback_enabled: bool = (
        os.environ.get("RECEIPT_OCR_ORIENTATION_FALLBACK", "0") == "1"
    )

    # --- row reconstruction ---
    row_y_overlap_threshold: float = float(
        os.environ.get("RECEIPT_OCR_ROW_Y_OVERLAP", 0.4)
    )
    row_gap_multiplier: float = float(os.environ.get("RECEIPT_OCR_ROW_GAP_MULT", 3.0))

    # --- quality flags ---
    low_confidence_threshold: float = float(
        os.environ.get("RECEIPT_OCR_LOW_CONFIDENCE", 0.5)
    )
    # Receipts have no legitimate math notation; disabling math mode avoids
    # stray LaTeX-ish tokens creeping into price digits.
    math_mode: bool = os.environ.get("RECEIPT_OCR_MATH_MODE", "0") == "1"

    # --- batching ---
    max_images_per_call: int = int(os.environ.get("RECEIPT_OCR_MAX_BATCH", 16))

    # --- OOM resilience ---
    oom_retry_batch_divisor: int = int(os.environ.get("RECEIPT_OCR_OOM_DIVISOR", 2))


settings = Settings()

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}
