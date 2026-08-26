"""Thin wrapper around Surya's predictors: lazy loading, orientation
fallback, and GPU-OOM-safe batched recognition.
"""

from __future__ import annotations

import logging
import os
import statistics
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from receipt_ocr import config  # noqa: F401  (must be imported before surya.*)
from receipt_ocr.config import settings

import torch
from surya.common.surya.schema import TaskNames
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import OCRResult

logger = logging.getLogger(__name__)

_ROTATIONS = (0, 90, 180, 270)


@dataclass
class OrientationResult:
    image: Image.Image
    rotation_degrees: int
    attempted_fallback: bool


class SuryaEngine:
    """Owns the (expensive-to-load) Surya model instances for one process."""

    def __init__(self) -> None:
        self._foundation: FoundationPredictor | None = None
        self._recognition: RecognitionPredictor | None = None
        self._detection: DetectionPredictor | None = None

    @property
    def foundation(self) -> FoundationPredictor:
        if self._foundation is None:
            logger.info("Loading Surya foundation model...")
            self._foundation = FoundationPredictor()
        return self._foundation

    @property
    def recognition(self) -> RecognitionPredictor:
        if self._recognition is None:
            self._recognition = RecognitionPredictor(self.foundation)
        return self._recognition

    @property
    def detection(self) -> DetectionPredictor:
        if self._detection is None:
            logger.info("Loading Surya detection model...")
            self._detection = DetectionPredictor()
        return self._detection

    def warm_up(self) -> None:
        """Force all three models to load now, rather than on first use."""
        _ = self.recognition
        _ = self.detection

    # -- orientation -----------------------------------------------------

    def _recognition_confidence_score(self, image: Image.Image) -> float:
        """Mean line confidence from a real recognition pass.

        Detection alone can't tell upright from upside-down: a 180-degree
        rotated line of text is still a plausible-looking wide "text-shaped"
        box to the detector, so detection confidence stays high either way.
        Recognition confidence is the signal that actually reflects whether
        the model is reading real, legible characters, so it reliably
        collapses for both sideways (90/270) and upside-down (180) pages.
        """
        [result] = self._recognize_raw([image])
        lines = result.text_lines
        if not lines:
            return 0.0
        return statistics.fmean(line.confidence or 0.0 for line in lines)

    def correct_orientation(
        self, image: Image.Image, assume_upright: bool = False
    ) -> OrientationResult:
        """Try 0/90/180/270 rotations and keep whichever one Surya can
        read with the highest confidence.

        Disabled by default (`settings.orientation_fallback_enabled`):
        testing showed the confidence gap between the correct orientation
        and a wrong one is often just a few hundredths, so this can (and
        during testing, did) confidently flip an already-upright image. It's
        kept here, opt-in, for cases where you've verified it helps on your
        own data; otherwise EXIF-based correction (always applied, see
        preprocess.py) plus the `low_confidence` manifest status (as a
        signal to prompt a retake) are the safer defaults.

        `assume_upright` should be True when the caller already trusts the
        image's orientation (e.g. it was corrected from reliable EXIF data)
        -- this skips the extra recognition passes entirely.
        """
        if not settings.orientation_fallback_enabled or assume_upright:
            return OrientationResult(image, 0, attempted_fallback=False)

        best_image, best_rotation, best_score = image, 0, self._recognition_confidence_score(image)
        for degrees in _ROTATIONS[1:]:
            rotated = image.rotate(-degrees, expand=True, resample=Image.BICUBIC)
            score = self._recognition_confidence_score(rotated)
            if score > best_score:
                best_image, best_rotation, best_score = rotated, degrees, score

        if best_rotation != 0:
            logger.info("Corrected orientation by %d degrees", best_rotation)
        return OrientationResult(best_image, best_rotation, attempted_fallback=True)

    # -- recognition -------------------------------------------------------

    def _recognize_raw(self, images: list[Image.Image]) -> list[OCRResult]:
        """Run detection+recognition with automatic batch-size backoff on
        GPU OOM. Shared by `recognize_batch` and the orientation scorer.
        """
        task_names = [TaskNames.ocr_with_boxes] * len(images)
        recognition_batch_size = int(os.environ.get("RECOGNITION_BATCH_SIZE", 96))
        detection_batch_size = int(os.environ.get("DETECTOR_BATCH_SIZE", 8))

        attempt = 0
        while True:
            try:
                return self.recognition(
                    images,
                    task_names=task_names,
                    det_predictor=self.detection,
                    detection_batch_size=detection_batch_size,
                    recognition_batch_size=recognition_batch_size,
                    math_mode=settings.math_mode,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                attempt += 1
                if attempt > 3 or (recognition_batch_size <= 1 and detection_batch_size <= 1):
                    raise
                recognition_batch_size = max(1, recognition_batch_size // settings.oom_retry_batch_divisor)
                detection_batch_size = max(1, detection_batch_size // settings.oom_retry_batch_divisor)
                logger.warning(
                    "CUDA OOM; retrying with recognition_batch_size=%d, detection_batch_size=%d",
                    recognition_batch_size,
                    detection_batch_size,
                )

    def recognize_batch(self, images: Sequence[Image.Image]) -> list[OCRResult]:
        """Run detection+recognition on a batch of already-preprocessed,
        already-oriented images.
        """
        return self._recognize_raw(list(images))
