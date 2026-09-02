"""Orchestrates preprocess -> orientation -> OCR -> row reconstruction -> save.

Public entry points: `process_image` (one photo) and `process_batch` (many
photos, batched through the GPU for throughput) -- both importable directly
by the rest of the app (e.g. an upload handler), and both used by the CLI.
"""

from __future__ import annotations

import logging
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from receipt_ocr import io_utils
from receipt_ocr.config import settings
from receipt_ocr.engine import SuryaEngine
from receipt_ocr.preprocess import PreprocessReport, preprocess_image
from receipt_ocr.reconstruct import LineFragment, reconstruct_text

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_ERROR = "error"


@dataclass
class ReceiptOCRResult:
    output_id: str
    source_path: str
    status: str
    text: str = ""
    num_lines: int = 0
    avg_confidence: float = 0.0
    error: str | None = None
    duration_s: float = 0.0
    output_paths: dict[str, str] = field(default_factory=dict)


def _default_output_id(path: str | Path) -> str:
    return Path(path).stem


def _fragments_from_ocr_result(ocr_result) -> list[LineFragment]:
    fragments = []
    for line in ocr_result.text_lines:
        text = line.text.strip()
        if not text:
            continue
        fragments.append(
            LineFragment(text=text, bbox=tuple(line.bbox), confidence=line.confidence or 0.0)
        )
    return fragments


def _classify_status(num_lines: int, avg_confidence: float) -> str:
    if num_lines == 0:
        return STATUS_EMPTY
    if avg_confidence < settings.low_confidence_threshold:
        return STATUS_LOW_CONFIDENCE
    return STATUS_OK


def _save_result(
    output_dir: str | Path,
    output_id: str,
    source_path: str | Path,
    preprocess_report: PreprocessReport,
    fragments: list[LineFragment],
    text: str,
) -> dict[str, str]:
    raw_payload = {
        "source_image": str(source_path),
        "preprocessing": asdict(preprocess_report),
        "text_lines": [
            {"text": f.text, "bbox": list(f.bbox), "confidence": f.confidence}
            for f in fragments
        ],
    }
    raw_path = io_utils.write_raw_json(output_dir, output_id, raw_payload)
    text_path = io_utils.write_text(output_dir, output_id, text)
    return {"raw_json": str(raw_path), "text": str(text_path)}


def _record(
    output_dir: str | Path,
    path: str | Path,
    output_id: str,
    status: str,
    num_lines: int,
    avg_confidence: float,
    duration_s: float,
    error: str | None = None,
) -> None:
    io_utils.append_manifest(
        output_dir,
        {
            "file": str(path),
            "output_id": output_id,
            "status": status,
            "num_lines": num_lines,
            "avg_confidence": round(avg_confidence, 4),
            "error": error,
            "duration_s": round(duration_s, 3),
        },
    )


def process_image(
    path: str | Path,
    output_dir: str | Path,
    output_id: str | None = None,
    engine: SuryaEngine | None = None,
) -> ReceiptOCRResult:
    """Run the full pipeline on one receipt photo and save its output.

    `output_id` defaults to the filename stem; pass a unique id (e.g. an
    upload UUID) to avoid collisions when filenames repeat across uploads.
    """
    output_id = output_id or _default_output_id(path)
    engine = engine or SuryaEngine()
    start = time.monotonic()

    try:
        image, report = preprocess_image(path)
        orientation = engine.correct_orientation(image, assume_upright=report.exif_transposed)
        report.rotation_applied_degrees = orientation.rotation_degrees

        [ocr_result] = engine.recognize_batch([orientation.image])
        fragments = _fragments_from_ocr_result(ocr_result)
        text, _rows = reconstruct_text(fragments)

        avg_confidence = (
            statistics.fmean(f.confidence for f in fragments) if fragments else 0.0
        )
        status = _classify_status(len(fragments), avg_confidence)
        output_paths = _save_result(output_dir, output_id, path, report, fragments, text)
        duration = time.monotonic() - start

        _record(output_dir, path, output_id, status, len(fragments), avg_confidence, duration)
        return ReceiptOCRResult(
            output_id=output_id,
            source_path=str(path),
            status=status,
            text=text,
            num_lines=len(fragments),
            avg_confidence=avg_confidence,
            duration_s=duration,
            output_paths=output_paths,
        )
    except Exception as exc:  # noqa: BLE001 -- one bad receipt must not kill a batch
        duration = time.monotonic() - start
        logger.exception("Failed to process %s", path)
        _record(output_dir, path, output_id, STATUS_ERROR, 0, 0.0, duration, error=str(exc))
        return ReceiptOCRResult(
            output_id=output_id,
            source_path=str(path),
            status=STATUS_ERROR,
            error=str(exc),
            duration_s=duration,
        )


def process_batch(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    engine: SuryaEngine | None = None,
    max_images_per_call: int | None = None,
) -> list[ReceiptOCRResult]:
    """Process many receipts, batching preprocessed images through the GPU
    in chunks for throughput. A failure on one image doesn't stop the batch.
    """
    engine = engine or SuryaEngine()
    max_images_per_call = max_images_per_call or settings.max_images_per_call

    results: list[ReceiptOCRResult] = []
    for chunk_start in range(0, len(paths), max_images_per_call):
        chunk = paths[chunk_start : chunk_start + max_images_per_call]
        prepared: list[tuple[str | Path, str, PreprocessReport, object]] = []

        for path in chunk:
            output_id = _default_output_id(path)
            start = time.monotonic()
            try:
                image, report = preprocess_image(path)
                orientation = engine.correct_orientation(image, assume_upright=report.exif_transposed)
                report.rotation_applied_degrees = orientation.rotation_degrees
                prepared.append((path, output_id, report, orientation.image, start))
            except Exception as exc:  # noqa: BLE001
                duration = time.monotonic() - start
                logger.exception("Failed to preprocess %s", path)
                _record(output_dir, path, output_id, STATUS_ERROR, 0, 0.0, duration, error=str(exc))
                results.append(
                    ReceiptOCRResult(
                        output_id=output_id,
                        source_path=str(path),
                        status=STATUS_ERROR,
                        error=str(exc),
                        duration_s=duration,
                    )
                )

        if not prepared:
            continue

        images = [item[3] for item in prepared]
        try:
            ocr_results = engine.recognize_batch(images)
        except Exception as exc:  # noqa: BLE001 -- whole chunk failed (e.g. persistent OOM)
            logger.exception("Failed to run OCR on a batch")
            for path, output_id, _report, _image, start in prepared:
                duration = time.monotonic() - start
                _record(output_dir, path, output_id, STATUS_ERROR, 0, 0.0, duration, error=str(exc))
                results.append(
                    ReceiptOCRResult(
                        output_id=output_id,
                        source_path=str(path),
                        status=STATUS_ERROR,
                        error=str(exc),
                        duration_s=duration,
                    )
                )
            continue

        for (path, output_id, report, _image, start), ocr_result in zip(prepared, ocr_results):
            fragments = _fragments_from_ocr_result(ocr_result)
            text, _rows = reconstruct_text(fragments)
            avg_confidence = (
                statistics.fmean(f.confidence for f in fragments) if fragments else 0.0
            )
            status = _classify_status(len(fragments), avg_confidence)
            output_paths = _save_result(output_dir, output_id, path, report, fragments, text)
            duration = time.monotonic() - start
            _record(output_dir, path, output_id, status, len(fragments), avg_confidence, duration)
            results.append(
                ReceiptOCRResult(
                    output_id=output_id,
                    source_path=str(path),
                    status=status,
                    text=text,
                    num_lines=len(fragments),
                    avg_confidence=avg_confidence,
                    duration_s=duration,
                    output_paths=output_paths,
                )
            )

    return results


def new_output_id() -> str:
    """Convenience for callers (e.g. an upload handler) that want a
    collision-proof id instead of relying on the source filename.
    """
    return uuid.uuid4().hex
