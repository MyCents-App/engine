"""HTTP client for the Surya OCR sidecar.

OCR runs in its own container because surya-ocr 0.17.1 pins transformers<5
while unsloth needs 5.5 — they cannot share an interpreter. Two images, one
private network, and only this service is exposed publicly.
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class OcrUnavailable(Exception):
    """The sidecar is unreachable — a deployment problem, not a bad upload."""


class OcrFailed(Exception):
    """The sidecar answered, but could not read this image."""


async def run(image: bytes, content_type: str = "image/jpeg") -> dict:
    """OCR one image. Returns {text, num_lines, avg_confidence, ocr_seconds}."""
    try:
        async with httpx.AsyncClient(timeout=settings.ocr_timeout_s) as client:
            resp = await client.post(
                settings.ocr_url,
                content=image,
                headers={"Content-Type": content_type or "image/jpeg"},
            )
    except httpx.RequestError as exc:
        raise OcrUnavailable(
            f"cannot reach the OCR service at {settings.ocr_url}: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise OcrFailed(f"OCR returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()


async def healthy() -> bool:
    url = settings.ocr_url.replace("/ocr", "/health")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            return (await client.get(url)).status_code == 200
    except httpx.RequestError:
        return False
