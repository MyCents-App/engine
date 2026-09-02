"""Engine configuration — all environment driven.

Nothing here may hardcode a host path. The service is built once as an image
and run on a machine that is not the one it was developed on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # --- model ---
    checkpoint: str = os.environ.get(
        "ENGINE_CHECKPOINT", "/models/qwen3.5-2b-qlora/checkpoint-550"
    )
    max_seq_length: int = _int("ENGINE_MAX_SEQ_LENGTH", 2048)
    max_new_tokens: int = _int("ENGINE_MAX_NEW_TOKENS", 768)
    load_in_4bit: bool = os.environ.get("ENGINE_LOAD_IN_4BIT", "1") == "1"

    # --- OCR sidecar ---
    ocr_url: str = os.environ.get("OCR_URL", "http://ocr:8001/ocr")
    ocr_timeout_s: float = _float("OCR_TIMEOUT_S", 300.0)

    # --- API ---
    # Empty disables auth. NEVER leave it empty when the service is reachable
    # through a tunnel — see DEPLOY.md.
    api_key: str = os.environ.get("ENGINE_API_KEY", "")
    max_upload_bytes: int = _int("ENGINE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
    max_pages: int = _int("ENGINE_MAX_PAGES", 5)

    # Generation is serialized on one GPU; requests beyond this wait, and
    # requests beyond the queue are rejected rather than piling up until the
    # client times out with no explanation.
    max_queue: int = _int("ENGINE_MAX_QUEUE", 8)
    queue_timeout_s: float = _float("ENGINE_QUEUE_TIMEOUT_S", 240.0)

    # Spread a basket-wide discount across item prices before returning.
    # On by default so exactly one component owns that arithmetic.
    apply_discount: bool = os.environ.get("ENGINE_APPLY_DISCOUNT", "1") == "1"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


settings = Settings()
