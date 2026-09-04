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
    # 4096, not the 2048 the QLoRA fine-tune was trained at. Training length
    # caps what the model LEARNED from, not what it can be served with, and
    # Qwen3.5's own context is far longer than either number — so nothing
    # about the published eval figures changes (every eval prompt was under
    # 1000 tokens and is scored identically here). What it buys is headroom:
    # the prompt budget is max_seq_length - max_new_tokens, and at 2048 that
    # is ~1280 tokens, which a single dense receipt already reaches and a
    # two- or three-photo receipt clears easily. Serving a long receipt is
    # the point of the multi-photo path; rejecting it with a 413 would not
    # be serving it. Costs a little KV-cache VRAM, nothing else.
    max_seq_length: int = _int("ENGINE_MAX_SEQ_LENGTH", 4096)
    max_new_tokens: int = _int("ENGINE_MAX_NEW_TOKENS", 768)
    load_in_4bit: bool = os.environ.get("ENGINE_LOAD_IN_4BIT", "1") == "1"

    # --- OCR sidecar ---
    ocr_url: str = os.environ.get("OCR_URL", "http://ocr:8001/ocr")
    ocr_timeout_s: float = _float("OCR_TIMEOUT_S", 300.0)

    # --- API ---
    # Browsers block a cross-origin response that carries no
    # Access-Control-Allow-Origin header, and the preflight OPTIONS must be
    # answered too, or the real POST is never sent. "*" suits an endpoint that
    # holds no user data; narrow it to a specific origin when there is one.
    allow_origin: str = os.environ.get("ENGINE_ALLOW_ORIGIN", "*")
    # Empty disables auth. NEVER leave it empty when the service is reachable
    # through a tunnel — see DEPLOY.md.
    api_key: str = os.environ.get("ENGINE_API_KEY", "")
    max_upload_bytes: int = _int("ENGINE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
    # Photos of ONE receipt accepted per request. A long receipt gets shot in
    # parts; three is the usual worst case and five leaves room. The cap is
    # there so a client looping over a folder cannot post fifty images as one
    # receipt and tie up the GPU.
    max_pages: int = _int("ENGINE_MAX_PAGES", 5)

    # Look for the overlap between consecutive photos and drop the repeated
    # lines before prompting (app/stitch.py). Off means plain concatenation,
    # which duplicates every item in the overlap. Only turn it off to compare
    # against that behaviour.
    stitch_pages: bool = os.environ.get("ENGINE_STITCH_PAGES", "1") == "1"
    # Longest overlap searched for, in lines. Raise only for receipts longer
    # than the search window; the cost is quadratic in this number.
    stitch_window: int = _int("ENGINE_STITCH_WINDOW", 60)

    # Generation is serialized on one GPU; requests beyond this wait, and
    # requests beyond the queue are rejected rather than piling up until the
    # client times out with no explanation.
    max_queue: int = _int("ENGINE_MAX_QUEUE", 8)
    queue_timeout_s: float = _float("ENGINE_QUEUE_TIMEOUT_S", 240.0)

    # Spread a basket-wide discount across item prices before returning.
    # On by default so exactly one component owns that arithmetic.
    apply_discount: bool = os.environ.get("ENGINE_APPLY_DISCOUNT", "1") == "1"

    # One throwaway generation at startup. The first generate() on a freshly
    # loaded model pays ~5s of CUDA kernel selection on top of the usual ~3s;
    # spending it here means the first real receipt is as fast as the tenth.
    warmup: bool = os.environ.get("ENGINE_WARMUP", "1") == "1"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


settings = Settings()
