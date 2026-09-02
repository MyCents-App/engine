"""Surya OCR as a tiny local HTTP service, so the OCR stage and the extraction stage can run
in separate Python environments and on separate devices.

Why a separate process at all: the SuryaOCR project pins transformers 4.57.6, while this
project's venv is on transformers 5.5.0 for unsloth. They cannot share one interpreter. Running
Surya in its own venv behind a socket keeps both working and matches the deployment shape
anyway (OCR on CPU, the extraction model on GPU).

Deliberately stdlib-only -- no FastAPI, no uvicorn, no new packages in either venv. The client
POSTs raw image bytes as the request body (not multipart), which needs no parser.

Run from the SuryaOCR project, using ITS venv:

    cd D:\\Documents\\SuryaOCR
    set TORCH_DEVICE=cpu
    .venv\\Scripts\\python.exe D:\\Documents\\train\\ocr_service.py

Set TORCH_DEVICE=cuda to run OCR on the GPU instead (measured on this box: ~3.5s/receipt vs
~20s on CPU, at a cost of ~3.6GB VRAM).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

HOST = os.environ.get("OCR_HOST", "127.0.0.1")
PORT = int(os.environ.get("OCR_PORT", "8001"))
# Default: the sibling SuryaOCR project next to this one, so moving the pair of folders
# does not break the link. Override with SURYA_PROJECT to point elsewhere.
SURYA_PROJECT = Path(os.environ.get("SURYA_PROJECT",
                                    str(Path(__file__).resolve().parent.parent / "SuryaOCR")))
MAX_BYTES = 25 * 1024 * 1024

# Surya's own batch-size env vars are read once, at first import of surya.*, so they must be
# set before receipt_ocr.config pulls it in. On CPU the big GPU-sized batches only waste RAM.
if os.environ.get("TORCH_DEVICE", "cpu") == "cpu":
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", "8")
    os.environ.setdefault("DETECTOR_BATCH_SIZE", "2")

sys.path.insert(0, str(SURYA_PROJECT / "src"))
os.chdir(SURYA_PROJECT)  # receipt_ocr.config loads a .env relative to cwd

from receipt_ocr.engine import SuryaEngine  # noqa: E402
from receipt_ocr.pipeline import process_image  # noqa: E402

_engine: SuryaEngine | None = None
_lock = Lock()  # one receipt at a time: the models are not safe to call concurrently
_tmpdir = tempfile.mkdtemp(prefix="ocr_service_")

_EXT_BY_TYPE = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
    "image/webp": ".webp", "image/heic": ".heic", "image/heif": ".heif",
    "image/bmp": ".bmp", "image/tiff": ".tif",
}


def engine() -> SuryaEngine:
    global _engine
    if _engine is None:
        _engine = SuryaEngine()
        _ = _engine.recognition  # force the weights to load now, not on the first request
    return _engine


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "device": os.environ.get("TORCH_DEVICE", "auto")})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ocr":
            self._send(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send(400, {"error": "empty body; POST the raw image bytes"})
            return
        if length > MAX_BYTES:
            self._send(413, {"error": f"image too large ({length} bytes, max {MAX_BYTES})"})
            return

        data = self.rfile.read(length)
        suffix = _EXT_BY_TYPE.get((self.headers.get("Content-Type") or "").split(";")[0].strip(), ".jpg")
        path = Path(_tmpdir) / f"upload_{time.time_ns()}{suffix}"
        path.write_bytes(data)

        try:
            t0 = time.time()
            with _lock:
                result = process_image(path, _tmpdir, engine=engine())
            elapsed = time.time() - t0

            if result.status == "error":
                self._send(500, {"error": result.error or "OCR failed"})
                return
            self._send(200, {
                "text": result.text,
                "status": result.status,
                "num_lines": result.num_lines,
                "avg_confidence": round(result.avg_confidence, 4),
                "ocr_seconds": round(elapsed, 2),
            })
        except Exception as exc:  # noqa: BLE001 -- one bad upload must not kill the service
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            path.unlink(missing_ok=True)

    def log_message(self, fmt, *args):
        sys.stderr.write("  [ocr] " + (fmt % args) + "\n")

    def log_error(self, fmt, *args):
        # See demo_server.log_error: a dropped idle keep-alive connection logs as
        # "Request timed out" but affects no request. Real errors still print.
        msg = fmt % args
        if "Request timed out" in msg or "Connection reset" in msg:
            return
        sys.stderr.write("  [ocr] ERROR " + msg + "\n")


def main() -> int:
    device = os.environ.get("TORCH_DEVICE", "auto")
    print(f"Loading Surya (TORCH_DEVICE={device}) ...", flush=True)
    t0 = time.time()
    engine()
    print(f"Surya ready in {time.time() - t0:.1f}s", flush=True)
    print(f"OCR service listening on http://{HOST}:{PORT}  (POST raw image bytes to /ocr)", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
