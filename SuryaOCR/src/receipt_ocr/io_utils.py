"""Output writing: per-receipt raw.json + text.txt, and a batch manifest."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_manifest_lock = threading.Lock()


def result_dir(output_dir: str | Path, output_id: str) -> Path:
    return Path(output_dir) / output_id


def write_raw_json(output_dir: str | Path, output_id: str, payload: dict[str, Any]) -> Path:
    directory = result_dir(output_dir, output_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "raw.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_text(output_dir: str | Path, output_id: str, text: str) -> Path:
    directory = result_dir(output_dir, output_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "text.txt"
    path.write_text(text, encoding="utf-8")
    return path


def append_manifest(output_dir: str | Path, entry: dict[str, Any]) -> Path:
    """Append one line to data/output/manifest.jsonl. Thread-safe within a
    single process; each line is a complete, independent JSON object.
    """
    path = Path(output_dir) / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _manifest_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return path
