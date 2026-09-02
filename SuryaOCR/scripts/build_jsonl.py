#!/usr/bin/env python
"""Aggregate per-receipt data/output/<id>/text.txt files into one JSONL file,
one line per receipt: {"input": "<row-reconstructed OCR text, \\n-joined>"}.

Usage:
    uv run python scripts/build_jsonl.py --output data/output --dest data/ocr_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="data/output", help="Directory containing <id>/text.txt subfolders"
    )
    parser.add_argument(
        "--dest", default="data/ocr_dataset.jsonl", help="Path to write the aggregated JSONL to"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    dest = Path(args.dest)

    # Order by manifest.jsonl (processing order) when available, falling back
    # to a sorted directory scan otherwise.
    manifest_path = output_dir / "manifest.jsonl"
    ids: list[str] = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("status") != "error":
                    ids.append(entry["output_id"])
    else:
        ids = sorted(p.name for p in output_dir.iterdir() if p.is_dir())

    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as out:
        for output_id in ids:
            text_path = output_dir / output_id / "text.txt"
            if not text_path.exists():
                continue
            text = text_path.read_text(encoding="utf-8")
            out.write(json.dumps({"input": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} record(s) to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
