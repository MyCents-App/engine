"""Command-line entry point: `receipt-ocr --input data/input --output data/output`."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from receipt_ocr.config import SUPPORTED_EXTENSIONS
from receipt_ocr.engine import SuryaEngine
from receipt_ocr.pipeline import STATUS_ERROR, process_batch, process_image

logger = logging.getLogger(__name__)


def _collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"No such file or directory: {input_path}")


def _already_done(output_dir: Path, path: Path) -> bool:
    return (output_dir / path.stem / "text.txt").exists()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Surya OCR over receipt photos.")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="Reprocess images that already have output"
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Max images per GPU call (default from RECEIPT_OCR_MAX_BATCH / config)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = _collect_inputs(input_path)
    paths = all_paths if args.overwrite else [p for p in all_paths if not _already_done(output_dir, p)]
    skipped = len(all_paths) - len(paths)

    if not paths:
        print(f"Nothing to do ({skipped} already processed, use --overwrite to redo).")
        return 0

    print(f"Processing {len(paths)} image(s) ({skipped} already done, skipped)...")

    engine = SuryaEngine()
    start = time.monotonic()

    if len(paths) == 1:
        results = [process_image(paths[0], output_dir, engine=engine)]
    else:
        results = process_batch(paths, output_dir, engine=engine, max_images_per_call=args.max_batch)

    elapsed = time.monotonic() - start

    ok = sum(1 for r in results if r.status != STATUS_ERROR)
    errors = [r for r in results if r.status == STATUS_ERROR]

    print(f"\nDone in {elapsed:.1f}s: {ok}/{len(results)} succeeded.")
    for r in errors:
        print(f"  ERROR  {r.source_path}: {r.error}")

    try:
        import torch

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak GPU memory used: {peak_gb:.2f} GB")
    except ImportError:
        pass

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
