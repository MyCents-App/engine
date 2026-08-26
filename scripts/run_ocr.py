#!/usr/bin/env python
"""Thin wrapper so `python scripts/run_ocr.py ...` works without installing
the package. Equivalent to the `receipt-ocr` console script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from receipt_ocr.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
