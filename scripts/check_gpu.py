#!/usr/bin/env python
"""Setup smoke test: confirms CUDA/GPU detection and that Surya's model
weights download and run end-to-end on a tiny synthetic image, without
needing any server process.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from receipt_ocr.engine import SuryaEngine  # noqa: E402


def main() -> int:
    print(f"torch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: no CUDA GPU detected -- Surya will run on CPU (slow).")

    print("\nLoading Surya models and running one sample OCR pass...")
    image = Image.new("RGB", (400, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 30), "Hello 123.45", fill="black")

    engine = SuryaEngine()
    [result] = engine.recognize_batch([image])
    lines = [line.text for line in result.text_lines]
    print(f"Detected {len(lines)} line(s): {lines}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory used: {peak_gb:.2f} GB")

    print("\nOK -- setup looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
