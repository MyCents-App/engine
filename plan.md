# Receipt OCR Stage (Surya OCR)

## Context

This repo will hold the first stage of an expense-tracking app's receipt pipeline: take a receipt photo (Thai + English, phone photos and thermal prints) and produce OCR text suitable for a *later* extraction stage (merchant/items/prices/total) that is out of scope here. The repo is currently empty (just a README), so this is a greenfield build.

Key research finding that shapes this plan: **current Surya OCR (`surya-ocr` ≥0.20, "Surya 2", as of mid-2026) was rewritten around a VLM served via `vllm` (NVIDIA GPU, needs Docker) or `llama.cpp` (CPU/Mac)**. `vllm` has no official native Windows support (Linux/WSL2 only), which is a poor fit for "my own PC" on Windows with a single GPU. The user confirmed: pin to the **last pre-rewrite version, `surya-ocr==0.17.1`** (released 2026-01-30), which is pure PyTorch (`FoundationPredictor` + `RecognitionPredictor` + `DetectionPredictor`, no server/Docker/vllm) and installs directly on native Windows + CUDA. This version already does automatic multilingual OCR (90+ languages incl. Thai) with no `langs` list needed, and outputs per-line `text`, `bbox`, `polygon`, `confidence` — exactly what's needed here. Verified directly from PyPI's JSON release history and the `v0.17.1` GitHub tag's README/pyproject.

The main engineering value-add beyond calling Surya is: (1) preprocessing tuned for phone photos + faded thermal prints, (2) reconstructing Surya's per-line detections into receipt-shaped **rows** (Surya often detects an item name and its price as two separate line boxes when there's a gap/dot-leader between them — these must be re-joined so item + price stay together and in order), and (3) a robust batch pipeline with sane VRAM budgeting for a 12GB card and a clear, debuggable output structure for the next (extraction) stage to consume.

## Environment & Dependencies

- **uv**-managed project, Python 3.11, `pyproject.toml`.
- `surya-ocr==0.17.1` (pinned — do not let this float to ≥0.20).
- `torch` installed from the PyTorch CUDA wheel index (`cu128` or newer, e.g. `--index-url https://download.pytorch.org/whl/cu128`) — required for RTX 5070 (Blackwell/sm_120) support; the plain PyPI `torch` wheel may be CPU-only or lack Blackwell kernels. Satisfies surya's `torch ^2.7.0` constraint.
- **Found during implementation:** `surya-ocr==0.17.1` only declares `transformers>=4.56.1` (no upper bound). `transformers` crossed a breaking 5.0 major version 4 days after 0.17.1 shipped, and 0.17.1's model code isn't compatible with it (`AttributeError: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'`). Pinned `transformers>=4.56.1,<5` to resolve to the last compatible 4.x release.
- `pillow-heif` — registers a Pillow opener for iPhone HEIC/HEIF uploads (mobile app users will upload/capture in mixed formats: JPEG, PNG, HEIC, WEBP).
- `opencv-python-headless` (already a transitive surya dep — reuse it directly) for contrast enhancement.
- Everything else (Pillow, numpy, pydantic, click) comes in transitively via surya.
- **Setup verification step (do this first, before building the pipeline):** install deps, then run a tiny smoke script confirming `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` correctly report the RTX 5070, and that `surya-ocr==0.17.1` downloads its model weights and runs on one sample image without needing any server process. If weights fail to resolve (Vik Paruchuri's HF repos could theoretically be retired in favor of Surya 2), fall back and flag to the user rather than silently downgrading further.
- Note (not a blocker, just FYI): Surya's model weights are licensed under a modified OpenRAIL-M (free for personal/research/startups under $2M funding/revenue) and the code is GPL-3.0. Fine for personal use; worth knowing if this app is ever commercialized at scale.

## Project Structure

```
SuryaOCR/
  pyproject.toml
  .env.example                # RECOGNITION_BATCH_SIZE, DETECTOR_BATCH_SIZE, TORCH_DEVICE, etc.
  src/receipt_ocr/
    __init__.py
    config.py                 # batch sizes, image size clamps, row-grouping tolerance, thresholds
    preprocess.py              # load_image(): HEIC/format handling, EXIF orientation, resize, CLAHE contrast
    engine.py                  # SuryaEngine: lazy-loaded singleton wrapping Foundation/Recognition/DetectionPredictor
    reconstruct.py             # groups Surya's text_lines into receipt rows, preserving item+price adjacency
    pipeline.py                # process_image() / process_batch(): orchestrates preprocess -> engine -> reconstruct -> save
    io_utils.py                # writes raw.json/text.txt per image + manifest.jsonl
  scripts/
    run_ocr.py                 # CLI entrypoint (argparse)
  data/
    input/  .gitkeep
    output/ .gitkeep
  tests/
    test_reconstruct.py        # unit tests for row-grouping on synthetic bbox data (no GPU needed)
```

## Component Design

### `preprocess.py`
1. Register HEIF opener (`pillow_heif.register_heif_opener()`) once at import time so `Image.open` transparently handles `.heic`/`.heif`.
2. `ImageOps.exif_transpose()` to fix phone-camera orientation from EXIF.
3. Convert to RGB (drop alpha/CMYK).
4. Resize: clamp longest side to `MAX_LONG_SIDE` (default 2560px) to bound VRAM/latency; upscale (bicubic) if longest side is below `MIN_LONG_SIDE` (default 1000px) — common for tightly-cropped thermal receipt photos where small text needs more pixels.
5. Mild CLAHE contrast enhancement on the L channel (LAB colorspace) via OpenCV — helps faded thermal prints. Always applied, but mild, since it's a DL OCR model (do **not** binarize — that hurts recognition, unlike classical OCR).
6. ~~Orientation fallback for images with no/incorrect EXIF~~ — **implemented but disabled by default; see "Deviations from this plan" below.** EXIF-based correction (step 2) remains always-on and handles the normal case.

### Deviations from this plan (found during implementation)

- **Orientation auto-correction guess is off by default.** The planned "detect at 0°, retry other rotations if it looks bad" heuristic doesn't work: Surya's detector finds plausible-looking line boxes on sideways text too, so a wrong orientation can still look fine by line-count/confidence. A recognition-confidence-based version was tried instead (score all 4 rotations by mean OCR confidence, keep the best), but testing showed the gap between correct and incorrect orientations is often only a few hundredths — it confidently flipped an already-upright test image to 180°. Shipping a guess that occasionally corrupts good images is worse than not guessing, so `RECEIPT_OCR_ORIENTATION_FALLBACK` defaults to `0`. The code path (`SuryaEngine.correct_orientation`) is still there, opt-in, for anyone who verifies it helps on their own data. The `low_confidence` manifest status is the safer fallback signal for "this photo may need a retake."

### `engine.py`
- `SuryaEngine` loads `FoundationPredictor`, `RecognitionPredictor(foundation_predictor)`, `DetectionPredictor()` once per process (expensive to reload weights).
- Batch sizes tuned for 12GB VRAM via env vars Surya already reads (`RECOGNITION_BATCH_SIZE`, `DETECTOR_BATCH_SIZE`): defaults **96** (~3.8GB, at ~40MB/item) and **8** (~3.5GB, at ~440MB/item) respectively — well under Surya's own defaults (512/36, which target 20GB/16GB and would OOM a 12GB card). Document these as tunable in `.env.example`.
- On `torch.cuda.OutOfMemoryError`: halve the relevant batch-size env var and retry once; if it still fails, mark that image as an error in the manifest rather than crashing the whole batch.
- Confirmed: `RecognitionPredictor.__call__` takes a `math_mode: bool` param directly (the Python-API equivalent of the CLI's `--disable_math`). Wired through as `settings.math_mode`, default off (`RECEIPT_OCR_MATH_MODE=0`) since receipts have no legitimate math notation.
- Runs `recognition_predictor(images, det_predictor=detection_predictor)` — accepts a **list** of images per call, batching internally. `process_batch()` should pass a whole folder's images in one call (chunked to a configurable max per call) rather than looping single-image calls, for throughput.

### `reconstruct.py` (the "keep item + price together" logic)
Given one image's `text_lines` (each with `bbox=[x1,y1,x2,y2]`, `text`, `confidence`):
1. Sort lines by vertical center `y_center = (y1+y2)/2`.
2. Greedily group into **rows**: a line joins the current row if its vertical span overlaps the row's running vertical span above a threshold (default 40%) — this re-merges an item name and its price when Surya detected them as separate boxes due to a gap/dot-leader/wide spacing.
3. Within a row, sort fragments left-to-right by `x1`.
4. Join fragments into one row string: single space for normally-spaced fragments; **two spaces** where the horizontal gap between adjacent fragments exceeds ~3x the row's average character width (signals a dot-leader/column gap between name and price) — keeps them visually distinguishable without inventing delimiters that could confuse the downstream extraction model.
5. Rows ordered top-to-bottom become the reconstructed receipt text (one row per line), which is exactly what the next stage should read.

This is purely geometric (language-agnostic), so it works the same for Thai item names and English/numeric prices on the same line.

### `pipeline.py` / `io_utils.py` — output structure
Per input image (keyed by a caller-supplied `output_id`, defaulting to filename stem, to avoid collisions on duplicate uploads):
```
data/output/<output_id>/
  raw.json     # {source_image, image_size, preprocessing_applied, text_lines:[{text,bbox,polygon,confidence}, ...]}
  text.txt     # reconstructed rows, UTF-8, one row per line — the direct input for the extraction stage
data/output/manifest.jsonl   # one line per processed image: {file, output_id, status: ok|empty|low_confidence|error, num_lines, avg_confidence, error, duration_s}
```
`raw.json` preserves full traceability (bboxes/confidence) for debugging misreads without re-running OCR; `text.txt` is the clean artifact to hand to the LLM extraction stage; `manifest.jsonl` gives a batch-run audit trail (important since this feeds a mobile upload flow where some photos will fail or be unusable).

Public API (importable by the rest of the app, e.g. an upload handler):
```python
from receipt_ocr.pipeline import process_image, process_batch
result = process_image(path, output_dir="data/output", output_id=None)
# result: text (str), lines (list), avg_confidence (float), status, output_paths
```

### `scripts/run_ocr.py` (CLI)
```
uv run python scripts/run_ocr.py --input data/input --output data/output [--overwrite] [--max-batch 16]
```
Accepts a single file or a directory; processes directories via `process_batch` for GPU throughput.

## Edge Cases

- Corrupt/unreadable file → caught per-image, logged as `error` in manifest, batch continues.
- Zero text detected → still writes empty `text.txt` + `raw.json`, manifest status `empty`.
- Low average confidence (< 0.5 default) → manifest status `low_confidence`, a useful signal the app could use to prompt "retake photo."
- Sideways scans without EXIF → handled by the detection-based orientation fallback above.
- GPU OOM → batch-size backoff + retry, described above.
- Duplicate filenames across uploads → caller passes a unique `output_id` (e.g. upload UUID) instead of relying on filename.
- Out of scope for v1 (documented as known limitations): multiple receipts in one photo, extremely long panorama/tiled receipts.

## Verification

1. Unit tests (`tests/test_reconstruct.py`) on synthetic bbox fixtures — no GPU required, fast, covers the row-grouping/joining logic including the "separated item+price" case.
2. Setup smoke test: confirm CUDA/GPU detection and that model weights download and run once end-to-end on a sample image.
3. Manual end-to-end pass: drop a handful of real sample receipts (at least one Thai thermal print, one English phone photo) into `data/input/`, run the CLI, and inspect `text.txt` for correct Thai/English UTF-8 rendering and that item names stay on the same row as their prices.
4. Print peak GPU memory (`torch.cuda.max_memory_allocated()`) at the end of a batch CLI run so the user can confirm they're comfortably within the 12GB budget and retune batch sizes if needed.
