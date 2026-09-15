# MyCents — receipt pipeline

Photo of a receipt in, structured JSON out. Both stages of the pipeline, in one branch:

```
  photo ──> Surya OCR ──> fine-tuned Qwen3.5-2B (QLoRA, 4-bit) ──> post-processing ──> JSON
            SuryaOCR/                     train/
```

The app shows that JSON to the user for confirmation; the confirmed draft goes to the MyCents
backend, which runs categorization. **Categorization is not in this repo.**

Neither is translation. Names are returned in the language the receipt printed (almost always
Thai); the app translates them on the phone and the backend serves catalog translations. The
response carries empty `nameEn` / `shopNameEn` slots for that — see
[`RECEIPT_API.md`](RECEIPT_API.md#translation-is-not-the-engines-job).

## Layout

```
SuryaOCR/           stage 1 — OCR. Its own project, its own venv.
  src/receipt_ocr/    preprocessing, Surya engine wrapper, line reconstruction
  scripts/            batch OCR over a folder, GPU check, JSONL builder
  plan.md             design and rationale

train/              stage 2 — extraction. Its own project, its own venv.
  app/                the deployed service
    api.py              FastAPI: /health, /ready, /v1/extract, /v1/extract-text
    extraction.py       model load + generation, serialized on one GPU
    ocr_client.py       HTTP client for the OCR service
    date_extract.py     receipt date, incl. Buddhist era (2569 -> 2026)
    stitch.py           joins several photos of one receipt, dropping the overlap
    config.py           all settings, environment driven
  ocr_service.py      wraps SuryaOCR as a localhost HTTP service on :8001
  postprocess.py      JSON salvage, price normalization, reconcile, discounts
  prompts.py          the frozen prompt — identical in training, eval and serving
  task.md             the model's contract and the model/code split
  reports/            eval results behind the accuracy numbers
  eval_stitch.py      measures the overlap removal (no GPU)
  docker/             two-container deployment; see DEPLOY.md
  demo_server.py      dev only: phone-facing HTML page, not the deployed path
  tests/              date parsing, page stitching, the API contract (no GPU needed)
  test_phase2.py      post-processing

RECEIPT_API.md      the API contract — give this to whoever calls the engine
RUN.md              how to start everything and expose it
train/DEPLOY.md     containerized deployment on a GPU host
```

**Two projects, two virtual environments, on purpose.** `surya-ocr==0.17.1` pins
`transformers<5` while unsloth needs 5.5, so they cannot share one interpreter. They talk over
localhost instead: the API on :8000 calls the OCR service on :8001, which stays bound to
127.0.0.1 and is never exposed.

## Quick start

```bash
cd SuryaOCR && uv sync                       # stage 1 deps
cd ../train && uv pip install -r requirements.lock.txt -r requirements-api.txt
```

Then see [`RUN.md`](RUN.md). You also need the fine-tuned checkpoint at
`train/checkpoints/qwen3.5-2b-qlora/checkpoint-550` — it is not in git.

## Long receipts: several photos, one receipt

`/v1/extract` takes up to five images as pages of **one** receipt, in capture order. Users are
told to overlap the shots — that is how nothing falls in the gap between frames — so the
repeated lines are removed deterministically before prompting (`train/app/stitch.py`) and the
model is called once on the joined text.

The thresholds are set so that a *missed* overlap is possible and a *falsely removed* one is
not: measured at 99.8% exact reconstruction with no line ever lost, and no false seam across
190 pairs of unrelated receipts. Evidence and the tuning sweep are in
[`train/reports/stitch_overlap.md`](train/reports/stitch_overlap.md); reproduce with
`train/eval_stitch.py`.

## What to know about the output

**Accuracy is 67.6%** — every item and the total exactly right, on held-out real receipts.
Design the confirm screen so the extraction is editable, not presented as a finished record.

**`reconciles: false`** means `Σ items + tax − discount` is more than 3% from the printed
total — usually a dropped or misread line. Surface it.

**`receiptDate: null`** means no date was found. Default to today. The parser declines rather
than guesses, because a wrong date is silent.
