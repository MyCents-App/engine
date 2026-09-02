# MyCents — receipt extraction engine

Turns receipt photos into structured JSON:

```
photo(s) -> Surya OCR -> fine-tuned Qwen3.5-2B (QLoRA, 4-bit) -> post-processing -> JSON
```

This is stage 2 of the MyCents pipeline. It does **not** categorize anything —
the Flutter app shows this output to the user for confirmation, and the
confirmed draft goes to the MyCents backend, which runs the categorization
pipeline against Postgres.

The model's contract, and the split between what the model does and what the
deterministic code layer does, is specified in [`task.md`](task.md).

## Layout

```
app/
  api.py            FastAPI service — the deployment target
  extraction.py     model load + generation (serialized on one GPU)
  ocr_client.py     HTTP client for the Surya sidecar
  date_extract.py   receipt date parsing, incl. Buddhist -> Common Era
  config.py         all settings, environment driven
ocr_service.py      Surya sidecar (stdlib HTTP; runs in its own container)
postprocess.py      task.md §4-5 code layer: JSON salvage, reconcile, discounts
prompts.py          the frozen prompt — identical in training, eval and serving
docker/             Dockerfiles, compose, .env.example
tests/              date parsing + post-processing (no GPU needed)
demo_server.py      LOCAL DEV ONLY — phone-facing HTML page, not deployed
```

## Endpoints

```
GET  /health           liveness (open)
GET  /ready            model loaded + OCR reachable (open); 503 until both
POST /v1/extract       multipart files[] in capture order -> draft JSON
POST /v1/extract-text  {"text": "..."} -> same path without the OCR cost
```

`/v1/*` requires `X-API-Key`. The response is deliberately shaped to match the
backend's `POST /api/v1/receipts/categorize` body, so the app forwards the
user-confirmed draft without renaming a single field.

## Deploying

See **[`DEPLOY.md`](DEPLOY.md)** — Docker, GPU requirements, ngrok, and the
two things this repo deliberately does not contain (the SuryaOCR project and
the model checkpoint).

## Tests

No GPU required — `unsloth` is imported lazily inside `load_model`, so the
date and post-processing logic is testable anywhere.

```bash
uv venv .venv-test --python 3.12
uv pip install --python .venv-test/bin/python pytest rapidfuzz
.venv-test/bin/python -m pytest tests/ -q
```

## Two things to know about the output

**`reconciles: false`** means `Σ items + tax − discount` differs from the
printed total by more than 3% — usually a dropped or misread line. Surface it
on the confirm screen rather than hiding it.

**`receiptDate: null`** means no date was found. Default to today and let the
user correct it. The parser declines rather than guesses, because a wrong date
is silent: Thai receipts print the Buddhist year (2569, not 2026), and reading
it literally puts every receipt 543 years in the future.
