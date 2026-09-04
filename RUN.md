# Running the engine

How to bring the receipt engine up from a cold machine and hand a working endpoint to the
frontend. For the API contract itself, see [`RECEIPT_API.md`](RECEIPT_API.md).

```
  frontend ──HTTPS──> cloudflare tunnel ──> API :8000 ──(localhost)──> OCR :8001
                                             Qwen3.5-2B              Surya OCR
                                             train/.venv             SuryaOCR/.venv
```

Two venvs because `surya-ocr` pins `transformers<5` and unsloth needs 5.5 — they cannot share
one interpreter. Only :8000 is ever exposed; :8001 stays bound to 127.0.0.1.

---

## Every time you want to demo

### 1. Start the engine

Double-click **`train/run_demo.bat`**, or run it from a terminal. It opens two windows:

- **window 1 — Surya OCR** on :8001
- **window 2 — extraction API** on :8000

It reads `ENGINE_API_KEY` from `train/.env` and refuses to start without one.

**Wait for window 2 to log `warm-up generation done`.** That takes roughly a minute: ~11s to
load the model into VRAM, ~5s for the warm-up generation, plus Surya's own startup. Until
then requests fail.

### 2. Check it locally before exposing it

```bash
curl http://localhost:8000/ready
```

Must return `{"ok": true, "model_loaded": true, "ocr_reachable": true, ...}`. A **503** means
something is still loading or the OCR window died — fix it here, before the tunnel.

### 3. Open the tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a box with a `https://….trycloudflare.com` URL. **That is the URL you send.**

The process never exits — that is correct, it holds the tunnel open. Leave the window running.
If a `precheck ... hard_fail=true` line appears, ignore it: it means one Cloudflare region was
unreachable, not that the tunnel failed. Verify with step 4 instead of reading the log.

### 4. Verify from outside

```bash
curl https://YOUR-URL.trycloudflare.com/ready
```

If that returns `{"ok": true, ...}`, the whole path works: internet → tunnel → API → OCR → GPU.

### 5. Send the frontend two things

1. the `https://….trycloudflare.com` URL
2. the key from `train/.env`, used as the header `X-API-Key`

Plus `RECEIPT_API.md` if they don't have it yet.

### Shutting down

Close the windows. Closing one stops that service; closing the tunnel window kills the public
URL immediately.

---

## Things that will confuse you at 9am

| Symptom | Cause |
|---|---|
| `{"detail":"Not Found"}` at the root URL | Expected — there is no page at `/`. Use `/ready`. |
| `{"detail":"Method Not Allowed"}` | You opened `/v1/extract` in a browser. It is POST-only. |
| `/ready` returns 503 | Model still loading, or the OCR window is not up. Wait, then check window 1. |
| 401 | Missing or wrong `X-API-Key`. `/health` and `/ready` don't need it; `/v1/*` do. |
| The URL from yesterday is dead | Expected. A quick tunnel issues a **new hostname every run**. Send the new one. |
| Frontend works in curl but not in the browser | CORS. Should be handled; if not, check `ENGINE_ALLOW_ORIGIN`. |
| `cloudflared` "doesn't finish" | Correct behaviour. It runs until you close it. |

---

## Configuration

`train/.env` holds the shared secret and is gitignored. Regenerate the key with:

```bash
train/.venv/Scripts/python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Everything else is environment driven, set by `run_demo.bat` — no host paths in the code:

| Variable | Default | Notes |
|---|---|---|
| `ENGINE_API_KEY` | *(required)* | Empty disables auth entirely. Never do that behind a tunnel. |
| `ENGINE_CHECKPOINT` | `train/checkpoints/qwen3.5-2b-qlora/checkpoint-550` | |
| `OCR_URL` | `http://127.0.0.1:8001/ocr` | |
| `OCR_DEVICE` | `cuda` | `cpu` frees ~3.6 GB VRAM, costs ~16s/page |
| `ENGINE_MAX_SEQ_LENGTH` | 4096 | Prompt budget is this minus `MAX_NEW_TOKENS`. Raised from 2048 so a multi-photo receipt fits — see below |
| `ENGINE_MAX_NEW_TOKENS` | 768 | |
| `ENGINE_MAX_PAGES` | 5 | Photos accepted as pages of one receipt |
| `ENGINE_STITCH_PAGES` | 1 | Remove the overlap between consecutive photos. `0` concatenates instead — only for comparison |
| `ENGINE_STITCH_WINDOW` | 60 | Longest overlap searched, in lines |
| `ENGINE_ALLOW_ORIGIN` | `*` | Narrow once there is a real frontend origin |
| `ENGINE_WARMUP` | 1 | The startup generation that absorbs first-call latency |

**Why `ENGINE_MAX_SEQ_LENGTH` is 4096 and not the 2048 the model was trained at.** Training
length caps what the model learned from, not what it can be served with, and Qwen3.5's own
context is far longer than either. Nothing about the published accuracy numbers changes —
every eval prompt was under 1000 tokens and is scored identically. What it buys is room: at
2048 the prompt budget is ~1280 tokens, which a single dense receipt already reaches (the
review set runs 560–981), so a two- or three-photo receipt would 413 on the exact case the
multi-photo path exists to serve. The cost is a little KV-cache VRAM.

Measured on the RTX 5070: model 2.0 GB VRAM, both services resident **4.9 GB of 12 GB**.

---

## The phone page

`train/run_page.bat` serves the old phone-facing HTML page (`demo_server.py`) on :8080, for
eyeballing the pipeline over Wi-Fi. It needs the OCR service already running, and it loads a
**second** copy of the model into VRAM — fine for a look, don't leave both running.

---

## Running the tests

No GPU needed — date parsing, page stitching, post-processing, and the API contract with the
model and the OCR sidecar stubbed out:

```bash
uv venv .venv-test --python 3.11
uv pip install --python .venv-test/Scripts/python.exe pytest fastapi httpx python-multipart rapidfuzz
cd train && ../.venv-test/Scripts/python.exe -m pytest tests/ test_phase2.py -q
```

`fastapi`/`httpx`/`python-multipart` are only needed for `tests/test_api_pages.py`, which
skips itself without them; `rapidfuzz` only for `test_phase2.py`.

To re-measure the overlap removal against the review set (needs `SuryaOCR/data/review/`,
which `review_photos.py` writes):

```bash
cd train && ../.venv-test/Scripts/python.exe eval_stitch.py
```
