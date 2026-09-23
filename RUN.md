# Running the engine

How to bring the receipt engine up from a cold machine and hand a working endpoint to the
frontend. For the API contract itself, see [`RECEIPT_API.md`](RECEIPT_API.md).

```
  frontend ──HTTPS──> ngrok tunnel ───────> API :8000 ──(localhost)──> OCR :8001
                                             Qwen3.5-2B              Surya OCR
                                             train/.venv             SuryaOCR/.venv
```

Two venvs because `surya-ocr` pins `transformers<5` and unsloth needs 5.5 — they cannot share
one interpreter. Only :8000 is ever exposed; :8001 stays bound to 127.0.0.1.

---

## One-time setup: ngrok

The tunnel is ngrok. It replaced Cloudflare quick tunnels, whose `trycloudflare.com`
hostnames stopped getting DNS records (Cloudflare error 1016) while the tunnel itself
reported connected.

1. Install: `winget install Ngrok.Ngrok`, then `ngrok update` (winget ships an old agent).
2. Sign up free at <https://dashboard.ngrok.com>, copy your authtoken, and run
   `ngrok config add-authtoken <token>`.
3. Optional but recommended: claim your free static domain (dashboard → **Domains**) and add
   it to `train/.env` as `NGROK_DOMAIN=your-name.ngrok-free.app`. Then the URL is the same
   every run and the frontend never has to change it.

---

## Every time you want to demo

### 1. Start the engine

Double-click **`train/run_demo.bat`**, or run it from a terminal. It opens three windows:

- **window 1 — Surya OCR** on :8001
- **window 2 — extraction API** on :8000
- **window 3 — ngrok tunnel** to :8000

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

### 3. Read the tunnel URL

Window 3 shows a `Forwarding  https://… -> http://localhost:8000` line. **That `https://…`
address is the URL you send.** It is your `NGROK_DOMAIN` if you set one, otherwise a new
random one each run. To open the tunnel by hand instead:

```bash
ngrok http 8000                                   # random URL
ngrok http 8000 --url=https://YOUR-DOMAIN         # your static domain
```

The process never exits — that is correct, it holds the tunnel open. Leave the window running.

### 4. Verify from outside

```bash
curl https://YOUR-URL/ready
```

If that returns `{"ok": true, ...}`, the whole path works: internet → tunnel → API → OCR → GPU.

### 5. Send the frontend two things

1. the `https://…` URL from window 3
2. the key from `train/.env`, used as the header `X-API-Key`

and remind them that browser code must also send `ngrok-skip-browser-warning: 1` (see
`RECEIPT_API.md`).

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
| The URL from yesterday is dead | Expected without `NGROK_DOMAIN`: each run gets a **new hostname**. Send the new one, or set a static domain. |
| Browser gets an HTML "You are about to visit…" page instead of JSON | ngrok's free-plan warning page. The client must send the header `ngrok-skip-browser-warning: 1`. |
| Frontend works in curl but not in the browser | The ngrok header above, or CORS — check `ENGINE_ALLOW_ORIGIN`. |
| ngrok `ERR_NGROK_4018` / authentication failed | No authtoken. Run `ngrok config add-authtoken <token>`. |
| ngrok `ERR_NGROK_108` | Another ngrok is already running on this account (free plan allows one). Close it. |
| `ngrok` "doesn't finish" | Correct behaviour. It runs until you close it. |

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
| `NGROK_DOMAIN` | *(unset)* | Set in `train/.env`. Your static ngrok domain, without `https://`. Unset = random URL per run |
| `ENGINE_CHECKPOINT` | `train/checkpoints/qwen3.5-2b-joint/checkpoint-125` | The joint adapter: extraction + a category per item. Picked in `train/reports/joint_eval.md` |
| `OCR_URL` | `http://127.0.0.1:8001/ocr` | |
| `OCR_DEVICE` | `cuda` | `cpu` frees ~3.6 GB VRAM, costs ~16s/page |
| `ENGINE_MAX_SEQ_LENGTH` | 4096 | Prompt budget is this minus `MAX_NEW_TOKENS`. Raised from 2048 so a multi-photo receipt fits — see below |
| `ENGINE_MAX_NEW_TOKENS` | 1024 | The joint answer runs ~33 tokens per item; 768 cut off receipts past ~23 items |
| `ENGINE_EMIT_SUBCATEGORY` | 0 | `1` sends the model's subcategory instead of `null`. Leave off until a checkpoint reaches 90% subcategory precision |
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
