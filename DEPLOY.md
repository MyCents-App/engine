# Deploying the extraction engine

The engine runs on a GPU machine that is **not** the developer's laptop. The
Flutter app and the MyCents backend run elsewhere and call it over a tunnel.

```
  Flutter app ─┐
               ├─ HTTPS ─> ngrok ─> api :8000 ─(private)─> ocr :8001
  backend ─────┘                      │                      │
                                 Qwen3.5-2B              Surya OCR
                                 (unsloth, 4-bit)        (transformers <5)
```

Two containers, because `surya-ocr==0.17.1` pins `transformers<5` and unsloth
needs 5.5 — they cannot share an interpreter. Only `api` is published; `ocr`
has no authentication of its own and must stay on the private network.

---

## 1. What the machine needs

| | |
|---|---|
| **NVIDIA GPU** | Required, and confirmed present on the target host. The checkpoint is 4-bit quantized via bitsandbytes, which is CUDA-only. §7 is a contingency note only. |
| VRAM | ~6 GB is comfortable: ~2 GB for the 4-bit model, ~3.6 GB for Surya on CUDA. Set `OCR_DEVICE=cpu` to trade ~16s/page for that 3.6 GB. |
| Disk | ~25 GB. The CUDA images are large, plus ~1 GB of Surya weights and the checkpoint. |
| Software | Docker Engine + Compose v2, and the **NVIDIA Container Toolkit** so `--gpus` works. |

Verify the GPU is visible to Docker before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If that does not print your card, fix it first — nothing below will work.

## 2. Two things this repo does not contain

Both are deliberate: they are large, and they change on a different cadence
than the code.

**a) The SuryaOCR project** (the `receipt_ocr` package). `ocr_service.py`
imports it. Put it on the host and point `SURYA_PROJECT_PATH` at it. Compose
passes it to the OCR image as a named build context.

**b) The fine-tuned checkpoint.** Mounted read-only at `/models`. Set
`CHECKPOINT_PATH` to the directory *containing* the checkpoint folder:

```
/opt/mycents/models/qwen3.5-2b-qlora/checkpoint-550
└────────┬────────┘
    CHECKPOINT_PATH=/opt/mycents/models
    ENGINE_CHECKPOINT=/models/qwen3.5-2b-qlora/checkpoint-550
```

## 3. Configure

```bash
cd docker
cp .env.example .env
```

Fill in `SURYA_PROJECT_PATH`, `CHECKPOINT_PATH`, and generate a key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

> **`ENGINE_API_KEY` is not optional in this deployment.** The service is
> reachable from the public internet through the tunnel and it fronts a GPU.
> Leaving the key empty disables auth entirely — the service logs a loud
> warning at startup if you do, and `/ready` reports `auth_enabled: false`.

## 4. Build and run

```bash
docker compose build          # 10-20 min the first time; the CUDA layers are big
docker compose up -d
docker compose logs -f api
```

First start is slow: Surya downloads ~1 GB of weights (cached in the
`hf-cache` volume afterwards) and the model loads into VRAM. The healthcheck
allows for this — `api` reports healthy only once `/health` answers.

```bash
curl -fsS http://localhost:8000/health          # {"ok": true}
curl -fsS http://localhost:8000/ready | jq      # model_loaded + ocr_reachable
```

`/ready` returns **503** until both are true. Use it, not `/health`, to decide
whether the engine can actually serve a receipt.

## 5. Expose it

```bash
docker compose --profile tunnel up -d
docker compose logs ngrok | grep -i 'url='
```

**Reserve a domain** (`NGROK_DOMAIN`) on the ngrok dashboard. Without one the
public URL changes on every restart, and both the Flutter app and the backend
have to be reconfigured each time.

Then point the backend at it — in `server/.env`:

```
ENGINE_URL=https://your-domain.ngrok.app
ENGINE_API_KEY=<the same key>
```

## 6. Using the API

Auth: `X-API-Key: <key>` on `/v1/*`. `/health` and `/ready` are open so a
container probe does not need a secret.

```bash
# one receipt
curl -X POST https://your-domain.ngrok.app/v1/extract \
     -H "X-API-Key: $ENGINE_API_KEY" \
     -F "files=@receipt.jpg"

# a long receipt, IN CAPTURE ORDER
curl -X POST https://your-domain.ngrok.app/v1/extract \
     -H "X-API-Key: $ENGINE_API_KEY" \
     -F "files=@page1.jpg" -F "files=@page2.jpg"

# replay saved OCR text — no GPU OCR cost, same model path
curl -X POST https://your-domain.ngrok.app/v1/extract-text \
     -H "X-API-Key: $ENGINE_API_KEY" -H "Content-Type: application/json" \
     -d '{"text": "7-ELEVEN\nมาม่า 18.00\nรวม 18.00"}'
```

The response is shaped to match the backend's
`POST /api/v1/receipts/categorize` body, so the app forwards the
user-confirmed draft with no field renaming:

```json
{
  "ok": true,
  "shopName": "7-Eleven",
  "receiptDate": "2026-09-02",
  "currency": "THB",
  "totalAmount": "46.01",
  "taxAmount": "3.01",
  "basketDiscount": null,
  "items": [{"name": "มาม่า", "price": "18.00"}],
  "ocrTexts": ["..."],
  "reconciles": true,
  "reconcileStatus": "ok",
  "engine": {"pages": 1, "ocr_seconds": 3.4, "model_seconds": 2.1,
             "prompt_tokens": 412, "token_budget": 1280}
}
```

Two fields worth acting on in the client:

- **`reconciles: false`** — the numbers do not add up (`Σ items + tax −
  discount ≠ total`, outside 3%). Usually a dropped or misread line. Draw the
  user's attention to the totals on the confirm screen.
- **`receiptDate: null`** — no date was found. Default to today and let the
  user correct it. A wrong date is worse than no date, so the parser declines
  rather than guesses.

### Page order matters

Pages are OCR'd separately and their text concatenated **in the order sent**
before a single model call, so an item whose name is cut off at the bottom of
page 1 and whose price appears at the top of page 2 is rejoined. Never let the
client reorder the images.

### 413 on a long receipt

The prompt budget is `ENGINE_MAX_SEQ_LENGTH − ENGINE_MAX_NEW_TOKENS` (1280
tokens by default). Over that, the request is **rejected** rather than
truncated — truncation silently drops the end of the receipt, and the model
still returns confident, well-formed JSON missing its last items. The error
reports the actual token count. Raise `ENGINE_MAX_SEQ_LENGTH` if the GPU has
headroom, and re-measure VRAM.

## 7. Serving stack — decided, and the contingency

**The engine serves the checkpoint through unsloth/transformers on CUDA.**
That is the stack the model was fine-tuned and evaluated with, using the exact
prompt in `prompts.py`, so the published accuracy numbers describe what this
service actually does. The target host has a suitable GPU, so this is the
path — nothing below is in play today.

**Not Ollama.** Ollama serves GGUF, so using it would mean converting the
checkpoint and swapping the serving stack underneath a model whose numbers
were measured on a different one. Every eval figure would need re-checking
before it could be trusted, and nothing in this repo performs that
conversion. If Ollama is ever wanted (say, to consolidate with other models
on the same host), treat it as a project with its own re-evaluation, not a
deployment tweak.

### Contingency, if the GPU ever becomes unavailable

1. **Another GPU host.** Least work; the compose file is unchanged.
2. **Full-precision CPU** — `ENGINE_LOAD_IN_4BIT=0` plus a CPU base image.
   Tens of seconds per receipt, ~5 GB RAM. Usable for a demo, not for use.
3. **GGUF + llama.cpp/Ollama**, with the re-evaluation caveat above.

## 8. Operating notes

```bash
docker compose ps
docker compose logs -f api
docker compose restart api        # after changing .env
docker compose down               # keeps the hf-cache volume
docker compose down -v            # ALSO deletes cached weights — a full re-download
```

- **One worker, on purpose.** The process holds the model in VRAM and
  serializes generation; a second worker would load a second copy of the
  weights and race for the same GPU.
- **Queue limits.** `ENGINE_MAX_QUEUE` callers may wait for the GPU; beyond
  that the service returns 503 immediately rather than letting everyone time
  out with no explanation. `/ready` reports `queue_depth`.
- **`demo_server.py` is a local development tool** — it serves a phone-facing
  HTML page and is deliberately **not** in either image. The deployed engine
  is API-only.
