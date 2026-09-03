# Receipt Extraction API

Send a photo of a receipt, get back structured JSON — merchant, date, line items with prices,
tax and the total. Runs on a GPU workstation, exposed over a Cloudflare tunnel.

---

## The two things you need

**1. Base URL.** The tunnel issues a **new hostname every time the server is started**, so
don't hardcode it. You'll get the current one before each session.

```js
const BASE = "https://SOMETHING.trycloudflare.com";
```

**2. API key.** Sent as an `X-API-Key` header on every `/v1/*` call. Sent to you separately —
it is deliberately not written down in this file.

```js
const KEY = "...";   // paste the key you were given
```

The service is on the public internet and fronts a GPU, so `/v1/*` is closed. Without the
header you get **401**. `/health` and `/ready` are open and need no key.

---

## `POST /v1/extract` — photo in, JSON out

The endpoint you actually need. Send the image **any** of these ways — all produce identical
results.

### Option A: FormData (normal upload)

```js
const fd = new FormData();
fd.append("files", photoFile);         // any field name works: files, file, image...

const res = await fetch(BASE + "/v1/extract", {
  method: "POST",
  headers: { "X-API-Key": KEY },       // do NOT set Content-Type yourself
  body: fd,
});
const data = await res.json();
```

### Option B: raw bytes

```js
const res = await fetch(BASE + "/v1/extract", {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "image/jpeg" },
  body: photoFile,                     // Blob / File / ArrayBuffer
});
```

JPEG, PNG, WEBP and iPhone HEIC all work. Max **25 MB** per image.

### Long receipts: several photos, one receipt

Append more than one file. Each page is OCR'd separately and the text joined **in the order
you send them** before a single model call, so an item whose name is cut off at the bottom of
page 1 and whose price appears at the top of page 2 is rejoined correctly.

```js
fd.append("files", page1);
fd.append("files", page2);             // capture order is load-bearing — never shuffle
```

Maximum **5 pages** per receipt. Send pages of *one* receipt this way — two unrelated
receipts in a single request produces nonsense (and `reconciles: false`).

---

## `POST /v1/extract-text` — same model, no OCR

You supply the OCR text instead of a photo. Skips ~2-3s of OCR and is fully deterministic, so
it's the better thing to develop against before wiring up the camera.

```js
fetch(BASE + "/v1/extract-text", {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ text: "7-ELEVEN\ncoke 20.00\nTotal 20.00" }),
});
```

Byte-identical prompting and post-processing to `/v1/extract`, so a result here is exactly
what the photo path would produce from that text.

---

## `GET /health` and `GET /ready`

No key required.

- **`/health`** → `{"ok": true}` as soon as the process is up. Tells you the tunnel works.
- **`/ready`** → 200 only when the model is loaded **and** OCR is reachable; **503** until
  then, which is roughly the first minute after startup.

**Use `/ready`, not `/health`, to decide whether the engine can actually serve a receipt.**

```json
{"ok": true, "model_loaded": true, "ocr_reachable": true,
 "queue_depth": 0, "max_pages": 5, "auth_enabled": true}
```

---

## Response

Field names match the MyCents backend's `POST /api/v1/receipts/categorize` body, so you can
forward the user-confirmed draft without renaming anything.

```json
{
  "ok": true,
  "shopName": "CP ALL, 7-Eleven",
  "receiptDate": "2026-08-28",
  "currency": "THB",
  "totalAmount": "39.00",
  "taxAmount": null,
  "basketDiscount": null,
  "items": [
    { "name": "H UHT นมยูเอชิ ด.16", "price": "13.00" },
    { "name": "ชีสโรลไส้กรอก",        "price": "26.00" }
  ],
  "ocrTexts": ["..."],
  "reconciles": true,
  "reconcileStatus": "ok",
  "engine": { "pages": 1, "ocr_seconds": 3.3, "model_seconds": 3.83,
              "prompt_tokens": 798, "token_budget": 1280, "total_seconds": 7.15 }
}
```

| Field | Notes |
|---|---|
| `shopName` | OCR errors already corrected. Thai and English both appear. |
| `receiptDate` | `YYYY-MM-DD`, or **`null`**. Thai receipts print the Buddhist year (2569); it is already converted to 2026 for you. |
| `currency` | Always `"THB"` today. |
| `totalAmount` | The printed total. |
| `taxAmount` | VAT, lifted out of `items[]` so it isn't categorized as a purchase. `null` when the receipt has none. |
| `basketDiscount` | A basket-wide discount when the receipt had one, else `null`. It has **already been spread across the item prices** — show it as information, don't subtract it again. |
| `items[]` | `name` + `price`. Prices are **line totals**, not unit prices — a "2 × 39.00" line arrives once at `78.00`. |
| `ocrTexts` | Raw OCR per page. Diagnostic — ignore it in the UI. |
| `reconciles` / `reconcileStatus` | Do the numbers add up? See below. |
| `engine` | Timings and token counts. Diagnostic. |

All money values are **strings** formatted `"NN.DD"` — no currency symbol, no thousands
separator, no sign. Parse them yourself if you need numbers.

### Two fields worth acting on

**`receiptDate: null`** — no date was found. Default to today and let the user correct it.
The parser declines rather than guesses, because a wrong date is silent: read a Buddhist year
literally and every receipt lands 543 years in the future, sorting to the top of the history
screen forever.

**`reconciles: false`** — `Σ items + tax − discount` differs from the printed total by more
than 3%. Usually a dropped or misread line. Draw the user's attention to the totals on the
confirm screen rather than hiding it.

---

## Trusting the result

This is a 2B-parameter model running locally, not a frontier API. On held-out real receipts
it gets every item and the total exactly right **67.6%** of the time.

**Design the screen so the extraction is editable, not presented as a finished record.**

| `reconcileStatus` | `reconciles` | Meaning |
|---|---|---|
| `ok` | `true` | Items sum to the total within 3%. Show normally. |
| `vat_gap` | `true` | Gap matches the VAT-on-top signature and tax is mentioned. Fine. |
| `overcount` | `false` | Items sum to *more* than the total. Duplicated line or missed discount — flag for review. |
| `unrecoverable_gap` | `false` | Doesn't add up and isn't tax. Probably a dropped item — flag for review. |
| `unparseable` | `false` | Output wasn't usable JSON. Offer a retake. |

---

## Timing

Measured end to end on real receipts through this exact service:

- **Typical photo request: 4 – 8s** (OCR ~2.3–3.3s + model ~1.6–3.9s)
- `/v1/extract-text` skips the OCR half: ~3s.
- Show a spinner. This is not an instant call.
- **Only one receipt is processed at a time.** A second request queues rather than failing,
  so two phones shooting together means the later one waits.
- Set your client timeout to **60 seconds**, not 10.

---

## Errors

Every failure returns JSON with a `detail` string, so you can always read the reason. CORS is
handled server-side, so a plain `fetch` from your origin works, preflight included.

| Status | Meaning | What to do |
|---|---|---|
| `400` | No image attached, an empty file, or a malformed JSON body | Check the file actually attached |
| `401` | Missing or wrong `X-API-Key` | Check the header name and the key |
| `413` | Image over 25 MB, more than 5 pages, or receipt text longer than the model accepts | Downscale, or shoot fewer/tighter frames |
| `422` | OCR couldn't read the image, or found no text in it | Ask for a retake |
| `502` | OCR service unreachable on our side | Ping us; retrying won't help |
| `503` | Model still loading, or too many requests queued | Check `/ready`; retry shortly |
| `504` | Generation took too long | Retry once, then ping us |

An unreadable receipt is **not** an HTTP error: a photo whose text the model can't turn into
JSON returns **200** with `"ok": false` and an `error` string instead of the fields. Check
`ok` before reading `shopName`.

---

## Quick check

Once you have the URL, confirm it works before touching app code:

```bash
# no key needed — is the engine actually ready?
curl https://SOMETHING.trycloudflare.com/ready

# the real thing
curl -X POST https://SOMETHING.trycloudflare.com/v1/extract \
     -H "X-API-Key: YOUR_KEY" \
     -F "files=@receipt.jpg"
```

If `/ready` returns `{"ok": true, ...}` the path from your machine to the GPU is working and
the photo call will go through.

---

## Availability

The engine runs on a workstation, not a datacenter:

- **That machine has to be awake with the services running.** Nothing responds otherwise.
- **The URL changes every time the tunnel restarts.** Don't hardcode it — read it from config
  so a new one is a one-line change.
- **`/ready` returns 503 for about a minute after startup** while the model loads into VRAM.
