# Receipt Extraction API

Send a photo of a receipt — or two or three photos of a long one — and get back structured
JSON: merchant, date, line items with prices, tax and the total. Runs on a GPU workstation,
exposed over an ngrok tunnel.

---

## If you already have this working

**Nothing you have breaks.** Multi-photo was added without removing anything: a one-photo
request returns exactly the fields it always did, plus `engine.pages`. `ocrTexts` was already
an array and still has one entry. `/ready` gained `max_pages`. Both upload styles and
`/v1/extract-text` are unchanged.

Four things to know:

- **Every item now carries `category` and `subcategory`.** Two new keys per item, nothing
  removed or renamed — a client that ignores unknown fields is unaffected. The category is
  meant as a **fallback** for the backend's own categorization, not a replacement for it; see
  [Categories](#categories-a-fallback-not-the-answer). `subcategory` is always `null` for now.
- **The tunnel moved from Cloudflare to ngrok. Browser code must add one header,
  `ngrok-skip-browser-warning: 1`, to every request** — without it ngrok answers with an
  HTML warning page instead of the API's JSON. See [The things you need](#the-things-you-need).
  curl and server-side code are unaffected.
- **Sending 2+ images used to be a `400`. It is now a valid multi-page request.** Only matters
  if you branch on that error.
- **To use multi-photo you have to opt in** — let the user take several shots and append them
  all to the same FormData. See [Long receipts](#long-receipts-several-photos-one-receipt).

---

## The things you need

**1. Base URL.** You'll get the current one before each session. It may change when the
server restarts, so read it from config rather than hardcoding it.

```js
const BASE = "https://SOMETHING.ngrok-free.app";
```

**2. API key.** Sent as an `X-API-Key` header on every `/v1/*` call. Sent to you separately —
it is deliberately not written down in this file.

**The MyCents app does not hold this key.** Since 22 Sep 2026 it calls the
backend's `/api/v1/engine/*`, which adds the key and forwards the request
here; the key lives only in `server/.env`. A key compiled into a phone app
is a key anyone with the APK has. Everything below still describes what the
app receives, because the backend forwards bodies and status codes
verbatim.

```js
const KEY = "...";   // paste the key you were given
```

The service is on the public internet and fronts a GPU, so `/v1/*` is closed. Without the
header you get **401**. `/health` and `/ready` are open and need no key.

**3. The ngrok header, from a browser.** The free ngrok plan intercepts browser requests with
an HTML "You are about to visit…" page. Sending `ngrok-skip-browser-warning` (any value)
skips it. Put it on **every** request, `/ready` included, so all calls share one set of
headers:

```js
const HEADERS = { "X-API-Key": KEY, "ngrok-skip-browser-warning": "1" };
```

If a call returns HTML instead of JSON, this header is missing.

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
  headers: HEADERS,                    // do NOT set Content-Type yourself
  body: fd,
});
const data = await res.json();
```

### Option B: raw bytes

```js
const res = await fetch(BASE + "/v1/extract", {
  method: "POST",
  headers: { ...HEADERS, "Content-Type": "image/jpeg" },
  body: photoFile,                     // Blob / File / ArrayBuffer
});
```

JPEG, PNG, WEBP and iPhone HEIC all work. Max **25 MB** per image.

### Long receipts: several photos, one receipt

A long receipt doesn't fit in one frame. Append more than one file — up to **5** — and they
are treated as pages of a single receipt.

```js
fd.append("files", page1);
fd.append("files", page2);
fd.append("files", page3);             // capture order is load-bearing — never shuffle
```

**Tell the user to overlap the shots.** Ending photo 2 a line or two above where photo 1
finished is the only way to guarantee nothing falls in the gap between frames, and the
repeated lines cost you nothing: each page is OCR'd separately, the pages are stitched into
one text with the overlap removed, and the model is called **once** on the result. An item
whose name sits at the bottom of one frame and whose price sits at the top of the next is
rejoined correctly.

Two rules for the client:

1. **Send them in capture order, top of the receipt first.** Overlap is detected by looking
   for the start of each page at the end of the ones before it. Shuffled photos aren't
   de-duplicated, and the items come back out of order.
2. **One receipt per request.** Two different receipts posted as pages produces nonsense —
   and if they're from the same shop, they may even be merged into one.

The stitch is reported back in `engine.stitch` so you can see what happened (see
[Response](#response)).

---

## `POST /v1/extract-text` — same model, no OCR

You supply the OCR text instead of a photo. Skips ~2-3s of OCR and is fully deterministic, so
it's the better thing to develop against before wiring up the camera.

```js
fetch(BASE + "/v1/extract-text", {
  method: "POST",
  headers: { ...HEADERS, "Content-Type": "application/json" },
  body: JSON.stringify({ text: "7-ELEVEN\ncoke 20.00\nTotal 20.00" }),
});
```

For a multi-photo receipt send `pages` instead — this is the only way to exercise the
stitching without a camera:

```js
body: JSON.stringify({ pages: ["...page 1 text...", "...page 2 text..."] })
```

Byte-identical prompting, stitching and post-processing to `/v1/extract`, so a result here is
exactly what the photo path would produce from that text.

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
  "shopNameEn": null,
  "receiptDate": "2026-08-28",
  "currency": "THB",
  "totalAmount": "39.00",
  "taxAmount": null,
  "basketDiscount": null,
  "items": [
    { "name": "H UHT นมยูเอชิ ด.16", "nameEn": null, "price": "13.00",
      "category": "Groceries", "subcategory": null },
    { "name": "ชีสโรลไส้กรอก",        "nameEn": null, "price": "26.00",
      "category": "Food & Dining", "subcategory": null }
  ],
  "ocrTexts": ["..."],
  "reconciles": true,
  "reconcileStatus": "ok",
  "engine": { "pages": 1, "ocr_seconds": 2.6, "model_seconds": 3.2,
              "prompt_tokens": 798, "token_budget": 3072, "total_seconds": 5.8 }
}
```

A **multi-photo** receipt adds two things and changes nothing else:

```json
{
  "ocrTexts": ["...page 1 raw...", "...page 2 raw..."],
  "stitchedText": "...the two pages joined, overlap removed...",
  "engine": {
    "pages": 2,
    "stitch": { "pages": 2, "duplicate_lines_removed": 4,
                "seams": [{ "page": 2, "overlap_lines": 4, "score": 0.94 }] }
  }
}
```

| Field | Notes |
|---|---|
| `shopName` | OCR errors already corrected, **in the language the receipt printed** — Thai and English both appear. |
| `shopNameEn` | Always `null` from the engine. See [Translation](#translation-is-not-the-engines-job). |
| `receiptDate` | `YYYY-MM-DD`, or **`null`**. Thai receipts print the Buddhist year (2569); it is already converted to 2026 for you. |
| `currency` | Always `"THB"` today. |
| `totalAmount` | The printed total. |
| `taxAmount` | VAT, lifted out of `items[]` so it isn't categorized as a purchase. `null` when the receipt has none. |
| `basketDiscount` | A basket-wide discount when the receipt had one, else `null`. It has **already been spread across the item prices** — show it as information, don't subtract it again. |
| `items[]` | `name` + `nameEn` + `price` + `category` + `subcategory`. `name` is the printed text; `nameEn` is always `null` from the engine. Prices are **line totals**, not unit prices — a "2 × 39.00" line arrives once at `78.00`. |
| `items[].category` | One of the eight MyCents categories, spelled exactly, or `null` when the model could not tell what the item is. A **fallback** — see [Categories](#categories-a-fallback-not-the-answer). |
| `items[].subcategory` | Always `null` today. Will carry the model's subcategory once it is accurate enough; see below. |
| `ocrTexts` | Raw OCR text, one entry **per photo**, in the order you sent them. Diagnostic — ignore it in the UI. |
| `stitchedText` | Only on multi-photo requests: the pages joined with the overlap removed — what the model actually read. Diagnostic. |
| `reconciles` / `reconcileStatus` | Do the numbers add up? See below. |
| `engine` | Timings, token counts and (multi-photo only) `stitch`. Diagnostic. |

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

### Categories: a fallback, not the answer

The same model call that reads the receipt also assigns each item one of the eight MyCents
categories. It is there for the **backend's** categorization pipeline, which keeps running
first:

1. catalog exact → 2. catalog fuzzy → 3. brand / merchant → 4. keyword
5. **the engine's `category`** — only for items stages 1-4 left unresolved
6. user review — when `category` is `null` too

Those lookups are deterministic and know this user's history; the model is a 2B network that
has seen a few thousand receipts. So the model's answer is used where they have none, never
over theirs. Two rules follow:

- **`null` is an answer.** It means the model declined — an item it could not identify, or a
  category that failed validation (the engine nulls anything outside the eight, never
  "repairs" it). Route it to user review exactly as a stage-5 decline is today.
- **`subcategory` is `null` on purpose.** A subcategory the model offers is right about 3 times
  in 4 against a 90% target, and the backend's own subcategory classifier only runs on nulls —
  a wrong value here would never be corrected, while a null gets the classifier. It switches on
  (`ENGINE_EMIT_SUBCATEGORY=1`) when a checkpoint clears 90%, with no change to this contract.

The tax row is never categorized: it is lifted out of `items[]` into `taxAmount`.

### Translation is not the engine's job

Names come back exactly as printed, which for most receipts means Thai. The engine **never
translates them** and never will: the model is fine-tuned to reproduce the printed text, and
the backend's categorization matches on that text — a category that depended on a
translation would change with the translator.

English is a display layer added *after* extraction, and the response leaves a slot for it:

1. **The app** translates `name` / `shopName` on the phone (Google ML Kit, offline, free) and
   writes the result into `nameEn` / `shopNameEn` before forwarding the confirmed draft.
2. **The backend** (`POST /api/v1/receipts/categorize`) overrides `nameEn` with the product
   catalog's English name whenever the item matched a known product, and caches the app's
   translation for items it did not know — so each translation is made once and then served
   to everyone.

So: translate on the phone if you want English on the confirm screen, put it in the slot,
forward. Never send an English name in `name` — that is what gets categorized.

---

## Trusting the result

This is a 2B-parameter model running locally, not a frontier API. On 73 real receipts it never
trained on, it gets every item and the total exactly right **67.1%** of the time, and gives the
right category for **91.6%** of the items it reads. Both figures are provisional until those 73
labels have been hand-checked (`train/reports/joint_eval.md`). The previous, extraction-only
model scored 62.9% on the ones among them it had not been trained on, and categorized nothing.

**Design the screen so the extraction is editable, not presented as a finished record.**

| `reconcileStatus` | `reconciles` | Meaning |
|---|---|---|
| `ok` | `true` | Items sum to the total within 3%. Show normally. |
| `vat_gap` | `true` | Gap matches the VAT-on-top signature and tax is mentioned. Fine. |
| `overcount` | `false` | Items sum to *more* than the total. Duplicated line or missed discount — flag for review. |
| `unrecoverable_gap` | `false` | Doesn't add up and isn't tax. Probably a dropped item — flag for review. |
| `unparseable` | `false` | Output wasn't usable JSON. Offer a retake. |

### On a multi-photo receipt

Overlap removal is measured at **99.8%** exact reconstruction, with **no line ever lost**
across 480 trials — the thresholds are deliberately set so that a *missed* overlap is possible
and a *falsely removed* one is not. See `train/reports/stitch_overlap.md`.

The residual failure is therefore always the safe direction: a repeated line survives into the
prompt, the model lists that item twice, and you get `reconcileStatus: "overcount"`. Read that
status on a multi-photo request as "an item may be listed twice" and let the user delete the
line. Do not de-duplicate `items[]` yourself — a receipt can legitimately print the same item
on two lines, and you cannot tell the two cases apart from the response.

---

## Timing

Measured on real receipts through this exact service. The model's time now grows with the
number of items — it writes a category for each one — at roughly **1s + 1.2s per item**:

- **Typical photo request (1–5 items): 5 – 10s** (OCR ~2.5–3.5s + model ~1.6–7s)
- **A long receipt is slower:** ~15s at 8 items, ~30s at 20+ (a full Makro basket).
- **Add ~2.5–3.5s per extra photo.** OCR runs once per image; the model still runs once for
  the whole receipt, so a three-photo receipt is roughly OCR×3 + one generation.
- `/v1/extract-text` skips the OCR half: median ~3s on a typical receipt.
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
| `413` | Image over 25 MB, more than 5 photos, or receipt text longer than the model accepts | Downscale, or shoot fewer/tighter frames |
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
curl https://SOMETHING.ngrok-free.app/ready

# the real thing
curl -X POST https://SOMETHING.ngrok-free.app/v1/extract \
     -H "X-API-Key: YOUR_KEY" \
     -F "files=@receipt.jpg"

# a long receipt, IN CAPTURE ORDER
curl -X POST https://SOMETHING.ngrok-free.app/v1/extract \
     -H "X-API-Key: YOUR_KEY" \
     -F "files=@page1.jpg" -F "files=@page2.jpg"
```

If `/ready` returns `{"ok": true, ...}` the path from your machine to the GPU is working and
the photo call will go through.

---

## Availability

The engine runs on a workstation, not a datacenter:

- **That machine has to be awake with the services running.** Nothing responds otherwise.
- **The URL may change when the tunnel restarts.** Don't hardcode it — read it from config
  so a new one is a one-line change.
- **`/ready` returns 503 for about a minute after startup** while the model loads into VRAM.
