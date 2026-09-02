"""MyCents extraction engine — HTTP API.

    receipt photo(s) --> Surya OCR (sidecar) --> fine-tuned Qwen3.5-2B
                     --> deterministic post-processing --> structured JSON

Machine-facing only: no HTML, no browser page. The Flutter app posts photos
here and gets back a draft it shows the user for confirmation; the confirmed
draft then goes to the MyCents backend for categorization.

The response is deliberately shaped to match that backend's
POST /api/v1/receipts/categorize body, so the app can pass it through with
the user's edits applied and no field renaming in between.

Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from decimal import Decimal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

import postprocess
from app import extraction, ocr_client
from app.config import settings
from app.date_extract import extract_receipt_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("engine")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.auth_enabled:
        logger.warning("=" * 70)
        logger.warning("ENGINE_API_KEY is empty — this service is UNAUTHENTICATED.")
        logger.warning("Never expose it through a tunnel in this state.")
        logger.warning("=" * 70)
    try:
        extraction.load_model()
    except Exception:  # noqa: BLE001 -- report and stay up so /health explains
        logger.exception("model failed to load; /ready will report not-ready")
    if settings.warmup and extraction.is_loaded():
        started = time.time()
        try:
            await extraction.extract("MINI MART\nwater  10.00\nTotal  10.00")
            logger.info("warm-up generation done in %.1fs", time.time() - started)
        except Exception:  # noqa: BLE001 -- a slow first request beats no service
            logger.warning("warm-up failed; the first request will just be slower",
                           exc_info=True)
    if not await ocr_client.healthy():
        logger.warning("OCR sidecar not reachable at %s", settings.ocr_url)
    yield


app = FastAPI(
    title="MyCents Extraction Engine",
    version="1.0.0",
    description=__doc__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

# Without this a browser-based client cannot call the engine at all: the
# preflight OPTIONS goes unanswered, so the real POST is never sent. Flutter's
# native HTTP client is unaffected either way, but a web frontend is not.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allow_origin],
    allow_methods=["GET", "POST", "OPTIONS"],
    # X-API-Key must be listed explicitly: it is not a CORS-safelisted header.
    allow_headers=["Content-Type", "X-API-Key"],
    max_age=86400,
)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Shared-secret auth.

    The service is reachable over a tunnel from the public internet and it
    fronts a GPU. Without a key, anyone who finds the URL can run generation
    on it. Compared with `==` via secrets.compare_digest to avoid leaking the
    key's length through response timing.
    """
    if not settings.auth_enabled:
        return
    import secrets
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key.",
        )


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

@app.get("/health", tags=["health"])
async def health() -> dict:
    """Liveness. No auth, no I/O — a container probe must not need a secret."""
    return {"ok": True}


@app.get("/ready", tags=["health"])
async def ready() -> JSONResponse:
    """Readiness: the model is loaded AND the OCR sidecar answers."""
    model_ok = extraction.is_loaded()
    ocr_ok = await ocr_client.healthy()
    body = {
        "ok": model_ok and ocr_ok,
        "model_loaded": model_ok,
        "ocr_reachable": ocr_ok,
        "checkpoint": settings.checkpoint,
        "queue_depth": extraction.queue_depth(),
        "auth_enabled": settings.auth_enabled,
    }
    code = 200 if body["ok"] else 503
    return JSONResponse(body, status_code=code)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _split_tax(pred: dict) -> tuple[list[dict], Decimal | None]:
    """Lift the model's vat/tax row out of items[].

    task.md rule 3 has the model add a "vat"/"tax" line item when tax sits on
    top of the listed prices. It is not a purchase: leaving it in items[]
    means it gets categorized as one, and every per-category spending total
    is then wrong by the tax. The MyCents backend stores it on
    receipts.tax_amount instead.

    Done AFTER reconciliation, so postprocess still sees the output shape it
    was written against.
    """
    items, tax = [], Decimal("0")
    for item in pred.get("items", []):
        name = str(item.get("name", "")).strip().lower()
        if name in {"vat", "tax"} or name.startswith(("vat ", "tax ", "ภาษี")):
            value = postprocess.parse_price_lenient(item.get("price"))
            if value is not None:
                tax += Decimal(str(value))
                continue
        items.append(item)
    return items, (tax if tax > 0 else None)


def _money(value) -> str | None:
    """Format a monetary value as NN.DD — the contract every price field uses.

    Without this a Decimal of 7 serializes as "7.0" while the item prices
    beside it read "7.00", and a client parsing one format trips on the other.
    """
    if value is None:
        return None
    parsed = postprocess.parse_price_lenient(str(value))
    return None if parsed is None else f"{parsed:.2f}"


def _to_response(pred: dict | None, ocr_texts: list[str], recon, timings: dict,
                 discount=None) -> dict:
    if pred is None:
        return {
            "ok": False,
            "error": "The model did not return valid JSON for this receipt.",
            "ocrTexts": ocr_texts,
            "engine": timings,
        }

    items, tax = _split_tax(pred)
    # `discount` is a parameter, NOT read off `pred`: apply_basket_discount has
    # already spread it across the item prices and deleted the key, so reading
    # it here reported null on every receipt that actually had one.
    joined = "\n".join(ocr_texts)
    receipt_date = extract_receipt_date(joined) if joined.strip() else None

    return {
        "ok": True,
        # Field names match the MyCents backend's /receipts/categorize body,
        # so the app forwards the user-confirmed draft without renaming.
        "shopName": pred.get("shop_name") or None,
        "receiptDate": receipt_date.isoformat() if receipt_date else None,
        "currency": "THB",
        "totalAmount": pred.get("total_price"),
        "taxAmount": _money(tax),
        "basketDiscount": _money(discount),
        "items": [{"name": i.get("name"), "price": i.get("price")} for i in items],
        "ocrTexts": ocr_texts,
        # Business-validity signal, computed on the model's own output before
        # the discount was spread (engine/task.md §5). False means the numbers
        # do not add up and the user should look closely.
        "reconciles": bool(recon.passes) if recon else False,
        "reconcileStatus": recon.status if recon else "unparseable",
        "engine": timings,
    }


async def _run(combined: str, timings: dict) -> dict:
    # ocrTexts stays a one-element list in the response: the app was written
    # against that shape and there is nothing to gain from breaking it.
    ocr_texts = [combined]
    if not combined.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No text was found in the uploaded image(s).",
        )

    # Reject over-long input rather than letting the tokenizer truncate it.
    # Truncation drops the END of the receipt — the totals and the last
    # items — and the model still returns confident, well-formed JSON. That
    # is the worst kind of failure: silent and plausible.
    tokens = extraction.count_prompt_tokens(combined)
    budget = extraction.token_budget()
    timings["prompt_tokens"] = tokens
    timings["token_budget"] = budget
    if tokens > budget:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Receipt text is {tokens} tokens; the model can accept "
                f"{budget}. Photograph the receipt in fewer, tighter frames, "
                "or raise ENGINE_MAX_SEQ_LENGTH if the GPU has room."
            ),
        )

    try:
        pred, raw, model_s = await extraction.extract(combined)
    except extraction.Overloaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except extraction.ModelNotLoaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except TimeoutError:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "The model did not finish in time. The service may be overloaded.",
        )

    timings["model_seconds"] = round(model_s, 2)

    # Reconcile BEFORE spreading the discount: afterwards items sum to the
    # total by construction, so the check would always pass and tell us
    # nothing.
    recon = postprocess.reconcile(pred, combined) if pred else None
    discount = pred.get("basket-wide_discount") if pred else None
    if settings.apply_discount and pred is not None:
        pred = postprocess.apply_basket_discount(pred)

    return _to_response(pred, ocr_texts, recon, timings, discount)


async def _read_uploads(request: Request) -> list[tuple[bytes, str]]:
    """The photo, however the client chose to send it.

    Multipart under ANY field name, or a raw image as the request body. Both
    were promised by the published contract and a frontend that already posts
    `file`, `image`, or raw bytes should not have to be rewritten to say
    `files`.

    Returns every file found rather than just the first, so the caller can
    tell "one photo" from "several" and reject the latter explicitly. Silently
    taking one of several would drop a receipt the user meant to send.

    multi_items(), NOT values(): starlette's FormData is a multidict whose
    values() collapses repeated keys, which would hide a two-file upload.
    UploadFile here is starlette's, not fastapi's subclass: form() builds the
    base class, so checking the subclass would match nothing.
    """
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()

    if ctype == "multipart/form-data":
        form = await request.form()
        return [(await v.read(), v.content_type or "image/jpeg")
                for _, v in form.multi_items() if isinstance(v, UploadFile)]

    body = await request.body()
    return [(body, ctype or "image/jpeg")] if body else []


@app.post("/v1/extract", tags=["extract"], dependencies=[Depends(require_api_key)])
async def extract_image(request: Request) -> dict:
    """One photo of one receipt -> a structured draft.

    The app captures a single frame, so this takes exactly one image. More
    than one is rejected rather than silently processing the first: a client
    attaching two photos has misunderstood the contract, and quietly dropping
    one loses a receipt the user believed they had sent.

    Accepts multipart/form-data under any field name, or a raw image as the
    request body. The request is read manually rather than declared as
    `file: UploadFile`, which would have bound the client to one exact field
    name; the cost is that /docs cannot render an upload widget.
    """
    uploads = await _read_uploads(request)
    if not uploads:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No image uploaded.")
    if len(uploads) > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{len(uploads)} images sent; this endpoint takes exactly one "
            "photo per receipt.",
        )

    started = time.time()
    payload, content_type = uploads[0]
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The image is empty.")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"The image is {len(payload)} bytes; the limit is "
            f"{settings.max_upload_bytes}.",
        )
    try:
        result = await ocr_client.run(payload, content_type)
    except ocr_client.OcrUnavailable as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))
    except ocr_client.OcrFailed as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))

    timings = {
        "ocr_seconds": round(float(result.get("ocr_seconds") or 0), 2),
        "checkpoint": settings.checkpoint,
    }
    body = await _run(result.get("text", ""), timings)
    body["engine"]["total_seconds"] = round(time.time() - started, 2)
    return body


@app.post("/v1/extract-text", tags=["extract"], dependencies=[Depends(require_api_key)])
async def extract_text(payload: dict) -> dict:
    """Same model path, but the caller supplies the OCR text.

    Lets an integration be tested without paying ~20s of OCR per iteration,
    and lets a saved text.txt be replayed against the model. Byte-identical
    prompting and post-processing to /v1/extract, so a result here is exactly
    what the photo path would produce.
    """
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            'Body must be {"text": "..."}.')
    started = time.time()
    body = await _run(text, {"ocr_seconds": 0.0,
                             "checkpoint": settings.checkpoint})
    body["engine"]["total_seconds"] = round(time.time() - started, 2)
    return body
