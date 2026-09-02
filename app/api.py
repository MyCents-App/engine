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

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

import postprocess
from app import extraction, ocr_client
from app.config import settings
from app.date_extract import extract_receipt_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("engine")

# Separator between pages of a multi-photo receipt. A blank line only — no
# marker text, because the model was trained on continuous OCR text and an
# invented token like "--- PAGE 2 ---" is a string it has never seen.
PAGE_SEPARATOR = "\n"


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
        "max_pages": settings.max_pages,
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


def _to_response(pred: dict | None, ocr_texts: list[str], recon, timings: dict) -> dict:
    if pred is None:
        return {
            "ok": False,
            "error": "The model did not return valid JSON for this receipt.",
            "ocrTexts": ocr_texts,
            "engine": timings,
        }

    items, tax = _split_tax(pred)
    discount = pred.get("basket-wide_discount")

    return {
        "ok": True,
        # Field names match the MyCents backend's /receipts/categorize body,
        # so the app forwards the user-confirmed draft without renaming.
        "shopName": pred.get("shop_name") or None,
        "receiptDate": (
            extract_receipt_date("\n".join(ocr_texts)).isoformat()
            if ocr_texts and extract_receipt_date("\n".join(ocr_texts)) else None
        ),
        "currency": "THB",
        "totalAmount": pred.get("total_price"),
        "taxAmount": str(tax) if tax is not None else None,
        "basketDiscount": discount,
        "items": [{"name": i.get("name"), "price": i.get("price")} for i in items],
        "ocrTexts": ocr_texts,
        # Business-validity signal, computed on the model's own output before
        # the discount was spread (engine/task.md §5). False means the numbers
        # do not add up and the user should look closely.
        "reconciles": bool(recon.passes) if recon else False,
        "reconcileStatus": recon.status if recon else "unparseable",
        "engine": timings,
    }


async def _run(ocr_texts: list[str], timings: dict) -> dict:
    combined = PAGE_SEPARATOR.join(t for t in ocr_texts if t.strip())
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
    if settings.apply_discount and pred is not None:
        pred = postprocess.apply_basket_discount(pred)

    return _to_response(pred, ocr_texts, recon, timings)


@app.post("/v1/extract", tags=["extract"], dependencies=[Depends(require_api_key)])
async def extract_images(files: list[UploadFile] = File(...)) -> dict:
    """Photos of ONE receipt -> a structured draft.

    Send several files for a long receipt, **in capture order**. Each page is
    OCR'd separately and the text concatenated in that order before a single
    model call, so an item whose name is cut off at the bottom of page 1 and
    whose price appears at the top of page 2 is rejoined correctly. Order is
    therefore load-bearing — never let the client shuffle the images.
    """
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No images uploaded.")
    if len(files) > settings.max_pages:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{len(files)} images; the limit is {settings.max_pages} per receipt.",
        )

    started = time.time()
    ocr_texts: list[str] = []
    ocr_seconds = 0.0

    for index, upload in enumerate(files):
        payload = await upload.read()
        if not payload:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"Image {index + 1} is empty.")
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Image {index + 1} is {len(payload)} bytes; the limit is "
                f"{settings.max_upload_bytes}.",
            )
        try:
            result = await ocr_client.run(payload, upload.content_type or "image/jpeg")
        except ocr_client.OcrUnavailable as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))
        except ocr_client.OcrFailed as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))
        ocr_texts.append(result.get("text", ""))
        ocr_seconds += float(result.get("ocr_seconds") or 0)

    timings = {
        "pages": len(files),
        "ocr_seconds": round(ocr_seconds, 2),
        "checkpoint": settings.checkpoint,
    }
    body = await _run(ocr_texts, timings)
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
    pages = payload.get("pages")
    ocr_texts = pages if isinstance(pages, list) and pages else [text]

    started = time.time()
    body = await _run(ocr_texts, {"pages": len(ocr_texts), "ocr_seconds": 0.0,
                                  "checkpoint": settings.checkpoint})
    body["engine"]["total_seconds"] = round(time.time() - started, 2)
    return body
