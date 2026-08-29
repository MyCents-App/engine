"""Phase 2: deterministic code-layer pieces from task.md Sections 4-5.

This module implements the parts of the pipeline that are explicitly the CODE layer's job, not
the model's job:
  - Robust extraction of a JSON object out of a raw model completion (strips <think> blocks from
    reasoning models, markdown fences, and stray prose).
  - Schema-shape validation of a candidate prediction (lenient about key order/extra whitespace,
    strict about the actual contract fields).
  - reconcile(): the Section 5 gap-repair decision logic (3% tolerance, ~6.5% VAT-gap signature,
    tax-token sniffing), run against the model's OWN output as a business-validity signal.

It is imported by eval_metrics.py (Phase 2), baseline_eval.py (Phase 3), and the fine-tuned-model
eval (Phase 5) so every phase applies identical logic.

Deliberately NOT implemented here (out of scope per the plan): the live retry-loop orchestration
that would re-invoke the model on a failed reconciliation. reconcile() gives the *decision* a
retry-loop would act on; wiring it into a serving loop is future work.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Constants (mirrors task.md Section 5)
# --------------------------------------------------------------------------------------

RECONCILIATION_TOLERANCE = 0.03  # 3%
VAT_SIGNATURE_LOW = 0.055  # ~6.5% signature, tolerance band 5.5-7.5% per task.md
VAT_SIGNATURE_HIGH = 0.075

# task.md's own token list. Note "tax" alone is a weak signal in practice -- almost every Thai
# receipt prints a "TAX ID#" boilerplate line regardless of whether VAT is added on top, so this
# check will very often be true. The real discriminating power is the residual-percentage band
# below; the token check is a secondary corroboration, implemented faithfully to the spec rather
# than tightened, since narrowing it risks disagreeing with the spec's own examples.
TAX_TOKEN_RE = re.compile(r"vat|tax|ภาษี", re.IGNORECASE)  # ภาษี

PRICE_RE = re.compile(r"^\d+\.\d{2}$")  # strict contract format: NN.DD

CONTRACT_KEYS = {"shop_name", "items", "total_price", "basket-wide_discount"}
REQUIRED_KEYS = {"shop_name", "items", "total_price"}


# --------------------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _find_balanced_object(text: str) -> str | None:
    """Scan for the first top-level `{...}` object, honoring string literals so braces inside
    item names (rare, but possible after OCR "correction") don't break the scan."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # never closed


def extract_json(raw_completion: str) -> tuple[dict | None, str | None]:
    """Best-effort extraction of a JSON object from a raw model completion.

    Returns (parsed_dict, None) on success, or (None, reason) on failure. Handles:
      - a leading <think>...</think> block from reasoning models (e.g. Qwen3/Qwen3.5)
      - markdown code fences around the JSON
      - stray prose before/after the JSON object
      - a trailing comma before a closing bracket (common small model slip)
    """
    if not isinstance(raw_completion, str) or not raw_completion.strip():
        return None, "empty completion"

    text = _THINK_RE.sub("", raw_completion)

    # Prefer content inside a fenced code block if present -- untuned models often wrap JSON in
    # ```json ... ``` even when told not to.
    fence_match = _FENCE_RE.search(text)
    search_text = fence_match.group(1) if fence_match else text

    candidate = _find_balanced_object(search_text)
    if candidate is None and fence_match is not None:
        # fall back to scanning the whole text in case the fence regex mis-captured
        candidate = _find_balanced_object(text)
    if candidate is None:
        return None, "no JSON object found in completion"

    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired), None
            except json.JSONDecodeError:
                pass
        return None, f"json decode error: {e}"


# --------------------------------------------------------------------------------------
# Schema validation (lenient on key order, strict on contract shape)
# --------------------------------------------------------------------------------------

def is_schema_valid(obj) -> tuple[bool, list[str]]:
    """Checks a prediction against the task.md output contract. Unlike data_prep's gold-data
    validator, this does not require exact key order (a model isn't expected to control JSON
    key emission order precisely) but is strict about which keys are allowed and each field's
    shape/format."""
    reasons: list[str] = []
    if not isinstance(obj, dict):
        return False, ["not a JSON object"]

    extra_keys = set(obj.keys()) - CONTRACT_KEYS
    if extra_keys:
        reasons.append(f"unexpected keys: {sorted(extra_keys)}")

    missing_keys = REQUIRED_KEYS - set(obj.keys())
    if missing_keys:
        reasons.append(f"missing required keys: {sorted(missing_keys)}")

    shop_name = obj.get("shop_name")
    if "shop_name" in obj and (not isinstance(shop_name, str) or not shop_name.strip()):
        reasons.append("shop_name is not a non-empty string")

    items = obj.get("items")
    if "items" in obj:
        if not isinstance(items, list) or not items:
            reasons.append("items is not a non-empty list")
        else:
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    reasons.append(f"items[{i}] is not an object")
                    continue
                if not isinstance(item.get("name"), str) or not item["name"].strip():
                    reasons.append(f"items[{i}].name is not a non-empty string")
                if not isinstance(item.get("price"), str) or not PRICE_RE.match(item["price"]):
                    reasons.append(f"items[{i}].price is not 'NN.DD': {item.get('price')!r}")

    if "total_price" in obj:
        tp = obj.get("total_price")
        if not isinstance(tp, str) or not PRICE_RE.match(tp):
            reasons.append(f"total_price is not 'NN.DD': {tp!r}")

    if "basket-wide_discount" in obj:
        d = obj.get("basket-wide_discount")
        if not isinstance(d, str) or not PRICE_RE.match(d):
            reasons.append(f"basket-wide_discount is not 'NN.DD': {d!r}")

    return len(reasons) == 0, reasons


# --------------------------------------------------------------------------------------
# Price parsing
# --------------------------------------------------------------------------------------

def parse_price_strict(value) -> float | None:
    """Only accepts the exact 'NN.DD' contract format."""
    if not isinstance(value, str) or not PRICE_RE.match(value):
        return None
    return float(value)


_LENIENT_STRIP_RE = re.compile(r"[^\d.,\-]")


def parse_price_lenient(value) -> float | None:
    """Best-effort numeric coercion for computing sums during reconciliation, even when a
    (likely untuned/zero-shot) model doesn't perfectly follow the NN.DD contract format --
    otherwise reconciliation_pass_rate would be near-zero for every baseline and hide real
    differences in whether the underlying numbers are close to right."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = _LENIENT_STRIP_RE.sub("", value.strip())
    if not s:
        return None

    if "," in s and "." in s:
        s = s.replace(",", "")  # comma = thousands separator
    elif "," in s:
        # ambiguous: Thai OCR sometimes prints a decimal comma (e.g. "43,50"). Treat a single
        # comma followed by exactly 2 digits as a decimal point; otherwise thousands separator.
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return None


def normalize_prediction(pred: dict) -> dict:
    """Coerce a model prediction's price strings into the NN.DD contract, in place of nothing.

    This is a CODE-LAYER job per task.md Section 4, not something to demand of the model. The
    model's contribution is the number; rendering it as exactly two decimals with no currency
    symbol and no thousands separator is deterministic string work, and failing a receipt whose
    every number is correct because the model wrote "169" instead of "169.00" would be measuring
    the wrong thing.

    Measured on the 68 held-out real receipts: recovers 2 (+2.9pp money-exact) that are
    numerically perfect and fail only on formatting.

    Deliberately conservative -- it only ever REFORMATS a value it can parse unambiguously. It
    never invents a missing price, never rounds a value to make a receipt reconcile, and returns
    the value untouched when it cannot parse it, so a genuinely wrong number stays wrong.
    """
    if not isinstance(pred, dict):
        return pred

    def fix(v):
        if isinstance(v, str) and PRICE_RE.match(v):
            return v  # already conformant, leave the exact string alone
        n = parse_price_lenient(v)
        return f"{n:.2f}" if n is not None and n >= 0 else v

    out = dict(pred)
    for key in ("total_price", "basket-wide_discount"):
        if key in out:
            out[key] = fix(out[key])
    if isinstance(out.get("items"), list):
        out["items"] = [
            {**it, "price": fix(it.get("price"))} if isinstance(it, dict) and "price" in it else it
            for it in out["items"]
        ]
    return out


def apply_basket_discount(pred: dict) -> dict:
    """Spread `basket-wide_discount` across items[] pro-rata and drop the field.

    This is the LAST code-layer step, producing what the downstream categorizer consumes:
    every item priced at what the customer actually paid for it, and no discount field left
    to interpret. The model's job ends at reporting the discount separately (task.md Rule 4);
    deciding how it lands on each line is deterministic arithmetic and belongs here.

    Pro-rata, not an equal split: each item is scaled by (subtotal - discount) / subtotal, so
    a cheap line takes a proportionally small cut. An equal split subtracts discount/n from
    every item and drives anything cheaper than that negative -- with a 2.00 "Shopping bag"
    line (which this dataset really contains) and a 40.00 basket discount over 4 items, the
    bag prices at -8.00.

    Rounding is settled by largest-remainder so the result sums to the total EXACTLY. Naive
    per-item rounding drifts a cent or two, which would make the reconcile check fail on
    output this function itself produced.

    Returns a new dict; the input is not modified. A prediction with no discount field, or
    one whose numbers don't parse, is returned unchanged.
    """
    if not isinstance(pred, dict) or "basket-wide_discount" not in pred:
        return pred

    items = pred.get("items")
    if not isinstance(items, list) or not items:
        return pred

    discount = parse_price_lenient(pred.get("basket-wide_discount"))
    if discount is None or discount <= 0:
        # Nothing to spread (a printed 0.00 discount is common); just drop the empty field.
        out = {k: v for k, v in pred.items() if k != "basket-wide_discount"}
        return out

    prices = []
    for it in items:
        if not isinstance(it, dict):
            return pred
        v = parse_price_lenient(it.get("price"))
        if v is None:
            return pred  # refuse to guess: leave the prediction for a human to look at
        prices.append(v)

    subtotal = sum(prices)
    if subtotal <= 0 or discount >= subtotal:
        return pred  # a discount at/above the subtotal is a misread, not something to apply

    # Work in integer cents so the allocation is exact and auditable.
    target_cents = int(round((subtotal - discount) * 100))
    exact = [p * 100 * (subtotal - discount) / subtotal for p in prices]
    floors = [int(f) for f in exact]
    shortfall = target_cents - sum(floors)
    # Largest-remainder: hand the leftover cents to the items with the biggest fractional part.
    order = sorted(range(len(exact)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in range(shortfall):
        floors[order[i % len(floors)]] += 1

    new_items = []
    for it, cents in zip(items, floors):
        new_items.append({**it, "price": f"{cents / 100:.2f}"})

    out = {k: v for k, v in pred.items() if k != "basket-wide_discount"}
    out["items"] = new_items
    return out


# --------------------------------------------------------------------------------------
# Printed VAT amount extraction (best-effort, from noisy OCR text)
# --------------------------------------------------------------------------------------

_VAT_LINE_AMOUNT_RE = re.compile(r"(\d+[.,]\d{1,2})(?!\s*%)")


def extract_printed_vat_amount(ocr_text: str) -> float | None:
    """Best-effort scan for a printed VAT amount near a tax token in the raw OCR text. Returns
    None if nothing plausible is found (caller falls back to the computed residual, per
    task.md Section 5 point 3). This is deliberately simple: OCR text is noisy and further
    de-garbling is the model's job (Rule 1), not this code layer's."""
    for line in ocr_text.splitlines():
        if not TAX_TOKEN_RE.search(line):
            continue
        matches = _VAT_LINE_AMOUNT_RE.findall(line)
        if matches:
            try:
                return float(matches[-1].replace(",", "."))
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------------------
# Reconciliation (task.md Section 5 gap-repair logic)
# --------------------------------------------------------------------------------------

@dataclass
class ReconcileResult:
    status: str  # "ok" | "vat_gap" | "unrecoverable_gap" | "overcount" | "unparseable"
    passes: bool  # True if already within tolerance, or successfully gap-repaired
    items_sum: float | None = None
    discount: float | None = None
    total: float | None = None
    residual: float | None = None
    residual_pct: float | None = None  # signed, fraction of total
    repaired_vat_amount: float | None = None  # set only when status == "vat_gap"
    reasons: list[str] = field(default_factory=list)


def reconcile(pred: dict, ocr_text: str) -> ReconcileResult:
    """Applies task.md Section 5's gap-repair decision logic to a model's own prediction,
    independent of gold -- a business-validity signal, not a correctness check against ground
    truth. `passes` means: this prediction either already reconciles, or reconciles once the
    deterministic code layer's repair (inserting a VAT line) is applied to it."""
    if not isinstance(pred, dict):
        return ReconcileResult(status="unparseable", passes=False, reasons=["prediction is not an object"])

    items = pred.get("items")
    if not isinstance(items, list) or not items:
        return ReconcileResult(status="unparseable", passes=False, reasons=["items missing/empty"])

    items_sum = 0.0
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return ReconcileResult(status="unparseable", passes=False, reasons=[f"items[{i}] not an object"])
        price = parse_price_lenient(item.get("price"))
        if price is None:
            return ReconcileResult(status="unparseable", passes=False, reasons=[f"items[{i}].price unparseable: {item.get('price')!r}"])
        items_sum += price

    discount = 0.0
    if "basket-wide_discount" in pred:
        discount = parse_price_lenient(pred["basket-wide_discount"]) or 0.0

    total = parse_price_lenient(pred.get("total_price"))
    if total is None or total <= 0:
        return ReconcileResult(status="unparseable", passes=False, reasons=["total_price missing/unparseable/non-positive"])

    residual = total - (items_sum - discount)
    residual_pct = residual / total

    base = dict(items_sum=items_sum, discount=discount, total=total, residual=residual, residual_pct=residual_pct)

    if abs(residual_pct) <= RECONCILIATION_TOLERANCE:
        return ReconcileResult(status="ok", passes=True, **base)

    if residual_pct < -RECONCILIATION_TOLERANCE:
        return ReconcileResult(status="overcount", passes=False, reasons=["sum(items) - discount exceeds total"], **base)

    if VAT_SIGNATURE_LOW <= residual_pct <= VAT_SIGNATURE_HIGH and TAX_TOKEN_RE.search(ocr_text):
        printed = extract_printed_vat_amount(ocr_text)
        vat_amount = printed if printed is not None else residual
        return ReconcileResult(status="vat_gap", passes=True, repaired_vat_amount=vat_amount, **base)

    return ReconcileResult(
        status="unrecoverable_gap",
        passes=False,
        reasons=["residual not explained by 3% tolerance or the ~6.5% VAT signature -- likely a dropped item or misread price"],
        **base,
    )
