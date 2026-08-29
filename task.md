# Receipt Extraction Model — Project Handoff Spec

## 1. Objective

Build the second stage of a receipt-processing pipeline for an expense-tracking mobile app.
The full pipeline is:

`receipt photo → Surya OCR (already deployed) → THIS COMPONENT → expense categorizer (separate model)`

**This component** takes the raw text produced by Surya OCR and converts it into structured JSON
(merchant, line items with prices, total, and — when present — a basket-wide discount). The
categorizer that follows is out of scope for this work.

The task is a **generative sequence-to-JSON** task, not span extraction. The target is not a copy
of the input: it corrects OCR character errors, normalizes merchant names, and restructures the
content. Any approach based on token classification / NER is out of scope.

## 2. Input and output

- **Input:** raw Surya OCR text. Mixed Thai + English. Noisy: character-level OCR errors, garbled
  lines, `<br>` artifacts, inconsistent number formatting.
- **Output:** a single JSON object with this contract:

```json
{
  "shop_name": "string",
  "items": [
    { "name": "string", "price": "NN.DD" }
  ],
  "total_price": "NN.DD",
  "basket-wide_discount": "NN.DD"        // present ONLY for basket-wide discounts (see Rule 4)
}
```

All monetary values are strings formatted as `NN.DD` (two decimals, no sign, no thousands
separators). `basket-wide_discount` is omitted entirely when there is no basket-wide discount.


## 3. Processing rules (model output)

1. **Buy-one-get-one:** include the item **once** in `items[]` (do not list the free duplicate).
2. **Discount naming a specific item:** subtract that discount from that item's price. The model
   performs subtraction only; the resulting price appears in `items[]`.
3. **VAT / tax excluded receipts type (or VAT-included but vat is added on top):** add a line item  named `vat` (or `tax`)
   with the tax amount in `items[]. **VAT included(vat is already added on item price):** do nothing (no vat line item).
4. **Basket-wide discount (no item named):** put the amount in the `basket-wide_discount` field. Do **not**
   spread it into item prices. The code layer performs the subtraction and the reconciliation.
5. Prices are **line totals**, not unit prices. Quantity lines (e.g. `2 ChickN'Roll ALC  78.00` or `mineral-water 3 30.00`)
  are kept as a single line at the line total. The downstream categorizer must expect line totals.

## 4. Division of labor: model vs. code layer

**Model produces:** OCR-corrected merchant name; `items[]` with per-item discounts already
subtracted (Rule 2); BOGO deduplicated (Rule 1); vat/tax line item when tax is on top (Rule 3);
the `basket-wide_discount` field for basket-wide discounts (Rule 4); `total_price`.

**Code layer (deterministic post-processor) performs:**
- Subtract `basket-wide_discount` from the item-sum.
- Run the reconciliation check (Section 5).
- VAT gap-repair (Section 5), gated — never blindly absorb a gap as VAT.
- Retry loop on failure. **"Retry" means re-invoking the model with a changed condition** (higher
  temperature or a corrective instruction such as "items do not sum to total; a tax line may be
  missing — extract again"). Re-running an identical prompt at temperature 0 is a no-op and is not
  a valid retry.

## 5. Validation invariants (used by the code layer and by evaluation)

- **No-discount receipts:** `sum(item prices) ≈ total_price`, within **3%**.
- **Basket-discount receipts:** `sum(item prices) − discount ≈ total_price`, within **3%**.
- **VAT-on-top signature:** a single 7% VAT added on top equals `0.07 / 1.07 = 6.54%` of the
  tax-inclusive total. In the dataset this holds to the cent for every VAT row.

**Gap-repair logic (code layer):**
1. `residual = total − (sum(items) − discount)`.
2. `|residual| ≤ 3%` of total → accept.
3. `residual > 3%`, `residual ≈ 6.5%` of total (tolerance band ~5.5–7.5% for rounding), **and** the
   OCR input contains a tax token (`vat` / `tax` / `ภาษี` / `VAT 7%`) → insert a vat line. Use the
   **printed** VAT number if parseable; fall back to `residual` only if it is not.
4. `residual > 3%` but NOT the ~6.5% signature → likely a dropped item or misread price, not tax.
   Retry to recover; do not label the gap as vat. If unrecoverable, flag for review.
5. `residual < −3%` (sum exceeds total) → over-count (missed basket discount or duplicated line).
   Never insert vat here.

