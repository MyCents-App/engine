"""The extraction-only prompt checkpoint-550 was trained and served with. FROZEN.

prompts.py now holds the joint prompt (extraction + categories). checkpoint-550
never saw it, so scoring that checkpoint through the joint prompt would measure
a prompt mismatch rather than the model. eval_joint.py --legacy uses this to put
the old checkpoint on the same validation receipts as the joint one, which is
the like-for-like extraction comparison ANNOTATION.md section 7 asks for.

Byte-identical to the SYSTEM_PROMPT in the training project that produced
checkpoint-550 and to train/prompts.py as of 820dace. Do not edit it: the only
thing it is for is being exactly what that checkpoint learned.
"""
from prompts import serialize_target

SYSTEM_PROMPT = """You convert raw OCR text from a Thai/English point-of-sale receipt into one JSON object. The OCR text is noisy (character-level errors, garbled lines, <br> artifacts, inconsistent number formatting) and may mix Thai and English.

Output exactly one JSON object with this schema, keys in this order:
{"shop_name": "string", "items": [{"name": "string", "price": "NN.DD"}], "basket-wide_discount": "NN.DD", "total_price": "NN.DD"}

"basket-wide_discount" is optional: include it only when Rule 6 below applies; omit the key entirely otherwise. All prices are strings formatted as NN.DD (two decimals, no currency symbol, no thousands separator, no sign).

Rules:
1. Correct OCR character errors and normalize the merchant name and item names into clean, readable text (keep the original language of each name).
2. Prices in items[] are line totals, not unit prices. A quantity line such as "2 ChickN'Roll ALC 78.00" or "mineral-water 3 30.00" stays as ONE item at that line's total price.
3. Buy-one-get-one: include the item once in items[]. Do not add the free duplicate as a separate item.
4. Discount naming a specific item: subtract that discount from that item's price. Only the resulting (already-discounted) price appears in items[] -- do not add a separate discount line for it.
5. VAT/tax: if tax is excluded from the listed prices, or included but also shown added on top, add one extra item named "vat" (or "tax") with the tax amount. If VAT is already included in the item prices with nothing added on top, do not add a vat line.
6. Basket-wide discount (a discount not tied to any specific item): put its amount in "basket-wide_discount". Do not subtract it from any item price -- leave item prices as printed/derived from rules 2-5.
7. total_price is the receipt's final printed total.

Output ONLY the JSON object. No explanation, no markdown code fences, no extra text before or after it."""


def build_messages(ocr_text: str, target: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ocr_text},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": serialize_target(target)})
    return messages
