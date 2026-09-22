"""Shared system prompt and chat-message builder for the receipt task.

ONE task: raw Surya OCR text in, one JSON object out carrying the shop, the items with
prices, AND a spending category per item. Extraction and categorization are the same
generation, against the same adapter -- not two adapters as CATEGORIZATION.md originally
planned. See ANNOTATION.md for the dataset contract and section 7 there for what the merge
costs.

Used by the dataset builder, train_qlora.py, the evaluators and app/extraction.py, so every
phase prompts the model identically -- baseline and fine-tuned numbers stay comparable, and
the model is trained on exactly the prompt it is evaluated and served with. Change it here
or nowhere.

Rules 1-7 are condensed from task.md Sections 2-4; rules 8-11 are the categorization half.
The code-layer responsibilities (task.md Sections 4-5) are deliberately NOT in this prompt --
see postprocess.py. In particular the model never spreads the basket discount, never parses
the date, and never translates.

Rule 1's "keep the original language of each name" is load-bearing beyond the model: the
backend categorizes on the printed (Thai) text and translation is a separate display layer
downstream (RECEIPT_API.md, "Translation is not the engine's job"). Do not relax it.

The taxonomy is imported rather than written out, so categorize_prompts.CATEGORIES stays the
single source of truth for what the eight categories and 47 subcategories are -- the
validator, the evaluator and this prompt cannot disagree about them.
"""
import json

from categorize_prompts import CATEGORIES

_TAXONOMY = "\n".join(
    f"- {category}: {', '.join(subs)}" for category, subs in CATEGORIES.items()
)

SYSTEM_PROMPT = f"""You convert raw OCR text from a Thai/English point-of-sale receipt into one JSON object. The OCR text is noisy (character-level errors, garbled lines, <br> artifacts, duplicated lines, inconsistent number formatting) and may mix Thai and English.

Output exactly one JSON object with this schema, keys in this order:
{{"shop_name": "string", "items": [{{"name": "string", "price": "NN.DD", "c": "category or null", "s": "subcategory or null"}}], "basket-wide_discount": "NN.DD", "total_price": "NN.DD"}}

"basket-wide_discount" is optional: include it only when Rule 6 below applies; omit the key entirely otherwise. All prices are strings formatted as NN.DD (two decimals, no currency symbol, no thousands separator, no sign).

Categories and their subcategories:
{_TAXONOMY}

Rules:
1. Correct OCR character errors and normalize the merchant name and item names into clean, readable text (keep the original language of each name).
2. Prices in items[] are line totals, not unit prices. A quantity line such as "2 ChickN'Roll ALC 78.00" or "mineral-water 3 30.00" stays as ONE item at that line's total price.
3. Buy-one-get-one: include the item once in items[]. Do not add the free duplicate as a separate item.
4. Discount naming a specific item: subtract that discount from that item's price. Only the resulting (already-discounted) price appears in items[] -- do not add a separate discount line for it.
5. VAT/tax: if tax is excluded from the listed prices, or included but also shown added on top, add one extra item named "vat" with the tax amount, and set its "c" and "s" to null. If VAT is already included in the item prices with nothing added on top, do not add a vat line.
6. Basket-wide discount (a discount not tied to any specific item): put its amount in "basket-wide_discount". Do not subtract it from any item price -- leave item prices as printed/derived from rules 2-5.
7. total_price is the receipt's final printed total.
8. Every item gets a "c" and an "s". "c" is one of the eight category names above, spelled exactly, or null. "s" is one of that same category's subcategories, spelled exactly, or null.
9. Use the shop and the other items as context: the same product means different things in different places. Ice on a restaurant receipt is Food & Dining; ice at a convenience store is Groceries.
10. Prefer null to a guess. A wrong subcategory is worse than none, so write null for "s" whenever no subcategory clearly fits. If the OCR text does not contain enough to identify what an item is, set "c" to null as well -- do not infer a category from the price alone.
11. Never invent an item name that is not recoverable from the OCR text, and never merge, drop or reorder items. Output exactly one entry per item line on the receipt.

Output ONLY the JSON object. No explanation, no markdown code fences, no extra text before or after it."""


def serialize_target(target: dict) -> str:
    """Canonical compact serialization for assistant completions: preserves source key order,
    keeps Thai characters unescaped (shorter + more natural token sequence than \\uXXXX escapes),
    and drops unnecessary whitespace to save tokens during training and inference."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_messages(ocr_text: str, target: dict | None = None) -> list[dict]:
    """Build a model-agnostic chat message list. When `target` is given, an assistant turn with
    the canonical JSON completion is appended (for SFT training data); otherwise the list ends
    after the user turn (for zero-shot / generation-time prompting).

    Deliberately does NOT apply any specific model's chat template here -- each phase applies the
    winning/candidate model's own tokenizer.apply_chat_template at load time, per the plan."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ocr_text},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": serialize_target(target)})
    return messages
