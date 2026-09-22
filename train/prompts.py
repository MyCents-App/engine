"""Shared system prompt and chat-message builder for the receipt-extraction task.

Used by data_prep.py (to build SFT training examples), baseline_eval.py (Phase 3 zero-shot
testing), and train_qlora.py / finetuned eval (Phase 4-5) so every phase prompts every model
identically -- baseline and fine-tuned numbers stay comparable, and the fine-tuned model is
trained on exactly the prompt it will be evaluated and served with.

Condensed from task.md Sections 2-4 (the model's share of the work; the code-layer
responsibilities in Section 4-5 are NOT the model's job and are deliberately left out of this
prompt -- see postprocess.py).

Rule 1's "keep the original language of each name" is load-bearing beyond the model: the
backend categorizes on the printed (Thai) text and translation is a separate display layer
downstream (RECEIPT_API.md, "Translation is not the engine's job"). Do not relax it.
"""
import json

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
