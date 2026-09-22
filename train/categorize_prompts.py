"""Shared prompt for the categorization task — the twin of prompts.py.

Used by the SFT data builder (server/scripts/build_categorization_sft.py),
by the trainer, and by the engine's /v1/categorize, so training and serving
prompt the model byte-identically. Change it here or nowhere.

The task: given a shop name and the items of ONE receipt (as the extraction
model returned them — Thai, as printed, optionally with a known English
name), assign each item a category and, when clear, a subcategory. One call
per receipt, not per item: the shop and the other items are the context that
tells a 2B model that "น้ำแข็ง" on a Suki Teenoi receipt is Food & Dining and
on a 7-Eleven receipt is Groceries.

Output is positional — the i-th entry is the i-th input item — so the
backend never has to match names back, and the model cannot drop or reorder
items without the code layer noticing (len(out) != len(in) is a hard fail).
"""

from __future__ import annotations

import json

CATEGORIES: dict[str, list[str]] = {
    "Food & Dining": [
        "Restaurants", "Fast food", "Coffee & café", "Street food",
        "Food delivery", "Bakery & desserts", "Drinks & beverages",
    ],
    "Groceries": [
        "Fresh produce", "Meat & seafood", "Dairy & eggs", "Snacks & sweets",
        "Instant & frozen", "Condiments & cooking", "Household supplies",
    ],
    "Transport": [
        "Public transit", "Ride-hailing", "Fuel & parking",
        "Vehicle maintenance", "Tolls & expressway",
    ],
    "Shopping": [
        "Clothing & shoes", "Electronics & gadgets", "Beauty & cosmetics",
        "Home & furniture", "Books & stationery", "Gifts",
    ],
    "Health & Wellness": [
        "Pharmacy & medicine", "Personal care", "Doctor & clinic", "Dental",
        "Fitness & gym", "Supplements & vitamins",
    ],
    "Bills & Utilities": [
        "Electricity", "Water", "Internet & phone", "Rent & housing",
        "Insurance", "Subscriptions",
    ],
    "Entertainment": [
        "Movies & cinema", "Streaming services", "Games & gaming",
        "Events & concerts", "Hobbies", "Social & nightlife",
    ],
    "Education": [
        "Tuition & fees", "Courses & workshops", "Books & materials",
        "Software & tools",
    ],
}

_TAXONOMY = "\n".join(
    f"- {cat}: {', '.join(subs)}" for cat, subs in CATEGORIES.items()
)

SYSTEM_PROMPT = f"""You assign spending categories to the line items of one Thai/English point-of-sale receipt. Item names are as printed on the receipt (Thai, English or mixed) and may carry OCR noise; some come with an English name in brackets.

Categories and their subcategories:
{_TAXONOMY}

Output exactly one JSON object: {{"items": [{{"c": "<category>", "s": "<subcategory or null>"}}, ...]}}
Rules:
1. Exactly one entry per input item, in the same order. Never merge, drop or add items.
2. "c" must be one of the eight category names, spelled exactly as above.
3. "s" must be one of that category's subcategories, or null when none clearly fits. Prefer null to a guess.
4. Use the shop and the other items as context: the same product means different things at a restaurant and a supermarket.
5. Output only the JSON object — no explanation, no code fences."""


def build_user_message(shop_name: str | None, items: list[dict]) -> str:
    """`items` are {"name": str, "name_en": str | None} in receipt order."""
    lines = [f"shop: {shop_name.strip() if shop_name else '(unknown)'}", "items:"]
    for i, item in enumerate(items, 1):
        name = item["name"].strip()
        en = (item.get("name_en") or "").strip()
        lines.append(f"{i}. {name} [{en}]" if en and en != name else f"{i}. {name}")
    return "\n".join(lines)


def serialize_target(labels: list[tuple[str, str | None]]) -> str:
    return json.dumps(
        {"items": [{"c": c, "s": s} for c, s in labels]},
        ensure_ascii=False, separators=(",", ":"),
    )


def build_messages(
    shop_name: str | None,
    items: list[dict],
    labels: list[tuple[str, str | None]] | None = None,
) -> list[dict]:
    """Model-agnostic chat messages; the trainer applies the tokenizer's chat
    template, exactly as prompts.build_messages does for extraction."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(shop_name, items)},
    ]
    if labels is not None:
        messages.append({"role": "assistant", "content": serialize_target(labels)})
    return messages


def parse_output(raw: str, expected: int) -> list[tuple[str, str | None]] | None:
    """The code layer's half of the contract: validate or reject, never repair.

    Returns None when the output is not usable — wrong length, unknown
    category, subcategory outside its category. A None means the backend
    falls through to user review, exactly as a stage-5 decline does today.
    """
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        entries = data["items"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(entries, list) or len(entries) != expected:
        return None
    out: list[tuple[str, str | None]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        c, s = entry.get("c"), entry.get("s")
        if c not in CATEGORIES:
            return None
        if s is not None and s not in CATEGORIES[c]:
            s = None  # a wrong subcategory is dropped, not fatal
        out.append((c, s))
    return out
