"""Check annotated receipts before anyone trains on them.

Catches the mistakes that are invisible by eye and expensive later: a category
spelled `Coffee & cafe` without the accent, a price written `7.0`, a basket
discount subtracted from the item prices as well as recorded, a subcategory
borrowed from the wrong category. Every one of those either crashes the
builder or, worse, silently teaches the model something wrong.

    .venv\\Scripts\\python.exe validate_annotations.py
    .venv\\Scripts\\python.exe validate_annotations.py --strict   # unfilled = error

Exits non-zero if anything is wrong, so it can gate the builder.

It also prints the category distribution across everything annotated so far.
That is the number worth watching while the work is in progress: the synthetic
catalog has 0 Transport, 0 Bills & Utilities and 1 Entertainment row, so real
receipts are the only source for those three, and it is worth knowing you have
none of them while there is still time to go and photograph some.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import postprocess
from categorize_prompts import CATEGORIES

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "data" / "annotations"

PRICE_RE = re.compile(r"^\d+\.\d{2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TOLERANCE = postprocess.RECONCILIATION_TOLERANCE   # 3%, same as the code layer

# Three categories the synthetic catalog cannot teach. Tracked separately so
# the summary can say plainly whether the annotation effort has reached them.
STARVED = ["Transport", "Bills & Utilities", "Entertainment"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--strict", action="store_true",
                   help="treat not-yet-annotated items as errors, not pending")
    return p.parse_args(argv)


def money(value, field: str, errors: list[str]) -> float | None:
    """Require the NN.DD contract exactly. Returns the value, or None."""
    if value is None:
        return None
    if not isinstance(value, str) or not PRICE_RE.match(value):
        errors.append(f"{field}: {value!r} is not 'NN.DD' "
                      f"(two decimals, no symbol, no thousands separator)")
        return None
    return float(value)


def category_hint(value) -> str:
    """Name the likely intent behind a bad category.

    The two mistakes that actually happen are a subcategory typed into the
    category field (`Coffee & café` is a subcategory of Food & Dining, not a
    category) and a near-miss on spelling or the accent. Both are one keypress
    to fix once named, and baffling otherwise.
    """
    text = str(value).strip()
    for category, subs in CATEGORIES.items():
        for sub in subs:
            if sub.lower() == text.lower():
                return (f" -- {sub!r} is a SUBcategory of {category!r}; put "
                        f"{category!r} in category and {sub!r} in subcategory")
    for category in CATEGORIES:
        if category.lower() == text.lower():
            return f" -- did you mean {category!r}? (capitalisation differs)"
    # Compare ignoring case, spaces and the accent, which is how 'Coffee &
    # cafe' and 'coffee&café' both end up here.
    def squash(s: str) -> str:
        return (s.lower().replace(" ", "").replace("&", "")
                .replace("é", "e").replace("-", ""))
    target = squash(text)
    for category, subs in CATEGORIES.items():
        if squash(category) == target:
            return f" -- did you mean {category!r}?"
        for sub in subs:
            if squash(sub) == target:
                return (f" -- did you mean the subcategory {sub!r}, under "
                        f"{category!r}? Note the accent in 'café'."
                        if "é" in sub else
                        f" -- {sub!r} is a subcategory of {category!r}")
    return ""


def check(record: dict, strict: bool) -> tuple[list[str], int, Counter]:
    """Returns (errors, items_pending_annotation, category counts)."""
    errors: list[str] = []
    counts: Counter = Counter()
    pending = 0

    for field in ("receipt_id", "shop_name", "items", "total_price"):
        if field not in record:
            errors.append(f"missing required field {field!r}")
    if errors:
        return errors, 0, counts

    if not record.get("shop_name") or not str(record["shop_name"]).strip():
        errors.append("shop_name is empty")

    date = record.get("receipt_date")
    if date is not None and not DATE_RE.match(str(date)):
        errors.append(f"receipt_date: {date!r} is not YYYY-MM-DD or null "
                      f"(remember Buddhist year 2569 -> 2026)")

    items = record.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items is empty")
        return errors, 0, counts

    subtotal = 0.0
    prices_ok = True
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where} is not an object")
            prices_ok = False
            continue

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{where}.name is empty")

        price = money(item.get("price"), f"{where}.price", errors)
        if price is None:
            prices_ok = False
        else:
            subtotal += price

        lowered = str(name or "").strip().lower()
        if lowered in {"vat", "tax"} or lowered.startswith(("vat ", "tax ", "ภาษี")):
            errors.append(
                f"{where}.name is {name!r} -- tax must not be an item. Move the "
                f"amount to tax_amount and set tax_added_on_top")

        category, subcategory = item.get("category"), item.get("subcategory")
        if category is None:
            pending += 1
            if strict:
                errors.append(f"{where}.category is not annotated")
            if subcategory is not None:
                errors.append(f"{where}: subcategory {subcategory!r} given with "
                              f"no category")
            continue

        if category not in CATEGORIES:
            errors.append(f"{where}.category {category!r} is not one of the "
                          f"eight{category_hint(category)}")
            continue

        counts[category] += 1
        if subcategory is not None and subcategory not in CATEGORIES[category]:
            owner = [c for c, subs in CATEGORIES.items() if subcategory in subs]
            hint = (f" -- it belongs to {owner[0]!r}" if owner
                    else " -- not in the taxonomy")
            errors.append(f"{where}.subcategory {subcategory!r} is not under "
                          f"{category!r}{hint}")

    discount = money(record.get("basket_discount"), "basket_discount", errors) or 0.0
    tax = money(record.get("tax_amount"), "tax_amount", errors) or 0.0
    total = money(record.get("total_price"), "total_price", errors)
    on_top = bool(record.get("tax_added_on_top"))

    if record.get("tax_amount") is None and on_top:
        errors.append("tax_added_on_top is true but tax_amount is null")

    # The invariant. Only meaningful once every price parsed.
    if prices_ok and total is not None and total > 0:
        expected = subtotal - discount + (tax if on_top else 0.0)
        drift = abs(expected - total) / total
        if drift > TOLERANCE:
            errors.append(
                f"does not balance: items {subtotal:.2f} - discount "
                f"{discount:.2f}" + (f" + tax {tax:.2f}" if on_top else "")
                + f" = {expected:.2f}, but total_price is {total:.2f} "
                f"({drift:.1%} off, tolerance {TOLERANCE:.0%}). Usually a unit "
                f"price where a line total belongs, or a basket discount "
                f"subtracted from the items as well as recorded.")

    return errors, pending, counts


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.dir.is_dir():
        sys.exit(f"error: no annotation folder at {args.dir}")

    paths = sorted(args.dir.glob("*.json"))
    if not paths:
        sys.exit(f"error: no .json files in {args.dir}")

    counts: Counter = Counter()
    seen_ids: dict[str, str] = {}
    bad_files = 0
    total_pending = 0
    total_items = 0

    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"\n{path.name}\n  ! unreadable JSON: {exc}")
            bad_files += 1
            continue

        errors, pending, file_counts = check(record, args.strict)
        counts.update(file_counts)
        total_pending += pending
        total_items += len(record.get("items") or [])

        receipt_id = record.get("receipt_id")
        if receipt_id in seen_ids:
            errors.append(f"receipt_id {receipt_id!r} already used by "
                          f"{seen_ids[receipt_id]}")
        elif receipt_id:
            seen_ids[receipt_id] = path.name

        if errors:
            bad_files += 1
            print(f"\n{path.name}")
            for error in errors:
                print(f"  ! {error}")

    print(f"\n{'-' * 68}")
    print(f"{len(paths)} file(s), {total_items} items, {bad_files} with problems")
    if total_pending:
        print(f"{total_pending} item(s) not yet annotated "
              f"({total_pending / total_items:.0%})" if total_items else "")

    if counts:
        print("\ncategory distribution:")
        labelled = sum(counts.values())
        for category in CATEGORIES:
            n = counts.get(category, 0)
            flag = ""
            if category in STARVED:
                flag = ("   <- synthetic data has none of this; only real "
                        "receipts teach it" if n == 0 else "   <- starved "
                        "category, keep going")
            print(f"  {category:20s} {n:5d}  {n / labelled:6.1%}{flag}")

    if bad_files:
        print(f"\n{bad_files} file(s) need fixing before the builder runs.")
        sys.exit(1)
    if total_pending and not args.strict:
        print(f"\nNo errors. {total_pending} item(s) still need a category "
              f"(--strict to fail on those).")
        return
    print("\nAll good.")


if __name__ == "__main__":
    main()
