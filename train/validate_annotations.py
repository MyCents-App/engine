"""Check annotated receipts before anyone trains on them.

Catches the mistakes that are invisible by eye and expensive later: a category
spelled `Coffee & cafe` without the accent, a price written `7.0`, a basket
discount subtracted from the item prices as well as recorded, a subcategory
borrowed from the wrong category. Every one of those either breaks the builder
or, worse, silently teaches the model something wrong.

Labels come from the receipt PHOTO, so a target name is what the receipt
printed rather than what OCR recovered. That is not an error -- mapping the
lossy text onto the truth is the job -- so name grounding is REPORTED and
never enforced.

    .venv\\Scripts\\python.exe validate_annotations.py                  # data/annotations/*.json
    .venv\\Scripts\\python.exe validate_annotations.py data/real.jsonl  # a built JSONL
    .venv\\Scripts\\python.exe validate_annotations.py --strict         # unfilled c = error

Exits non-zero if anything is wrong, so it can gate the builder.

It prints two things worth watching while the work is in progress. The
category distribution, because three of the eight categories are ones real
receipts are the only source for and it is worth knowing you have none of them
while there is still time to photograph some. And name grounding -- how often
an item name appears verbatim in the OCR input -- because that is how much
reconstruction the model is being asked to do, and a low number is an OCR or
photography problem rather than something more epochs will fix.
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
TOLERANCE = postprocess.RECONCILIATION_TOLERANCE   # 3%, same as the code layer
TAX_NAMES = {"vat", "tax"}

# Three categories the synthetic catalog cannot teach (0 / 0 / 1 rows), so real
# receipts are the only source. Flagged so their absence is visible early.
STARVED = ["Transport", "Bills & Utilities", "Entertainment"]

ALLOWED_TARGET_KEYS = {"shop_name", "items", "basket-wide_discount", "total_price"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="*", type=Path,
                   help="JSONL files or folders of per-receipt JSON "
                        f"(default: {DEFAULT_DIR})")
    p.add_argument("--strict", action="store_true",
                   help="treat items with no category as errors, not pending")
    return p.parse_args(argv)


def money(value, field: str, errors: list[str]) -> float | None:
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
                        f"{category!r} in 'c' and {sub!r} in 's'")
    for category in CATEGORIES:
        if category.lower() == text.lower():
            return f" -- did you mean {category!r}? (capitalisation differs)"

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


def grounding(record: dict) -> tuple[int, int]:
    """(item names literally present in the input, names checked).

    Reported, never enforced. Labels come from the photo, so a target name is
    what the receipt printed rather than what OCR recovered -- and mapping the
    lossy text onto the truth is the job. But the gap is worth watching: it is
    how much reconstruction the model is being asked to do, and if it is large
    the honest reading is that OCR (or the photography) is the thing to fix,
    not the number of training epochs.
    """
    text = record.get("input")
    if not isinstance(text, str):
        return 0, 0
    grounded = checked = 0
    for item in record.get("target", {}).get("items") or []:
        name = str((item or {}).get("name") or "").strip()
        if not name or name.lower() in TAX_NAMES:
            continue
        checked += 1
        if name in text:
            grounded += 1
    return grounded, checked


def check(record: dict, strict: bool) -> tuple[list[str], int, Counter]:
    """Returns (errors, items awaiting a category, category counts)."""
    errors: list[str] = []
    counts: Counter = Counter()
    pending = 0

    text = record.get("input")
    if not isinstance(text, str) or not text.strip():
        errors.append("input is empty -- the model is prompted with OCR text, "
                      "so a record without it teaches nothing")
    kind = (record.get("meta") or {}).get("kind")
    if not kind:
        errors.append("meta.kind is missing ('real' or 'synthetic'); the "
                      "evaluator reports the two separately and cannot without it")

    target = record.get("target")
    if not isinstance(target, dict):
        errors.append("target is missing or not an object")
        return errors, 0, counts

    unexpected = set(target) - ALLOWED_TARGET_KEYS
    if unexpected:
        errors.append(f"target has unexpected key(s) {sorted(unexpected)} -- the "
                      f"contract is {sorted(ALLOWED_TARGET_KEYS)}")

    if not target.get("shop_name") or not str(target["shop_name"]).strip():
        errors.append("target.shop_name is empty")

    if "basket-wide_discount" in target and target["basket-wide_discount"] is None:
        errors.append("basket-wide_discount is null -- omit the key entirely "
                      "when there is no basket discount")

    items = target.get("items")
    if not isinstance(items, list) or not items:
        errors.append("target.items is empty")
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
        lowered = str(name or "").strip().lower()
        is_tax = lowered in TAX_NAMES or lowered.startswith(("vat ", "tax ", "ภาษี"))

        price = money(item.get("price"), f"{where}.price", errors)
        if price is None:
            prices_ok = False
        else:
            subtotal += price

        category, subcategory = item.get("c"), item.get("s")

        if is_tax and category is not None:
            errors.append(f"{where} is the tax row and must have c: null -- "
                          f"categorized as a purchase, every per-category "
                          f"spending total is wrong by the tax")
            continue

        if category is None:
            if subcategory is not None:
                errors.append(f"{where}: s is {subcategory!r} with c null")
            if is_tax:
                continue
            pending += 1
            if strict:
                errors.append(f"{where}.c is not annotated (null). If the OCR "
                              f"genuinely destroyed the name, that is correct "
                              f"-- note it and move on")
            continue

        if category not in CATEGORIES:
            errors.append(f"{where}.c {category!r} is not one of the eight"
                          f"{category_hint(category)}")
            continue

        counts[category] += 1
        if subcategory is not None and subcategory not in CATEGORIES[category]:
            owner = [c for c, subs in CATEGORIES.items() if subcategory in subs]
            hint = (f" -- it belongs to {owner[0]!r}" if owner
                    else " -- not in the taxonomy")
            errors.append(f"{where}.s {subcategory!r} is not under "
                          f"{category!r}{hint}")

    discount = money(target.get("basket-wide_discount"),
                     "basket-wide_discount", errors) or 0.0
    total = money(target.get("total_price"), "total_price", errors)

    # The invariant. The vat row, when present, is one of the item prices, so
    # it is already inside subtotal.
    if prices_ok and total is not None and total > 0:
        expected = subtotal - discount
        drift = abs(expected - total) / total
        if drift > TOLERANCE:
            errors.append(
                f"does not balance: items {subtotal:.2f} - discount "
                f"{discount:.2f} = {expected:.2f}, but total_price is "
                f"{total:.2f} ({drift:.1%} off, tolerance {TOLERANCE:.0%}). "
                f"Usually a unit price where a line total belongs, or a basket "
                f"discount subtracted from the items as well as recorded.")

    return errors, pending, counts


def load(paths: list[Path]) -> list[tuple[str, dict]]:
    """Read JSONL files and folders of per-receipt JSON into (label, record)."""
    out: list[tuple[str, dict]] = []
    for path in paths:
        if path.is_dir():
            files = sorted(path.glob("*.json"))
            if not files:
                sys.exit(f"error: no .json files in {path}")
            for file in files:
                try:
                    out.append((file.name, json.loads(file.read_text(encoding="utf-8"))))
                except json.JSONDecodeError as exc:
                    out.append((file.name, {"_unreadable": str(exc)}))
        elif path.is_file():
            for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
                line = line.strip()
                if not line:
                    continue
                label = f"{path.name}:{lineno}"
                try:
                    out.append((label, json.loads(line)))
                except json.JSONDecodeError as exc:
                    out.append((label, {"_unreadable": str(exc)}))
        else:
            sys.exit(f"error: no such path: {path}")
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = args.paths or [DEFAULT_DIR]
    records = load(paths)
    if not records:
        sys.exit("error: nothing to check")

    counts: Counter = Counter()
    seen: dict[str, str] = {}
    kinds: Counter = Counter()
    bad = pending_total = item_total = 0
    grounded_total = grounded_checked = 0

    for label, record in records:
        if "_unreadable" in record:
            print(f"\n{label}\n  ! unreadable JSON: {record['_unreadable']}")
            bad += 1
            continue

        errors, pending, file_counts = check(record, args.strict)
        counts.update(file_counts)
        pending_total += pending
        item_total += len(record.get("target", {}).get("items") or [])
        g, c = grounding(record)
        grounded_total += g
        grounded_checked += c
        kinds[(record.get("meta") or {}).get("kind", "?")] += 1

        rid = record.get("id")
        if rid and rid in seen:
            errors.append(f"id {rid!r} already used by {seen[rid]}")
        elif rid:
            seen[rid] = label

        if errors:
            bad += 1
            print(f"\n{label}")
            for error in errors:
                print(f"  ! {error}")

    print(f"\n{'-' * 68}")
    print(f"{len(records)} record(s), {item_total} items, {bad} with problems")
    print("kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if pending_total and item_total:
        print(f"{pending_total} item(s) have no category "
              f"({pending_total / item_total:.0%})")

    if grounded_checked:
        share = grounded_total / grounded_checked
        print(f"\nname grounding: {grounded_total}/{grounded_checked} "
              f"({share:.0%}) of item names appear verbatim in the OCR input")
        if share < 0.70:
            print("  The rest are reconstructions the model has to learn from "
                  "context.\n  A low number here usually means the OCR or the "
                  "photography is the thing\n  to fix, not the number of "
                  "training epochs -- worth a look first.")

    if counts:
        print("\ncategory distribution:")
        labelled = sum(counts.values())
        for category in CATEGORIES:
            n = counts.get(category, 0)
            flag = ""
            if category in STARVED:
                flag = ("   <- synthetic data has none of this; only real "
                        "receipts teach it" if n == 0
                        else "   <- starved category, keep going")
            print(f"  {category:20s} {n:5d}  {n / labelled:6.1%}{flag}")

    if bad:
        print(f"\n{bad} record(s) need fixing before the builder runs.")
        sys.exit(1)
    if pending_total and not args.strict:
        print(f"\nNo errors. {pending_total} item(s) still need a category "
              f"(--strict to fail on those).")
        return
    print("\nAll good.")


if __name__ == "__main__":
    main()
