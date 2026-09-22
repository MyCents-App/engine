"""Turn the engine's own drafts into annotation skeletons to correct.

Annotating 200 receipts from blank files is not the job. The extraction
checkpoint is 67.6% money-exact, so most of the shop name, the item names and
the prices are already right -- the work is correcting what is wrong and
adding the two category fields per item, which is a fraction of the typing.

    # 1. engine up (RUN.md), photos in SuryaOCR/data/input
    .venv\\Scripts\\python.exe review_photos.py --no-pause
    # 2. drafts -> skeletons
    .venv\\Scripts\\python.exe make_annotations.py
    # 3. edit data/annotations/*.json, then
    .venv\\Scripts\\python.exe validate_annotations.py

Reads the response JSON review_photos.py saves (camelCase, the shape the app
receives) and writes the annotation shape ANNOTATION.md specifies. The OCR
text rides along in `ocr_pages`: the extraction model is prompted with OCR
text, never with the clean names typed here, so keeping it means one pass over
a photo yields a complete training pair for both adapters and nothing has to
be re-OCR'd later.

Existing files are never overwritten -- a second run after more photos have
been shot adds only what is new, so a re-run cannot destroy an afternoon of
annotation. --force overrides that, per file.

KNOWN LIMIT: review_photos.py posts ONE photo per request, so a receipt shot
across two frames arrives as two partial drafts. For those, post both files in
a single /v1/extract call and save the response into the review folder by
hand; see ANNOTATION.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DRAFTS = ROOT.parent / "SuryaOCR" / "data" / "review"
DEFAULT_OUT = ROOT / "data" / "annotations"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS,
                   help="folder of review_photos.py responses (default: %(default)s)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="where to write skeletons (default: %(default)s)")
    p.add_argument("--annotator", default="",
                   help="pre-fill the annotator field")
    p.add_argument("--force", action="store_true",
                   help="overwrite skeletons that already exist (DESTROYS edits)")
    return p.parse_args(argv)


def skeleton(draft: dict, stem: str, annotator: str) -> dict:
    """Map the API response onto the annotation shape.

    `taxAmount` is already lifted out of items[] by the API's _split_tax, and
    the extraction prompt's rule 5 only has the model emit a vat row when tax
    sits on TOP of the listed prices -- so a non-null taxAmount implies
    tax_added_on_top. It is written as a guess for the annotator to confirm
    against the photo, not as a fact.
    """
    tax = draft.get("taxAmount")
    items = []
    for item in draft.get("items") or []:
        items.append({
            "name": item.get("name"),
            "name_en": item.get("nameEn"),   # the engine always sends null
            "price": item.get("price"),
            "category": None,                # <- you fill these two
            "subcategory": None,
        })

    return {
        "receipt_id": stem,
        "photos": [draft.get("_photo") or f"{stem}.jpg"],

        "shop_name": draft.get("shopName"),
        "receipt_date": draft.get("receiptDate"),

        "items": items,

        "basket_discount": draft.get("basketDiscount"),
        "tax_amount": tax,
        "tax_added_on_top": tax is not None,
        "total_price": draft.get("totalAmount"),

        "annotator": annotator,
        "notes": "",

        # Everything below is carried through for the builder and for triage.
        # Leave it alone while annotating.
        "ocr_pages": draft.get("ocrTexts") or [],
        "_engine_reconciles": draft.get("reconciles"),
        "_engine_reconcile_status": draft.get("reconcileStatus"),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.drafts.is_dir():
        sys.exit(f"error: no draft folder at {args.drafts}\n"
                 f"       run review_photos.py first (see RUN.md)")

    drafts = sorted(args.drafts.glob("*.json"))
    if not drafts:
        sys.exit(f"error: no .json drafts in {args.drafts}")

    args.out.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    suspect: list[str] = []

    for path in drafts:
        try:
            draft = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {path.name}: unreadable ({exc})")
            failed += 1
            continue

        if not draft.get("ok"):
            # The model returned nothing usable for this photo. There is no
            # draft to correct, so it needs annotating from the photo by hand
            # rather than a skeleton full of nulls.
            print(f"  ! {path.name}: engine returned ok=false -- annotate by hand")
            failed += 1
            continue

        dest = args.out / path.name
        if dest.exists() and not args.force:
            skipped += 1
            continue

        record = skeleton(draft, path.stem, args.annotator)
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        written += 1
        if draft.get("reconciles") is False:
            suspect.append(path.stem)

    print(f"\n{written} skeleton(s) written to {args.out}")
    if skipped:
        print(f"{skipped} already existed and were left alone (--force to replace)")
    if failed:
        print(f"{failed} draft(s) could not be used")

    if suspect:
        print(f"\n{len(suspect)} receipt(s) the engine could not reconcile -- its "
              f"numbers do not add up, so check these against the photo first:")
        for stem in suspect:
            print(f"  {stem}")

    print("\nNext: fill category and subcategory on every item "
          "(ANNOTATION.md has the taxonomy), then run validate_annotations.py")


if __name__ == "__main__":
    main()
