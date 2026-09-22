"""Turn the engine's own drafts into annotation skeletons to correct.

Annotating 200 receipts from blank files is not the job. On real held-out
receipts the extraction checkpoint scores 61.8% money-exact, 80.8% price
recall and 79.4% total-exact, so most of the shop name, the item names and the
prices arrive correct -- the work is fixing what is wrong and adding the two
category fields per item.

    # 1. engine up (RUN.md), photos in SuryaOCR/data/input
    .venv\\Scripts\\python.exe review_photos.py --no-pause
    # 2. drafts -> skeletons
    .venv\\Scripts\\python.exe make_annotations.py
    # 3. edit data/annotations/*.json, then
    .venv\\Scripts\\python.exe validate_annotations.py
    # 4. collect into the file the trainer reads
    .venv\\Scripts\\python.exe make_annotations.py --collect data/real.jsonl

Writes the record shape ANNOTATION.md section 1 specifies -- {id, meta, input,
target} -- one file per receipt because a 200-line JSONL is miserable to edit
by hand and one bad keystroke breaks every record after it. --collect
concatenates the finished files into the JSONL the builder wants.

`input` is the raw OCR text the engine read, straight off the response. The
model is prompted with OCR text and never with the clean names typed here, so
this is the half of the pair that must not be tidied.

Existing files are never overwritten, so a re-run after shooting more photos
adds only what is new and cannot destroy an afternoon of annotation. --force
overrides that.

KNOWN LIMIT: review_photos.py posts ONE photo per request, so a receipt shot
across two frames arrives as two partial drafts. For those, post both files in
a single /v1/extract call and save the response into the review folder by
hand -- `input` must be the stitched text, which is what the model sees.
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
                   help="where skeletons live (default: %(default)s)")
    p.add_argument("--kind", default="real",
                   help="meta.kind for these records (default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="overwrite skeletons that already exist (DESTROYS edits)")
    p.add_argument("--collect", type=Path, default=None,
                   help="skip generation; concatenate --out/*.json into this JSONL")
    return p.parse_args(argv)


def skeleton(draft: dict, stem: str, kind: str) -> dict:
    """Map the API response onto the annotation record.

    Two fields of the response are deliberately not carried over. `receiptDate`
    is parsed deterministically by app/date_extract.py and is not the model's
    job, and `nameEn` is always null from the engine because translation is the
    app's job (RECEIPT_API.md).

    The vat row was already lifted out of items[] by the API's _split_tax, so a
    non-null taxAmount has to be put back as an item -- that is the shape the
    model is trained to emit and the shape reconcile() is written against.
    """
    items = [{"name": item.get("name"), "price": item.get("price"),
              "c": None, "s": None}                     # <- you fill c and s
             for item in draft.get("items") or []]

    tax = draft.get("taxAmount")
    if tax is not None:
        items.append({"name": "vat", "price": tax, "c": None, "s": None})

    target: dict = {"shop_name": draft.get("shopName"), "items": items}

    discount = draft.get("basketDiscount")
    warning = None
    if discount is not None:
        # apply_basket_discount has ALREADY spread this across the item prices
        # and the response reports it separately. Writing both here would count
        # it twice, and the pre-spread prices cannot be recovered exactly
        # (the spread rounds). The prices have to come off the photo.
        target["basket-wide_discount"] = discount
        warning = (f"basket discount {discount} was already spread across the "
                   f"item prices by the engine -- RE-READ every price off the "
                   f"photo, they are not as printed")

    target["total_price"] = draft.get("totalAmount")

    # One page per request today; join defensively in case a stitched response
    # was saved here by hand.
    pages = draft.get("ocrTexts") or []
    text = draft.get("stitchedText") or ("\n".join(pages) if pages else "")

    record = {
        "id": stem,
        "meta": {"kind": kind},
        "input": text,
        "target": target,
    }
    if warning:
        record["_warning"] = warning
    if draft.get("reconciles") is False:
        record["_engine_reconcile_status"] = draft.get("reconcileStatus")
    return record


def collect(out_dir: Path, dest: Path) -> None:
    """Concatenate finished skeletons into one JSONL, dropping the underscore
    bookkeeping fields the trainer has no use for."""
    paths = sorted(out_dir.glob("*.json"))
    if not paths:
        sys.exit(f"error: nothing to collect in {out_dir}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = unannotated = 0
    with dest.open("w", encoding="utf-8") as fh:
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            if any(item.get("c") is None and
                   str(item.get("name", "")).strip().lower() not in {"vat", "tax"}
                   for item in record.get("target", {}).get("items", [])):
                unannotated += 1
            clean = {k: v for k, v in record.items() if not k.startswith("_")}
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
            written += 1

    print(f"{written} record(s) -> {dest}")
    if unannotated:
        print(f"WARNING: {unannotated} record(s) still have items with no "
              f"category. Run validate_annotations.py --strict first.")


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.collect:
        collect(args.out, args.collect)
        return

    if not args.drafts.is_dir():
        sys.exit(f"error: no draft folder at {args.drafts}\n"
                 f"       run review_photos.py first (see RUN.md)")

    drafts = sorted(args.drafts.glob("*.json"))
    if not drafts:
        sys.exit(f"error: no .json drafts in {args.drafts}")

    args.out.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    suspect: list[str] = []
    discounted: list[str] = []

    for path in drafts:
        try:
            draft = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {path.name}: unreadable ({exc})")
            failed += 1
            continue

        if not draft.get("ok"):
            # No draft to correct. Needs doing from the photo by hand, and the
            # OCR text is still in the response to paste in as `input`.
            print(f"  ! {path.name}: engine returned ok=false -- annotate by hand")
            failed += 1
            continue

        dest = args.out / path.name
        if dest.exists() and not args.force:
            skipped += 1
            continue

        record = skeleton(draft, path.stem, args.kind)
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        written += 1
        if draft.get("reconciles") is False:
            suspect.append(path.stem)
        if draft.get("basketDiscount") is not None:
            discounted.append(path.stem)

    print(f"\n{written} skeleton(s) written to {args.out}")
    if skipped:
        print(f"{skipped} already existed and were left alone (--force to replace)")
    if failed:
        print(f"{failed} draft(s) could not be used")

    if suspect:
        print(f"\n{len(suspect)} receipt(s) the engine could not reconcile -- its "
              f"numbers do not add up, check these against the photo first:")
        for stem in suspect:
            print(f"  {stem}")

    if discounted:
        print(f"\n{len(discounted)} receipt(s) have a basket discount the engine "
              f"already spread across the item prices.\nRE-READ every price off "
              f"the photo for these -- they are not as printed:")
        for stem in discounted:
            print(f"  {stem}")

    print("\nNext: fill c and s on every item (ANNOTATION.md has the taxonomy), "
          "then validate_annotations.py")


if __name__ == "__main__":
    main()
