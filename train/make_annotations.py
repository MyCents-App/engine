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
    p.add_argument("--merge-llm", type=Path, default=None,
                   help="folder of LLM answers ({target, _flags}); splice each "
                        "onto the OCR text from --drafts and write to --out")
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


def ocr_text(draft: dict) -> str:
    """The text the model was actually prompted with, straight off the response."""
    pages = draft.get("ocrTexts") or []
    return draft.get("stitchedText") or ("\n".join(pages) if pages else "")


def _stem(value: str) -> str:
    """'photo-7.jpg' -> 'photo-7'. Tolerates a bare stem or a full path."""
    return Path(str(value).strip().replace("\\", "/")).stem


def load_llm_answers(path: Path) -> dict[str, dict]:
    """Read the labeller's answers, however they were handed over.

    Accepts, in order of how likely someone is to produce it:

      a folder     one <stem>.json per receipt
      one object   {"photo-1.jpg": {...}, "photo-2.jpg": {...}}
      one array    [{"id": "photo-1", "target": {...}}, ...]
      one JSONL    the same, one object per line

    Pairing needs a photo name somewhere, so for the array and JSONL forms
    each entry must carry one under `id`, `photo`, `file`, `filename`, `stem`
    or `receipt_id`. Requiring matching filenames was our storage detail
    leaking into somebody else's workflow; any of these is fine.
    """
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            sys.exit(f"error: no .json answers in {path}")
        out: dict[str, dict] = {}
        for file in files:
            try:
                out[file.stem] = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"  ! {file.name}: unreadable ({exc})")
        return out

    if not path.is_file():
        sys.exit(f"error: no such path: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        sys.exit(f"error: {path} is empty")

    entries: list | dict
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        entries = []
        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"error: {path}:{lineno}: {exc}")

    if isinstance(entries, dict):
        # Ambiguous: a one-line JSONL parses as a plain object, so an answer
        # for a single receipt looks exactly like a mapping of photo name ->
        # answer. Tell them apart by content -- an answer carries `target` or
        # `items`, a mapping does not -- otherwise the dict's own keys ("id",
        # "target") get read as photo names.
        if "target" in entries or "items" in entries:
            entries = [entries]
        else:
            bad = [k for k, v in entries.items() if not isinstance(v, dict)]
            if bad:
                sys.exit(f"error: {path}: keys {bad[:5]} do not map to objects; "
                         f"expected {{\"photo-1.jpg\": {{...}}, ...}}")
            return {_stem(key): value for key, value in entries.items()}

    out = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            sys.exit(f"error: {path}: entry {i} is not an object")
        for field in ("id", "photo", "file", "filename", "stem", "receipt_id"):
            if entry.get(field):
                out[_stem(entry[field])] = entry
                break
        else:
            sys.exit(f"error: {path}: entry {i} has no id/photo/file field, so "
                     f"there is no way to tell which photo it belongs to. "
                     f"Add one, or hand the answers over as an object keyed by "
                     f"photo filename.")
    return out


def merge_llm(llm_path: Path, drafts_dir: Path, out_dir: Path,
              kind: str, force: bool) -> None:
    """Splice an LLM's labels onto the exact OCR text from the engine's draft.

    The labeller is deliberately never asked to reproduce the OCR text. Large
    models are unreliable at echoing long noisy text verbatim -- they tidy
    <br> artifacts, normalise spacing, drop a duplicated line -- and any drift
    there trains the model on input the OCR engine does not produce. That
    corruption is invisible afterwards: the pair looks well-formed, and the
    model simply learns to expect text it will never be given.

    So `input` is taken from the draft, byte for byte, and only `target` comes
    from the labeller.
    """
    answers = load_llm_answers(llm_path)
    if not answers:
        sys.exit(f"error: no usable answers in {llm_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    flagged: list[str] = []
    unmatched: list[str] = []

    for stem, answer in sorted(answers.items()):
        draft_path = drafts_dir / f"{stem}.json"
        if not draft_path.exists():
            unmatched.append(stem)
            failed += 1
            continue

        try:
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {stem}: draft unreadable ({exc})")
            failed += 1
            continue

        # Accept either {"target": {...}} or a bare target, since a labeller
        # told to emit one object sometimes emits the inner one.
        target = answer.get("target", answer)
        if not isinstance(target, dict) or "items" not in target:
            print(f"  ! {stem}: no usable 'target' object")
            failed += 1
            continue

        text = ocr_text(draft)
        if not text.strip():
            print(f"  ! {stem}: the draft has no OCR text")
            failed += 1
            continue

        dest = out_dir / f"{stem}.json"
        if dest.exists() and not force:
            skipped += 1
            continue

        record = {"id": stem, "meta": {"kind": kind},
                  "input": text, "target": target}
        for field in ("_flags", "_needs_review"):
            if answer.get(field):
                record[field] = answer[field]
        if answer.get("_needs_review"):
            flagged.append(stem)

        dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        written += 1

    print(f"\n{written} record(s) written to {out_dir}")
    if skipped:
        print(f"{skipped} already existed and were left alone (--force to replace)")
    if failed:
        print(f"{failed} answer(s) could not be used")
    if unmatched:
        available = sorted(p.stem for p in drafts_dir.glob("*.json"))
        print(f"\n{len(unmatched)} answer(s) had no matching draft, so there is "
              f"no OCR text to pair them with:")
        for stem in unmatched[:10]:
            print(f"  {stem}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
        print(f"\nThe drafts available are named: "
              f"{', '.join(available[:6])}{' ...' if len(available) > 6 else ''}")
        print("Each answer is matched to a draft by photo name. Name them the "
              "same, or\nhand the answers over as one object keyed by photo "
              "filename.")
    # The reverse of `unmatched`: a photo that was OCR'd but never labelled.
    # Easy to lose one in a batch of 200, and it shows up as a quietly smaller
    # dataset rather than as an error.
    missing = sorted({p.stem for p in drafts_dir.glob("*.json")} - set(answers))
    if missing:
        print(f"\n{len(missing)} photo(s) were OCR'd but have no label:")
        for stem in missing[:10]:
            print(f"  {stem}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if flagged:
        print(f"\n{len(flagged)} record(s) the labeller flagged for review:")
        for stem in flagged:
            print(f"  {stem}")
    print("\nNext: validate_annotations.py, then --collect")


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

    if args.merge_llm:
        if not args.drafts.is_dir():
            sys.exit(f"error: no draft folder at {args.drafts} -- the OCR text "
                     f"comes from there, not from the labeller")
        merge_llm(args.merge_llm, args.drafts, args.out, args.kind, args.force)
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
