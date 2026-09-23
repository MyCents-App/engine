"""Build the joint training set: real receipts + categorized synthetic ones.

    .venv\\Scripts\\python.exe build_dataset.py

Inputs (all under data/, which is gitignored):

  annotations/*.json          the real receipts, {id, meta, input, target} --
                              photo-labelled targets on our own Surya text
  synthetic_source.jsonl      the 3,200 synthetic OCR/JSON pairs the extraction
                              checkpoint was trained on, {input, target}, no
                              categories
  synth_work/pair_labels.json {"<shop>\\t<item name>": [c, s]} -- a category
                              for every distinct (shop, item) in the synthetic
                              source

Outputs:

  train_sft.jsonl   messages for train_qlora.py: real-train (x --real-repeat)
                    + synthetic
  val_sft.jsonl     messages for train_qlora.py: real-validation ONLY
  train.jsonl       the training records in {id, meta, input, target} form,
                    each once -- the naming data_prep.py used for extraction
  val.jsonl         the validation records, same form -- what the evaluator reads
  split_report.md   what went where, and why

Why categories are labelled per (shop, item) and not per receipt: a category
is a function of the item AND the shop (ice is Food & Dining at a restaurant
and Groceries at 7-Eleven), and the synthetic generator reused its catalog, so
15,904 item lines are only 1,653 distinct pairs. Labelling the pairs once and
joining them on makes the same product at the same shop always carry the same
answer -- per-receipt labelling would have taught the model noise.

Why the validation set is real-only and split by a hash of `id`: see
ANNOTATION.md section 5. Hashing a stable id keeps a receipt on the same side
of the split across rebuilds, added receipts and fixed labels; splitting by
position does not, and a validation receipt that leaks into training inflates
every number with nothing to flag it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import validate_annotations as va
from prompts import build_messages

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# One spelling per shop across both sets. Without this the model sees
# "CP ALL, 7-Eleven" on 600 synthetic receipts and "7-Eleven" on 79 real ones,
# learns both, and is scored wrong on whichever it did not pick.
SHOP_ALIASES = {
    "CP ALL, 7-Eleven": "7-Eleven",
    "7-ELEVEN": "7-Eleven",
    "watsons": "Watsons",
    "Bonchon": "BONCHON CHICKEN",
    "CP AXTRA": "CP Axtra PCL",
    "BIGC RAJDUMRI": "BIG C SUPERCENTER",
    "THE PIZZA COMPANY": "The Pizza Company",
    "BOOTS HEALTH & BEAUTY": "Boots",
    # Same restaurant, same phone number on every receipt; the one header OCR
    # read cleanly prints ม่าม่า.
    "麻麻 - MAMA หม่าล่า": "麻麻 - MAMA ม่าม่า",
}

TAX_NAMES = {"vat", "tax"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, default=DATA / "annotations")
    p.add_argument("--synthetic-source", type=Path, default=DATA / "synthetic_source.jsonl")
    p.add_argument("--pair-labels", type=Path,
                   default=DATA / "synth_work" / "pair_labels.json")
    p.add_argument("--out", type=Path, default=DATA)
    p.add_argument("--synthetic", type=int, default=1000,
                   help="synthetic receipts to keep (default: %(default)s)")
    p.add_argument("--val-share", type=float, default=0.37,
                   help="share of real receipts held out for validation; 0.37 of "
                        "188 is ~70, ANNOTATION.md section 5 (default: %(default)s)")
    p.add_argument("--discount-share", type=float, default=0.10,
                   help="cap on synthetic receipts carrying a basket discount "
                        "(default: %(default)s)")
    p.add_argument("--real-repeat", type=int, default=2,
                   help="times each real TRAINING receipt appears in train_sft.jsonl. "
                        "ANNOTATION.md planned ~17%% real items from ~130 receipts; "
                        "115 remain after cleaning, which is ~6%% at 1x. 2x is that "
                        "document's own next step (default: %(default)s)")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

def canonical(target: dict) -> dict:
    """Keys in the order the prompt's schema line gives them, shop aliased."""
    shop = target.get("shop_name")
    out = {"shop_name": SHOP_ALIASES.get(shop, shop),
           "items": [{"name": i["name"], "price": i["price"],
                      "c": i.get("c"), "s": i.get("s")} for i in target["items"]]}
    if "basket-wide_discount" in target:
        out["basket-wide_discount"] = target["basket-wide_discount"]
    out["total_price"] = target["total_price"]
    return out


def load_real(folder: Path) -> list[dict]:
    records = []
    for path in sorted(folder.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records.append({"id": raw["id"], "meta": {"kind": "real"},
                        "input": raw["input"], "target": canonical(raw["target"])})
    return records


def load_synthetic(path: Path, labels: dict) -> tuple[list[dict], list[str]]:
    """Attach c/s to every synthetic item. Returns (records, rejects)."""
    records, rejects = [], []
    for n, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        rid = f"syn-{n:04d}"
        target = raw["target"]
        shop = target["shop_name"]
        items = []
        for item in target["items"]:
            if str(item["name"]).strip().lower() in TAX_NAMES:
                c, s = None, None
            else:
                key = f"{shop}\t{item['name']}"
                if key not in labels:
                    sys.exit(f"error: no label for {key!r} ({rid}) -- rebuild pair_labels.json")
                c, s = labels[key]
            items.append({**item, "c": c, "s": s})
        record = {"id": rid, "meta": {"kind": "synthetic"}, "input": raw["input"],
                  "target": canonical({**target, "items": items})}
        errors, _, _ = va.check(record, strict=False)
        if errors:
            rejects.append(f"{rid}: {errors[0][:90]}")
            continue
        records.append(record)
    return records, rejects


# --------------------------------------------------------------------------
# Split and selection
# --------------------------------------------------------------------------

def is_validation(rid: str, share: float) -> bool:
    bucket = int(hashlib.sha1(rid.encode("utf-8")).hexdigest(), 16) % 10_000
    return bucket < share * 10_000


def pairs_of(record: dict) -> set[tuple[str, str]]:
    shop = record["target"]["shop_name"]
    return {(shop, i["name"]) for i in record["target"]["items"]
            if i["name"].lower() not in TAX_NAMES}


def select_synthetic(records: list[dict], n: int, discount_share: float,
                     seed: int) -> list[dict]:
    """Pick n receipts: every shop represented, big chains damped, the whole
    (shop, item) catalog covered, long receipts favoured, discounts capped.

    Shop quotas are proportional to sqrt(count), so 7-Eleven's 600 of 3,200
    becomes ~11% instead of ~19% while a 10-receipt shop still gets a few.
    Within a shop, receipts are taken greedily by how many (shop, item) pairs
    they add that nothing chosen so far covers -- the first run memorised a
    resampled catalog, so repetition is the thing to spend the budget against
    -- with ties going to longer receipts, which are where the model miscounts.
    """
    rng = random.Random(seed)
    by_shop: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_shop[r["target"]["shop_name"]].append(r)

    weights = {s: math.sqrt(len(rs)) for s, rs in by_shop.items()}
    total_w = sum(weights.values())
    quota = {s: min(len(by_shop[s]), max(1, round(n * w / total_w)))
             for s, w in weights.items()}
    # Nudge the rounding onto exactly n, largest shops absorbing the change.
    order = sorted(by_shop, key=lambda s: -len(by_shop[s]))
    while sum(quota.values()) != n:
        step = 1 if sum(quota.values()) < n else -1
        for s in order:
            if sum(quota.values()) == n:
                break
            if 1 <= quota[s] + step <= len(by_shop[s]):
                quota[s] += step

    max_discount = int(n * discount_share)
    covered: set = set()
    chosen: list[dict] = []
    discounts = 0
    for shop in order:
        pool = by_shop[shop][:]
        rng.shuffle(pool)
        for _ in range(quota[shop]):
            best, best_key = None, None
            for r in pool:
                has_discount = "basket-wide_discount" in r["target"]
                if has_discount and discounts >= max_discount:
                    continue
                key = (len(pairs_of(r) - covered), len(r["target"]["items"]))
                if best_key is None or key > best_key:
                    best, best_key = r, key
            if best is None:
                break
            pool.remove(best)
            chosen.append(best)
            covered |= pairs_of(best)
            discounts += "basket-wide_discount" in best["target"]

    # A shop whose remaining receipts all carry a discount stops short once
    # the cap is reached; hand its unused slots to the rest, same greedy key.
    taken = {id(r) for r in chosen}
    leftover = [r for r in records if id(r) not in taken
                and "basket-wide_discount" not in r["target"]]
    rng.shuffle(leftover)
    while len(chosen) < n and leftover:
        best = max(leftover, key=lambda r: (len(pairs_of(r) - covered),
                                            len(r["target"]["items"])))
        leftover.remove(best)
        chosen.append(best)
        covered |= pairs_of(best)
    return chosen


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_messages(record: dict) -> dict:
    return {"id": record["id"], "meta": record["meta"],
            "messages": build_messages(record["input"], record["target"])}


def describe(records: list[dict]) -> dict:
    items = [i for r in records for i in r["target"]["items"]]
    goods = [i for i in items if i["name"].lower() not in TAX_NAMES]
    cats = Counter(i["c"] for i in goods)
    return {
        "receipts": len(records),
        "items": len(items),
        "vat_rows": len(items) - len(goods),
        "discount_receipts": sum("basket-wide_discount" in r["target"] for r in records),
        "long_10plus": sum(len(r["target"]["items"]) >= 10 for r in records),
        "c_null": cats.get(None, 0),
        "s_null_given_c": sum(1 for i in goods if i["c"] and i["s"] is None),
        "categories": {c: cats.get(c, 0) for c in va.CATEGORIES},
        "shops": len({r["target"]["shop_name"] for r in records}),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    labels = {k: tuple(v) for k, v in
              json.loads(args.pair_labels.read_text(encoding="utf-8")).items()}

    real = load_real(args.annotations)
    for record in real:
        errors, _, _ = va.check(record, strict=False)
        if errors:
            sys.exit(f"error: real record {record['id']} fails validation: {errors[0]}"
                     f"\n       run validate_annotations.py and fix it first")
    real_val = [r for r in real if is_validation(r["id"], args.val_share)]
    real_train = [r for r in real if not is_validation(r["id"], args.val_share)]

    synthetic_all, rejects = load_synthetic(args.synthetic_source, labels)
    synthetic = select_synthetic(synthetic_all, args.synthetic,
                                 args.discount_share, args.seed)

    train = real_train * args.real_repeat + synthetic
    random.Random(args.seed).shuffle(train)

    out = args.out
    write_jsonl(out / "train.jsonl", real_train + synthetic)
    write_jsonl(out / "val.jsonl", real_val)
    write_jsonl(out / "train_sft.jsonl", [as_messages(r) for r in train])
    write_jsonl(out / "val_sft.jsonl", [as_messages(r) for r in real_val])

    stats = {"real train": describe(real_train), "real val": describe(real_val),
             "synthetic": describe(synthetic)}
    all_pairs = set().union(*(pairs_of(r) for r in synthetic_all))
    kept_pairs = set().union(*(pairs_of(r) for r in synthetic))

    lines = ["# Joint dataset split", "",
             f"Built by `build_dataset.py` (seed {args.seed}). Validation is real "
             f"receipts only, chosen by a hash of `id`.", "",
             "| | " + " | ".join(stats) + " |",
             "|---|" + "---|" * len(stats)]
    for field in ["receipts", "items", "vat_rows", "discount_receipts",
                  "long_10plus", "shops", "c_null", "s_null_given_c"]:
        lines.append(f"| {field} | " + " | ".join(str(s[field]) for s in stats.values()) + " |")
    for category in va.CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(
            str(s["categories"][category]) for s in stats.values()) + " |")
    lines += ["",
              f"Synthetic source: {len(synthetic_all) + len(rejects)} records, "
              f"{len(rejects)} rejected by the validator, {len(synthetic)} selected. "
              f"The selection covers {len(kept_pairs)} of {len(all_pairs)} distinct "
              f"(shop, item) pairs.", "",
              f"train_sft.jsonl: {len(train)} records "
              f"({len(real_train)} real x{args.real_repeat} + {len(synthetic)} "
              f"synthetic; real is {len(real_train) * args.real_repeat / len(train):.0%} "
              f"of records and {stats['real train']['items'] * args.real_repeat / (stats['real train']['items'] * args.real_repeat + stats['synthetic']['items']):.0%} "
              f"of items). "
              f"val_sft.jsonl: {len(real_val)} records, all real.", ""]
    if rejects:
        lines += ["Rejected synthetic records:", ""] + [f"- {r}" for r in rejects] + [""]
    (out / "split_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
