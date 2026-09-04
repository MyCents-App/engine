"""Scored evaluation of the review run against hand-written gold labels.

review_photos.py is the eyes-on pass -- it prints a receipt and lets you judge it. This is
the scored counterpart: it takes the JSON that run wrote to SuryaOCR/data/review/ and scores
it against SuryaOCR/receipts.json using eval_metrics, the same harness Phase 3 and Phase 5
used, so these numbers sit on the same axis as the earlier tables.

The scoring rule follows the stated objective: a PRICE is right only if it is exactly right,
a NAME only has to be recognisable. Concretely --

  * every money metric (total, per-line prices, item count) is exact-match, no tolerance;
  * item lines are paired on exact price equality, and name similarity is then REPORTED on
    those pairs rather than being a pass/fail gate;
  * shop name is scored three ways: exact, plain similarity, and "acceptable".

"Acceptable" for a shop name uses token_set_ratio rather than the plain ratio eval_metrics
reports, because the gold labels carry legal-entity and branch decoration the app does not
need -- "H&M" against "H&M (Mega Bangna, Branch No. 00013)" is a good answer, and a plain
edit-distance ratio scores it 16/100 purely for being shorter. token_set_ratio scores a
prediction whose tokens are a subset of the gold's at 100, while still rejecting a wrong
merchant. Item names stay on the plain ratio: most are Thai, which does not word-segment on
spaces, so token-based matching would be meaningless there.

    .venv\\Scripts\\python.exe eval_review.py                    # score + write reports
    .venv\\Scripts\\python.exe eval_review.py --threshold 70     # stricter name bar
    .venv\\Scripts\\python.exe eval_review.py --no-write         # console only
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rapidfuzz import fuzz

import eval_metrics
from eval_metrics import DEFAULT_NAME_SIM_THRESHOLD, aggregate, score_example
from postprocess import parse_price_strict

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLD = ROOT.parent / "SuryaOCR" / "receipts.json"
DEFAULT_REVIEW = ROOT.parent / "SuryaOCR" / "data" / "review"
DEFAULT_REPORTS = ROOT / "reports" / "review_eval"

RULE = "=" * 100
THIN = "-" * 100


def natural_key(path: Path):
    """photo-2 before photo-10 -- plain sorting puts photo-10 second."""
    digits = "".join(c for c in path.stem if c.isdigit())
    return (int(digits) if digits else 0, path.stem)


def _norm(s) -> str:
    return " ".join(str(s or "").strip().lower().split())


# --------------------------------------------------------------------------------------
# Schema adapters: the API speaks camelCase, the eval harness speaks the task.md contract
# --------------------------------------------------------------------------------------

def to_contract(obj: dict) -> dict:
    """{shopName, totalAmount, items[{name, price}], basketDiscount} -> the task.md contract.

    Only renames keys; no value is cleaned up or reformatted, so a malformed price stays
    malformed and is scored as one.
    """
    out = {
        "shop_name": obj.get("shopName") or "",
        "items": [
            {"name": it.get("name", ""), "price": it.get("price")}
            for it in (obj.get("items") or [])
            if isinstance(it, dict)
        ],
        "total_price": obj.get("totalAmount"),
    }
    if obj.get("basketDiscount") not in (None, ""):
        out["basket-wide_discount"] = obj["basketDiscount"]
    return out


def pair_items_by_price(gold_items: list[dict], pred_items: list[dict]):
    """The same greedy exact-price pairing eval_metrics._name_quality uses, but returning the
    pairs so the report can show gold and predicted names side by side. Returns
    (pairs, unmatched_gold, unmatched_pred)."""
    remaining = list(enumerate(pred_items))
    pairs, unmatched_gold = [], []
    for g in gold_items:
        g_price = parse_price_strict(g.get("price"))
        cands = [
            (i, p) for i, p in remaining
            if g_price is not None
            and (pv := parse_price_strict(p.get("price"))) is not None
            and abs(pv - g_price) <= 1e-6
        ]
        if not cands:
            unmatched_gold.append(g)
            continue
        g_name = _norm(g.get("name"))
        best_i, best_p, best_sim = None, None, -1.0
        for i, p in cands:
            sim = fuzz.ratio(g_name, _norm(p.get("name")))
            if sim > best_sim:
                best_i, best_p, best_sim = i, p, sim
        pairs.append((g, best_p, best_sim))
        remaining = [(i, p) for i, p in remaining if i != best_i]
    return pairs, unmatched_gold, [p for _, p in remaining]


SCORECARD_COLUMNS = [
    "Receipt", "Total amount exact", "Line price accuracy", "Item names acceptable",
    "Item count exact", "Exact on money", "Shop name acceptable", "Schema-valid JSON",
]

NAME_METHOD_NOTE = """\
### How "item names acceptable" is calculated

Two names are compared character by character, and the overlap is expressed as a percentage:

    score = 100 x 2 x (characters the two names share) / (length of both names added together)

`Toast` vs `Roast` share o, a, s, t -- 4 characters out of 5 + 5 -- so they score
(2 x 4) / 10 = 80%. A name counts as **acceptable at {threshold:.0f}% or above**.

Two rules matter as much as the formula:

1. **A name is only scored if its price already matched.** Gold and predicted lines are paired
   on the exact same price first; names are then compared within those pairs. A line whose price
   is wrong is counted as a price error and is not scored again as a name error.
2. **Shop names are compared word by word instead**, because the gold labels carry branch and
   legal-entity decoration the app does not need. Predicting "H&M" for
   "H&M (Mega Bangna, Branch No. 00013)" scores 16% character by character purely for being
   shorter, but 100% word by word, since every word it gave is in the gold name. Item names
   cannot use this rule: most are Thai, which puts no spaces between words.
"""


def scorecard_rows(rows: list[dict], threshold: float) -> list[list[str]]:
    """The seven headline metrics, one row per receipt, plus a total row. Percentages are
    formatted for reading rather than for further arithmetic -- per_receipt.csv holds the
    raw values."""

    def frac(num: int, den: int) -> str:
        if not den:
            return "n/a"
        return f"{num}/{den} ({num / den:.0%})"

    out = []
    for r in rows:
        out.append([
            r["photo"],
            "Yes" if r["total_exact"] else "No",
            frac(r["n_matched_lines"], r["n_gold_items"]),
            frac(r["n_names_ok"], r["n_matched_lines"]),
            "Yes" if r["count_exact"] else "No",
            "Yes" if r["money_exact"] else "No",
            "Exact" if r["shop_exact"] else ("Yes" if r["shop_acceptable"] else "No"),
            "Valid" if r["schema_valid"] else "Invalid",
        ])
    n = len(rows)
    out.append([
        f"All {n}",
        frac(sum(r["total_exact"] for r in rows), n),
        frac(sum(r["n_matched_lines"] for r in rows), sum(r["n_gold_items"] for r in rows)),
        frac(sum(r["n_names_ok"] for r in rows), sum(r["n_matched_lines"] for r in rows)),
        frac(sum(r["count_exact"] for r in rows), n),
        frac(sum(r["money_exact"] for r in rows), n),
        frac(sum(r["shop_acceptable"] for r in rows), n),
        frac(sum(r["schema_valid"] for r in rows), n),
    ])
    return out


def scorecard_markdown(rows: list[dict], threshold: float) -> str:
    body = scorecard_rows(rows, threshold)
    total_row = body[-1]
    lines = [
        "# Per-receipt scorecard",
        "",
        f"{len(rows)} receipts. Money is scored exact-match -- a price is either the printed "
        "number or it is wrong. Names are scored on how closely they resemble the label.",
        "",
        "| " + " | ".join(SCORECARD_COLUMNS) + " |",
        "|" + "---|" * len(SCORECARD_COLUMNS),
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body[:-1]]
    lines += ["| **" + "** | **".join(total_row) + "** |", ""]
    lines += [NAME_METHOD_NOTE.format(threshold=threshold), ""]
    lines += [
        "### What each column means", "",
        "- **Total amount exact** -- the receipt's grand total matches the label exactly.",
        "- **Line price accuracy** -- of the labelled item lines, how many the pipeline "
        "produced at exactly the right price.",
        "- **Item names acceptable** -- of those price-matched lines, how many names cleared "
        f"the {threshold:.0f}% bar.",
        "- **Item count exact** -- the pipeline returned the same number of lines as the label.",
        "- **Exact on money** -- every line price *and* the total right, with no extra or "
        "missing line. The strictest column.",
        "- **Shop name acceptable** -- the merchant name is the right merchant, allowing for "
        "dropped branch and legal-entity decoration.",
        "- **Schema-valid JSON** -- the output parsed and obeyed the field contract.",
        "",
    ]
    return "\n".join(lines)


def load_pairs(gold_path: Path, review_dir: Path):
    """(photo_id, gold, prediction) in photo order. Gold is a flat list, so entry i is the
    label for the i-th photo -- the same order review_photos.py walked."""
    gold_list = json.loads(gold_path.read_text(encoding="utf-8"))
    review_files = sorted(review_dir.glob("*.json"), key=natural_key)
    if len(gold_list) != len(review_files):
        raise SystemExit(
            f"{len(gold_list)} gold receipts in {gold_path.name} but {len(review_files)} "
            f"review files in {review_dir} -- they must line up one-to-one, in photo order."
        )
    return [
        (f.stem, gold, json.loads(f.read_text(encoding="utf-8")))
        for gold, f in zip(gold_list, review_files)
    ]


# --------------------------------------------------------------------------------------
# Scoring one review run
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_NAME_SIM_THRESHOLD,
                    help="name similarity 0-100 at or above which a name counts as acceptable")
    ap.add_argument("--no-write", action="store_true", help="print only, write no report files")
    args = ap.parse_args()

    pairs = load_pairs(args.gold, args.review)

    scores, rows, item_rows = [], [], []
    for photo_id, gold_raw, pred_raw in pairs:
        gold = to_contract(gold_raw)
        pred = to_contract(pred_raw)
        ocr_text = (pred_raw.get("ocrTexts") or [""])[0]
        engine = pred_raw.get("engine") or {}

        s = score_example(
            json.dumps(pred, ensure_ascii=False),
            gold,
            ocr_text,
            name_sim_threshold=args.threshold,
            latency_s=engine.get("total_seconds"),
        )
        scores.append(s)

        item_pairs, missed_gold, spurious = pair_items_by_price(gold["items"], pred["items"])
        shop_brand_sim = fuzz.token_set_ratio(_norm(gold["shop_name"]), _norm(pred["shop_name"]))
        rows.append({
            "photo": photo_id,
            "ok": bool(pred_raw.get("ok", False)),
            "gold_shop": gold["shop_name"],
            "pred_shop": pred["shop_name"],
            "shop_sim": round(s.shop_name_similarity, 1),
            "shop_brand_sim": round(shop_brand_sim, 1),
            "shop_exact": s.shop_name_exact,
            "shop_acceptable": shop_brand_sim >= args.threshold,
            "gold_total": gold["total_price"],
            "pred_total": pred["total_price"],
            "total_exact": s.total_price_exact,
            "n_gold_items": len(gold["items"]),
            "n_pred_items": len(pred["items"]),
            "count_exact": s.item_count_exact,
            "prices_exact": s.price_multiset_exact,
            "money_exact": s.money_exact,
            "price_recall": round(s.item_price_only_recall, 3),
            "name_sim": round(s.name_quality_mean, 1),
            "name_ok_rate": round(s.name_acceptable_rate, 3),
            "n_missed": len(missed_gold),
            "n_spurious": len(spurious),
            "n_matched_lines": len(item_pairs),
            "n_names_ok": sum(1 for _, _, sim in item_pairs if sim >= args.threshold),
            "schema_valid": s.schema_valid,
            "recon": s.reconciliation_status,
            "latency_s": engine.get("total_seconds"),
        })
        for g, p, sim in item_pairs:
            item_rows.append({"photo": photo_id, "status": "price match", "price": g.get("price"),
                              "gold_name": g.get("name"), "pred_name": p.get("name"),
                              "name_sim": round(sim, 1), "name_ok": sim >= args.threshold})
        for g in missed_gold:
            item_rows.append({"photo": photo_id, "status": "MISSED (no pred line at this price)",
                              "price": g.get("price"), "gold_name": g.get("name"),
                              "pred_name": "", "name_sim": "", "name_ok": ""})
        for p in spurious:
            item_rows.append({"photo": photo_id, "status": "SPURIOUS (price not in gold)",
                              "price": p.get("price"), "gold_name": "",
                              "pred_name": p.get("name"), "name_sim": "", "name_ok": ""})

    agg = aggregate(scores)
    agg["shop_name_acceptable_rate"] = sum(r["shop_acceptable"] for r in rows) / len(rows)
    agg["shop_name_brand_sim_mean"] = sum(r["shop_brand_sim"] for r in rows) / len(rows)

    # ---- console ---------------------------------------------------------------------
    print(RULE)
    print(f"Review-run evaluation  |  {len(rows)} receipts  |  "
          f"name-acceptable threshold = {args.threshold:.0f}/100")
    print(f"gold: {args.gold}")
    print(f"pred: {args.review}")
    print(RULE)
    print(f"{'photo':<9} {'MONEY':<6} {'total':<6} {'count':<6} {'prices':<7} "
          f"{'g/p':<7} {'nameSim':<8} {'shop':<6} {'recon':<18} shop name (gold -> pred)")
    print(THIN)
    for r in rows:
        mark = lambda b: " OK " if b else "MISS"  # noqa: E731
        shop = "exact" if r["shop_exact"] else ("ok" if r["shop_acceptable"] else "BAD")
        name_sim = f"{r['name_sim']:.0f}" if r["n_gold_items"] else "-"
        shop_txt = r["gold_shop"] if r["shop_exact"] else f"{r['gold_shop']}  ->  {r['pred_shop']}"
        print(f"{r['photo']:<9} {mark(r['money_exact']):<6} {mark(r['total_exact']):<6} "
              f"{mark(r['count_exact']):<6} {mark(r['prices_exact']):<7} "
              f"{str(r['n_gold_items']) + '/' + str(r['n_pred_items']):<7} {name_sim:<8} "
              f"{shop:<6} {r['recon']:<18} {shop_txt}")
    print(THIN)

    def pct(k):
        v = agg.get(k)
        return "-" if v is None else f"{v:.1%}"

    print("\nHEADLINE -- money (exact match, no tolerance)")
    print(f"  Money fully exact (every line price + total)   {pct('money_exact_rate')}")
    print(f"  Total amount exact                             {pct('total_price_exact_rate')}")
    print(f"  All line prices exact (multiset)               {pct('price_multiset_exact_rate')}")
    print(f"  Item count exact                               {pct('item_count_exact_rate')}")
    print(f"  Per-line price recall (names ignored)          {pct('item_price_only_recall_mean')}")
    print("\nNAMES -- acceptable, not exact")
    print(f"  Item name similarity on price-matched lines    {agg['name_quality_mean']:.1f}/100")
    print(f"  Item names acceptable (>= {args.threshold:.0f})                 {pct('name_acceptable_rate')}")
    print(f"  Shop name acceptable (>= {args.threshold:.0f}, token_set)      {pct('shop_name_acceptable_rate')}")
    print(f"  Shop brand similarity (token_set)              {agg['shop_name_brand_sim_mean']:.1f}/100")
    print(f"  Shop name plain similarity (ratio)             {agg['shop_name_similarity_mean']:.1f}/100")
    print(f"  Shop name exact (reference only)               {pct('shop_name_exact_rate')}")
    print("\nOUTPUT VALIDITY & SPEED")
    print(f"  JSON valid / schema valid                      {pct('json_validity_rate')} / {pct('schema_valid_rate')}")
    print(f"  Self-reconciles (lines vs total)               {pct('reconciliation_pass_rate')}")
    print(f"  Reconcile breakdown                            {agg['reconciliation_status_breakdown']}")
    lat = agg.get("mean_latency_s")
    print(f"  Mean end-to-end latency                        {lat:.2f}s" if lat
          else "  Mean end-to-end latency                        -")
    print("\nName-coupled item metrics (stricter: price AND name must both agree)")
    print(f"  Item precision / recall / F1                   "
          f"{pct('item_precision_mean')} / {pct('item_recall_mean')} / {pct('item_f1_mean')}")
    print(f"  Whole-record exact match                       {pct('exact_match_rate')}")

    bad = [r for r in rows if not r["money_exact"]]
    if bad:
        print(f"\n{THIN}\nReceipts that missed on money ({len(bad)}/{len(rows)}) "
              f"-- these are the ones to look at")
        for r in bad:
            why = []
            if not r["total_exact"]:
                why.append(f"total {r['gold_total']} -> {r['pred_total']}")
            if not r["count_exact"]:
                why.append(f"{r['n_gold_items']} gold lines -> {r['n_pred_items']} predicted")
            if r["n_missed"]:
                why.append(f"{r['n_missed']} gold price(s) not produced")
            if r["n_spurious"]:
                why.append(f"{r['n_spurious']} predicted price(s) not in gold")
            print(f"  {r['photo']:<9} {'; '.join(why) or 'price mismatch'}")

    shop_diff = [r for r in rows if not r["shop_exact"]]
    if shop_diff:
        print(f"\n{THIN}\nShop names that differ from gold ({len(shop_diff)}/{len(rows)}) "
              f"-- 'ok' means the prediction is a subset/variant of the gold name")
        for r in shop_diff:
            verdict = "ok " if r["shop_acceptable"] else "BAD"
            print(f"  {r['photo']:<9} {verdict}  ratio {r['shop_sim']:>5}  brand {r['shop_brand_sim']:>5}  "
                  f"{r['gold_shop']}  ->  {r['pred_shop']}")

    weak = [i for i in item_rows if i["status"] == "price match" and i["name_ok"] is False]
    if weak:
        print(f"\n{THIN}\nRight price, questionable name ({len(weak)} lines) -- judge these yourself")
        for i in weak:
            print(f"  {i['photo']:<9} {i['price']:>8}  sim {i['name_sim']:>5}  "
                  f"{i['gold_name']}  ->  {i['pred_name']}")

    print(f"\n{THIN}\nPer-receipt scorecard")
    card = scorecard_rows(rows, args.threshold)
    widths = [max(len(SCORECARD_COLUMNS[i]), max(len(r[i]) for r in card))
              for i in range(len(SCORECARD_COLUMNS))]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(SCORECARD_COLUMNS, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in card[:-1]:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    print("  " + "  ".join(c.ljust(w) for c, w in zip(card[-1], widths)))

    # ---- files -----------------------------------------------------------------------
    if not args.no_write:
        args.reports.mkdir(parents=True, exist_ok=True)
        (args.reports / "scorecard.md").write_text(
            scorecard_markdown(rows, args.threshold), encoding="utf-8")
        with open(args.reports / "scorecard.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(SCORECARD_COLUMNS)
            w.writerows(card)
        eval_metrics.write_csv({"review-run": agg}, args.reports / "metrics.csv")
        eval_metrics.write_examples_jsonl(args.reports / "examples.jsonl", scores)
        # utf-8-sig so Excel opens the Thai item names correctly
        for name, data in (("per_receipt.csv", rows), ("per_item.csv", item_rows)):
            with open(args.reports / name, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
        (args.reports / "aggregate.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.reports / "summary.md").write_text(
            "# Review-run evaluation\n\n"
            f"{len(rows)} receipts, name-acceptable threshold {args.threshold:.0f}/100. "
            "Money is scored exact-match; names are scored on similarity.\n\n"
            + eval_metrics.format_comparison_table({"review-run": agg}) + "\n",
            encoding="utf-8")
        print(f"\nWrote {args.reports}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
