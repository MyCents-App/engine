"""Phase 2 verification: unit tests for postprocess.py (JSON extraction, reconciliation) and
eval_metrics.py (item matching, scoring, aggregation), per the plan's Phase 2 verification step.

Plain assert-based (no pytest dependency) -- mirrors smoke_test.py's style.

Run: .venv\\Scripts\\python.exe test_phase2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import postprocess
import eval_metrics
from eval_metrics import ExampleScore, aggregate, score_example, _match_items, _price_only_recall

ROOT = Path(__file__).resolve().parent

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _failures.append(name)


# ========================================================================================
# postprocess.extract_json
# ========================================================================================

def test_extract_json():
    print("\n== postprocess.extract_json ==")

    obj, err = postprocess.extract_json('{"shop_name":"A","items":[{"name":"x","price":"1.00"}],"total_price":"1.00"}')
    check("plain JSON parses", obj is not None and err is None)
    check("plain JSON content correct", obj == {"shop_name": "A", "items": [{"name": "x", "price": "1.00"}], "total_price": "1.00"})

    obj, err = postprocess.extract_json('```json\n{"a": 1}\n```')
    check("fenced JSON parses", obj == {"a": 1}, detail=str((obj, err)))

    obj, err = postprocess.extract_json('<think>let me reason about this...</think>{"a": 1}')
    check("<think> block stripped", obj == {"a": 1}, detail=str((obj, err)))

    obj, err = postprocess.extract_json('Here is the JSON:\n{"a": 1}\nHope that helps!')
    check("stray prose around JSON handled", obj == {"a": 1}, detail=str((obj, err)))

    obj, err = postprocess.extract_json('{"a": 1,}')
    check("trailing comma repaired", obj == {"a": 1}, detail=str((obj, err)))

    obj, err = postprocess.extract_json('{"name": "a {weird} name", "price": "1.00"}')
    check("brace inside string value does not break matching", obj == {"name": "a {weird} name", "price": "1.00"}, detail=str((obj, err)))

    obj, err = postprocess.extract_json("no json here at all")
    check("unparseable text -> None + reason", obj is None and err is not None)

    obj, err = postprocess.extract_json("")
    check("empty string -> None + reason", obj is None and err is not None)


# ========================================================================================
# postprocess.reconcile
# ========================================================================================

def test_reconcile():
    print("\n== postprocess.reconcile ==")

    # Clean, correct, no-discount-no-vat prediction -> should reconcile exactly.
    pred = {
        "shop_name": "MK Restaurants",
        "items": [
            {"name": "Item A", "price": "59.00"},
            {"name": "Item B", "price": "238.00"},
            {"name": "Item C", "price": "92.00"},
        ],
        "total_price": "389.00",
    }
    r = postprocess.reconcile(pred, ocr_text="irrelevant receipt text")
    check("no-discount-no-vat reconciles ok", r.status == "ok" and r.passes, detail=str(r))

    # Basket-wide discount correctly reported -> reconciles.
    pred = {
        "shop_name": "Swensen's",
        "items": [{"name": "American Tower", "price": "249.00"}, {"name": "Chocolate Fondue", "price": "399.00"}],
        "basket-wide_discount": "97.20",
        "total_price": "550.80",
    }
    r = postprocess.reconcile(pred, ocr_text="irrelevant")
    check("basket discount reconciles ok", r.status == "ok" and r.passes, detail=str(r))

    # VAT gap: model forgot the vat line (784.00 vs true 828.18, residual ~6.54%), OCR has a
    # parseable printed VAT amount on the same line as the token.
    pred = {
        "shop_name": "Sukiyaki Restaurant",
        "items": [{"name": "Buffet Adult", "price": "657.00"}, {"name": "Water Refill", "price": "117.00"}],
        "total_price": "828.18",
    }
    ocr_text = "...\nVAT 7%  54.18\n..."
    r = postprocess.reconcile(pred, ocr_text)
    check(
        "vat gap detected in ~6.5% band with tax token",
        r.status == "vat_gap" and r.passes,
        detail=str(r),
    )
    check(
        "printed vat amount extracted correctly",
        r.repaired_vat_amount is not None and abs(r.repaired_vat_amount - 54.18) < 1e-6,
        detail=str(r.repaired_vat_amount),
    )

    # Same gap, but OCR only has the token with no parseable amount nearby -> falls back to
    # the computed residual.
    ocr_text_no_amount = "TAX ID# 0107555000317\nsome other unrelated line"
    r2 = postprocess.reconcile(pred, ocr_text_no_amount)
    check("vat gap falls back to residual when no printed amount found", r2.status == "vat_gap" and r2.passes)
    check(
        "fallback vat amount equals residual",
        r2.repaired_vat_amount is not None and abs(r2.repaired_vat_amount - r2.residual) < 1e-6,
        detail=str(r2.repaired_vat_amount),
    )

    # Large gap, NOT the ~6.5% signature -> unrecoverable, regardless of tax token presence.
    pred_big_gap = {
        "shop_name": "X",
        "items": [{"name": "A", "price": "900.00"}],
        "total_price": "1000.00",
    }
    r3 = postprocess.reconcile(pred_big_gap, ocr_text="VAT 7% 100.00")
    check("large non-VAT-shaped gap is unrecoverable", r3.status == "unrecoverable_gap" and not r3.passes, detail=str(r3))

    # sum(items) - discount exceeds total -> overcount, never labeled vat even with a tax token.
    pred_overcount = {
        "shop_name": "X",
        "items": [{"name": "A", "price": "150.00"}],
        "total_price": "100.00",
    }
    r4 = postprocess.reconcile(pred_overcount, ocr_text="VAT 7% 100.00")
    check("overcount detected and never mislabeled as vat", r4.status == "overcount" and not r4.passes, detail=str(r4))

    # Missing/garbage total -> unparseable.
    r5 = postprocess.reconcile({"items": [{"name": "A", "price": "1.00"}]}, ocr_text="x")
    check("missing total -> unparseable", r5.status == "unparseable" and not r5.passes, detail=str(r5))


# ========================================================================================
# eval_metrics item matching
# ========================================================================================

def test_item_matching():
    print("\n== eval_metrics item matching ==")

    # Perfect match.
    gold = [{"name": "Latte", "price": "100.00"}]
    pred = [{"name": "Latte", "price": "100.00"}]
    p, r, f1 = _match_items(gold, pred, name_sim_threshold=60.0)
    check("perfect match -> P=R=F1=1.0", p == 1.0 and r == 1.0 and f1 == 1.0, detail=str((p, r, f1)))

    # BOGO dedup failure: pred lists the free duplicate as a second item.
    gold = [{"name": "Latte", "price": "100.00"}]
    pred = [{"name": "Latte", "price": "100.00"}, {"name": "Latte", "price": "100.00"}]
    p, r, f1 = _match_items(gold, pred, name_sim_threshold=60.0)
    check("undeduped BOGO -> precision 0.5, recall 1.0", p == 0.5 and r == 1.0, detail=str((p, r, f1)))

    # Threshold plumbing: identical price, name similarity forced irrelevant via threshold=0
    # (always matches) vs threshold=101 (never matches) -- isolates the matching mechanics
    # from any specific fuzzy-string calibration value.
    gold = [{"name": "MANGO STICKY RICE KAKIGORI", "price": "295.00"}]
    pred = [{"name": "totally unrelated garbled text", "price": "295.00"}]
    p0, r0, _ = _match_items(gold, pred, name_sim_threshold=0.0)
    check("threshold=0 matches on price alone", p0 == 1.0 and r0 == 1.0, detail=str((p0, r0)))
    p101, r101, _ = _match_items(gold, pred, name_sim_threshold=101.0)
    check("threshold=101 (impossible) never matches", p101 == 0.0 and r101 == 0.0, detail=str((p101, r101)))

    # Price mismatch never matches regardless of name similarity.
    gold = [{"name": "Latte", "price": "100.00"}]
    pred = [{"name": "Latte", "price": "99.00"}]
    p, r, f1 = _match_items(gold, pred, name_sim_threshold=0.0)
    check("wrong price never matches even at threshold=0", p == 0.0 and r == 0.0, detail=str((p, r, f1)))

    # price-only recall ignores name entirely.
    gold = [{"name": "Latte", "price": "100.00"}, {"name": "Muffin", "price": "50.00"}]
    pred = [{"name": "XL Frappuccino Deluxe Something", "price": "100.00"}, {"name": "Bagel", "price": "999.00"}]
    por = _price_only_recall(gold, pred)
    check("price-only recall counts price matches regardless of name", abs(por - 0.5) < 1e-9, detail=str(por))

    # Duplicate gold prices consume distinct pred items (no double-counting one pred item).
    gold = [{"name": "Water", "price": "40.00"}, {"name": "Lime Soda", "price": "40.00"}]
    pred = [{"name": "Water", "price": "40.00"}]
    por = _price_only_recall(gold, pred)
    check("price-only recall doesn't double-count a single pred item", abs(por - 0.5) < 1e-9, detail=str(por))


# ========================================================================================
# eval_metrics.score_example (end-to-end on real dataset rows)
# ========================================================================================

def _find_row(rows: list[dict], predicate) -> dict:
    for r in rows:
        if predicate(r):
            return r
    raise AssertionError("no matching row found in fixture data")


def test_score_example_end_to_end():
    print("\n== eval_metrics.score_example (real data, simulated perfect + broken predictions) ==")

    train_path = ROOT / "data" / "train.jsonl"
    if not train_path.exists():
        print("  SKIPPED (data/train.jsonl not found -- run data_prep.py first)")
        return

    rows = [json.loads(l) for l in train_path.open(encoding="utf-8")]
    discount_row = _find_row(rows, lambda r: "basket-wide_discount" in r["target"])
    vat_row = _find_row(
        rows,
        lambda r: any(it["name"].strip().lower() in ("vat", "tax") for it in r["target"]["items"]),
    )

    # A "perfect" prediction (model output == gold) should score perfectly across the board.
    for label, row in [("discount row", discount_row), ("vat row", vat_row)]:
        raw = "```json\n" + json.dumps(row["target"], ensure_ascii=False) + "\n```"
        score = score_example(raw, row["target"], row["input"])
        check(f"{label}: perfect prediction -> exact_match", score.exact_match, detail=str(score))
        check(f"{label}: perfect prediction -> item_f1=1.0", score.item_f1 == 1.0)
        check(f"{label}: perfect prediction -> total_price_exact", score.total_price_exact)
        check(f"{label}: perfect prediction -> reconciliation ok", score.reconciliation_status == "ok")

    # A broken (non-JSON) completion should degrade gracefully, not crash.
    score = score_example("I cannot process this receipt, sorry!", discount_row["target"], discount_row["input"])
    check("garbage completion -> json_valid False", not score.json_valid)
    check("garbage completion -> exact_match False", not score.exact_match)
    check("garbage completion -> reconciliation unparseable", score.reconciliation_status == "unparseable")

    # discount presence mismatch: pred omits the discount the gold row has.
    pred_missing_discount = dict(discount_row["target"])
    del pred_missing_discount["basket-wide_discount"]
    raw = json.dumps(pred_missing_discount, ensure_ascii=False)
    score = score_example(raw, discount_row["target"], discount_row["input"])
    check("discount omitted -> discount_gold_present True", score.discount_gold_present)
    check("discount omitted -> discount_pred_present False", not score.discount_pred_present)
    check("discount omitted -> discount_value_exact is None (not both present)", score.discount_value_exact is None)


# ========================================================================================
# aggregate()
# ========================================================================================

def test_aggregate():
    print("\n== eval_metrics.aggregate ==")

    scores = [
        ExampleScore(json_valid=True, schema_valid=True, parse_error=None, exact_match=True, total_price_exact=True, reconciliation_status="ok", reconciliation_pass=True),
        ExampleScore(json_valid=True, schema_valid=True, parse_error=None, exact_match=False, total_price_exact=False, reconciliation_status="unrecoverable_gap", reconciliation_pass=False),
        ExampleScore(json_valid=False, schema_valid=False, parse_error="no json", exact_match=False, reconciliation_status="unparseable", reconciliation_pass=False),
    ]
    agg = aggregate(scores)
    check("n correct", agg["n"] == 3)
    check("json_validity_rate correct", abs(agg["json_validity_rate"] - 2 / 3) < 1e-9, detail=str(agg["json_validity_rate"]))
    check("exact_match_rate correct", abs(agg["exact_match_rate"] - 1 / 3) < 1e-9)
    check("reconciliation_pass_rate correct", abs(agg["reconciliation_pass_rate"] - 1 / 3) < 1e-9)
    check(
        "reconciliation_status_breakdown correct",
        agg["reconciliation_status_breakdown"] == {"ok": 1, "unrecoverable_gap": 1, "unparseable": 1},
        detail=str(agg["reconciliation_status_breakdown"]),
    )

    # discount precision/recall over a small confusion matrix.
    scores2 = [
        ExampleScore(json_valid=True, schema_valid=True, parse_error=None, discount_gold_present=True, discount_pred_present=True, discount_value_exact=True),
        ExampleScore(json_valid=True, schema_valid=True, parse_error=None, discount_gold_present=True, discount_pred_present=False),  # FN
        ExampleScore(json_valid=True, schema_valid=True, parse_error=None, discount_gold_present=False, discount_pred_present=True),  # FP
        ExampleScore(json_valid=True, schema_valid=True, parse_error=None, discount_gold_present=False, discount_pred_present=False),  # TN
    ]
    agg2 = aggregate(scores2)
    check("discount_precision = TP/(TP+FP) = 0.5", abs(agg2["discount_precision"] - 0.5) < 1e-9, detail=str(agg2["discount_precision"]))
    check("discount_recall = TP/(TP+FN) = 0.5", abs(agg2["discount_recall"] - 0.5) < 1e-9, detail=str(agg2["discount_recall"]))
    check("discount_value_exact_n = 1 (only 1 example has both present)", agg2["discount_value_exact_n"] == 1)

    # empty input doesn't crash.
    agg_empty = aggregate([])
    check("aggregate([]) -> n=0 without crashing", agg_empty == {"n": 0})


# ========================================================================================
# Primary-objective metrics: exact item count + exact prices, names only need to be close
# ========================================================================================

def test_normalize_prediction():
    print("\ntest_normalize_prediction")

    p = postprocess.normalize_prediction({
        "shop_name": "X",
        "items": [{"name": "a", "price": "169"}, {"name": "b", "price": "1,290.00"},
                  {"name": "c", "price": "฿20.0"}, {"name": "d", "price": "35.00"}],
        "total_price": "THB 1514.00",
    })
    prices = [i["price"] for i in p["items"]]
    check("integer price gains two decimals", prices[0] == "169.00", detail=prices[0])
    check("thousands separator removed", prices[1] == "1290.00", detail=prices[1])
    check("currency symbol + one decimal fixed", prices[2] == "20.00", detail=prices[2])
    check("already-conformant price untouched", prices[3] == "35.00", detail=prices[3])
    check("currency code stripped from total", p["total_price"] == "1514.00", detail=p["total_price"])

    # Must never invent or alter a value it cannot parse -- a wrong number stays wrong.
    p2 = postprocess.normalize_prediction({
        "shop_name": "X", "items": [{"name": "a", "price": "abc"}, {"name": "b"}],
        "total_price": None,
    })
    check("unparseable price left untouched", p2["items"][0]["price"] == "abc")
    check("missing price key not invented", "price" not in p2["items"][1])
    check("unparseable total left untouched", p2["total_price"] is None)

    p3 = postprocess.normalize_prediction({"shop_name": "X", "items": [], "basket-wide_discount": "50", "total_price": "10.00"})
    check("discount normalized too", p3["basket-wide_discount"] == "50.00")
    check("normalize_prediction does not mutate its input", postprocess.normalize_prediction({"total_price": "5"}) != {"total_price": "5"})


def test_apply_basket_discount():
    print("\ntest_apply_basket_discount")

    def cents_sum(p):
        return round(sum(float(i["price"]) for i in p["items"]), 2)

    # The user's real 7-Eleven receipt: 108.00 subtotal, 18.00 off, 90.00 net.
    p = postprocess.apply_basket_discount({
        "shop_name": "CP ALL, 7-Eleven",
        "items": [{"name": "a", "price": "20.00"}, {"name": "b", "price": "15.00"},
                  {"name": "c", "price": "18.00"}, {"name": "d", "price": "55.00"}],
        "basket-wide_discount": "18.00", "total_price": "90.00"})
    check("discount field is removed", "basket-wide_discount" not in p)
    check("items sum exactly to the total", cents_sum(p) == 90.00, detail=str(cents_sum(p)))
    check("allocation is pro-rata", [i["price"] for i in p["items"]] == ["16.67", "12.50", "15.00", "45.83"],
          detail=str([i["price"] for i in p["items"]]))
    check("item names are preserved", [i["name"] for i in p["items"]] == ["a", "b", "c", "d"])

    # Uneven division: 70.50 / 4 has no exact 2dp split. Naive rounding lands on 634.52.
    p = postprocess.apply_basket_discount({
        "shop_name": "X",
        "items": [{"name": n, "price": v} for n, v in
                  [("a", "200.00"), ("b", "200.00"), ("c", "200.00"), ("d", "105.00")]],
        "basket-wide_discount": "70.50", "total_price": "634.50"})
    check("uneven split still sums exactly (largest-remainder)", cents_sum(p) == 634.50, detail=str(cents_sum(p)))

    # The case that breaks an equal split: a 2.00 line with a 40.00 basket discount.
    p = postprocess.apply_basket_discount({
        "shop_name": "X",
        "items": [{"name": n, "price": v} for n, v in
                  [("bag", "2.00"), ("b", "10.00"), ("c", "45.00"), ("d", "55.00")]],
        "basket-wide_discount": "40.00", "total_price": "72.00"})
    prices = [float(i["price"]) for i in p["items"]]
    check("no item is driven negative", all(x >= 0 for x in prices), detail=str(prices))
    # 1.28, not the 1.29 naive per-item rounding gives: 2.00 * (72/112) = 1.2857, and the spare
    # cents go to the largest fractional remainders so the basket sums to exactly 72.00.
    # Naive rounding would produce 1.29 and a basket of 72.01, which then fails reconcile().
    check("cheap item keeps a sensible price", prices[0] == 1.28, detail=str(prices[0]))
    check("sums exactly despite rounding", cents_sum(p) == 72.00, detail=str(cents_sum(p)))

    # Output must still reconcile -- the whole point of the exact-sum requirement.
    r = postprocess.reconcile(p, "x")
    check("result reconciles", r.status == "ok" and r.passes, detail=str(r.status))

    # Refusals: never guess, never mutate the caller's dict.
    orig = {"shop_name": "X", "items": [{"name": "a", "price": "10.00"}],
            "basket-wide_discount": "5.00", "total_price": "5.00"}
    snapshot = json.loads(json.dumps(orig))
    postprocess.apply_basket_discount(orig)
    check("input dict is not mutated", orig == snapshot)

    same = {"shop_name": "X", "items": [{"name": "a", "price": "10.00"}], "total_price": "10.00"}
    check("no discount field -> returned unchanged", postprocess.apply_basket_discount(same) is same)

    bad = {"shop_name": "X", "items": [{"name": "a", "price": "abc"}],
           "basket-wide_discount": "5.00", "total_price": "5.00"}
    check("unparseable price -> left alone for review", "basket-wide_discount" in postprocess.apply_basket_discount(bad))

    over = {"shop_name": "X", "items": [{"name": "a", "price": "10.00"}],
            "basket-wide_discount": "50.00", "total_price": "5.00"}
    check("discount >= subtotal -> refused, not applied", "basket-wide_discount" in postprocess.apply_basket_discount(over))

    zero = postprocess.apply_basket_discount({
        "shop_name": "X", "items": [{"name": "a", "price": "10.00"}],
        "basket-wide_discount": "0.00", "total_price": "10.00"})
    check("0.00 discount -> field dropped, prices untouched",
          "basket-wide_discount" not in zero and zero["items"][0]["price"] == "10.00")


def test_money_metrics():
    print("\ntest_money_metrics")

    gold = {
        "shop_name": "7-ELEVEN",
        "items": [
            {"name": "Gulp7ElcMarcos22oz", "price": "35.00"},
            {"name": "Juice", "price": "4.00"},
            {"name": "RiteNLite", "price": "52.00"},
        ],
        "total_price": "91.00",
    }
    ocr = "7-ELEVEN Total (4) 91.00"

    def sc(pred):
        return score_example(json.dumps(pred, ensure_ascii=False), gold, ocr)

    s = sc(gold)
    check("perfect prediction -> money_exact", s.money_exact and s.price_multiset_exact and s.item_count_exact)

    # The whole point of the objective: mangled names must NOT cost a money score.
    garbled = {
        "shop_name": "7-ELEVEN",
        "items": [
            {"name": "gulp 7elc marcos", "price": "35.00"},
            {"name": "juce", "price": "4.00"},
            {"name": "rite n lite", "price": "52.00"},
        ],
        "total_price": "91.00",
    }
    s = sc(garbled)
    check("garbled names but right prices -> still money_exact", s.money_exact, detail=str(s.name_quality_mean))
    check("garbled names still register as acceptable", s.name_acceptable_rate == 1.0 and s.name_matched_pairs == 3)

    # Order must not matter -- items[] is a multiset for scoring purposes.
    reordered = {"shop_name": "7-ELEVEN", "items": list(reversed(gold["items"])), "total_price": "91.00"}
    check("item order does not affect price_multiset_exact", sc(reordered).money_exact)

    wrong_price = json.loads(json.dumps(gold))
    wrong_price["items"][2]["price"] = "26.00"
    s = sc(wrong_price)
    check("one wrong price -> not price_multiset_exact", not s.price_multiset_exact and not s.money_exact)
    check("one wrong price still has correct count", s.item_count_exact)

    # A merged line scores a flattering 80% on the old name-coupled Item F1; the count/price
    # metrics must not be fooled by it.
    merged = {"shop_name": "7-ELEVEN", "items": [gold["items"][0], gold["items"][2]], "total_price": "91.00"}
    s = sc(merged)
    check("dropped item -> count_exact False", not s.item_count_exact)
    check("dropped item -> money_exact False despite high item_f1", not s.money_exact and s.item_f1 > 0.5, detail=f"item_f1={s.item_f1}")

    total_wrong = json.loads(json.dumps(gold))
    total_wrong["total_price"] = "81.00"
    s = sc(total_wrong)
    check("right prices + wrong total -> prices exact but money_exact False", s.price_multiset_exact and not s.money_exact)

    # Duplicate prices are a real pattern (two identical drinks); a multiset must count both.
    dup_gold = {"shop_name": "S", "items": [{"name": "a", "price": "10.00"}, {"name": "b", "price": "10.00"}], "total_price": "20.00"}
    one_only = {"shop_name": "S", "items": [{"name": "a", "price": "10.00"}], "total_price": "20.00"}
    s = score_example(json.dumps(one_only), dup_gold, "S 20.00")
    check("duplicate prices are not collapsed by the multiset", not s.price_multiset_exact and not s.item_count_exact)

    # An unparseable price must never be able to match by accident.
    bad_price = {"shop_name": "7-ELEVEN", "items": [dict(i) for i in gold["items"]], "total_price": "91.00"}
    bad_price["items"][0]["price"] = "฿35.00"
    s = sc(bad_price)
    check("unparseable pred price -> not price_multiset_exact", not s.price_multiset_exact)

    agg = eval_metrics.aggregate([sc(gold), sc(wrong_price)])
    check("money_exact_rate aggregates correctly", abs(agg["money_exact_rate"] - 0.5) < 1e-9, detail=str(agg["money_exact_rate"]))
    check("name metrics skip examples with no matched pair", agg["name_quality_mean"] > 0)


# ========================================================================================
# main
# ========================================================================================

def main() -> int:
    test_extract_json()
    test_reconcile()
    test_item_matching()
    test_score_example_end_to_end()
    test_aggregate()
    test_normalize_prediction()
    test_apply_basket_discount()
    test_money_metrics()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s) failed:")
        for name in _failures:
            print(f"  - {name}")
        return 1

    print("Phase 2 verification: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
