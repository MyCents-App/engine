"""eval_joint.py's scoring, with hand-written completions -- no model, no GPU."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rapidfuzz")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_joint  # noqa: E402
from eval_joint import CategoryTally, pair_items, score_record, strip_categories  # noqa: E402

GOLD = {
    "shop_name": "7-Eleven",
    "items": [
        {"name": "มาม่าคัพ รสต้มยำกุ้ง", "price": "15.00", "c": "Groceries", "s": "Instant & frozen"},
        {"name": "น้ำดื่มคริสตัล 1500", "price": "14.00", "c": "Groceries", "s": None},
        {"name": "ขนมปังไส้กรอก", "price": "25.00", "c": "Food & Dining", "s": "Bakery & desserts"},
        {"name": "vat", "price": "3.78", "c": None, "s": None},
    ],
    "total_price": "57.78",
}
RECORD = {"id": "photo-x", "meta": {"kind": "real"}, "input": "7-ELEVEN\nvat 3.78\n57.78",
          "target": GOLD}


def answer(target: dict) -> str:
    return json.dumps(target, ensure_ascii=False)


def test_perfect_answer_scores_perfectly_on_both_halves():
    score, items, why = score_record(RECORD, answer(GOLD), legacy=False)
    assert score.money_exact and score.exact_match and why == ""
    tally = CategoryTally()
    tally.add(GOLD["items"], items)
    s = tally.summary()
    assert s["cat_acc"] == 1.0 and s["sub_precision"] == 1.0 and s["tax_row_clean"] == 1.0
    assert s["paired_share"] == 1.0 and s["taxonomy_valid"] == 1.0


def test_a_wrong_category_does_not_touch_extraction():
    pred = json.loads(json.dumps(GOLD))
    pred["items"][0]["c"], pred["items"][0]["s"] = "Food & Dining", None
    score, items, _ = score_record(RECORD, answer(pred), legacy=False)
    assert score.money_exact and score.exact_match   # c/s stripped before extraction scoring
    tally = CategoryTally()
    tally.add(GOLD["items"], items)
    assert tally.summary()["cat_acc"] == pytest.approx(2 / 3)
    assert tally.confusion[("Groceries", "Food & Dining")] == 1


def test_a_dropped_item_costs_money_exact_not_every_later_category():
    """Positional scoring would compare item 2 with gold item 1 onward and mark
    every one wrong. Pairing by price and name keeps them right."""
    pred = json.loads(json.dumps(GOLD))
    del pred["items"][0]
    score, items, why = score_record(RECORD, answer(pred), legacy=False)
    assert not score.money_exact and why == "3 items for 4"
    tally = CategoryTally()
    tally.add(GOLD["items"], items)
    s = tally.summary()
    assert s["cat_acc"] == 1.0
    assert s["paired_share"] == pytest.approx(2 / 3)


def test_misread_price_still_pairs_by_name():
    gold = [{"name": "ขนมปังไส้กรอก", "price": "25.00"}]
    pred = [{"name": "ขนมปังไส้กรอก", "price": "26.00"}]
    assert pair_items(gold, pred) == [(0, 0)]


def test_same_price_pairs_the_closer_name():
    gold = [{"name": "coke", "price": "20.00"}, {"name": "water", "price": "20.00"}]
    pred = [{"name": "water", "price": "20.00"}, {"name": "coke", "price": "20.00"}]
    assert pair_items(gold, pred) == [(0, 1), (1, 0)]


def test_subcategory_outside_its_category_is_invalid():
    pred = json.loads(json.dumps(GOLD))
    pred["items"][1]["s"] = "Drinks & beverages"     # exists only under Food & Dining
    _, items, _ = score_record(RECORD, answer(pred), legacy=False)
    tally = CategoryTally()
    tally.add(GOLD["items"], items)
    assert tally.summary()["taxonomy_valid"] == pytest.approx(2 / 3)


def test_categorized_tax_row_is_counted_against_the_model():
    pred = json.loads(json.dumps(GOLD))
    pred["items"][3]["c"] = "Groceries"
    _, items, _ = score_record(RECORD, answer(pred), legacy=False)
    tally = CategoryTally()
    tally.add(GOLD["items"], items)
    assert tally.summary()["tax_row_clean"] == 0.0
    assert tally.summary()["cat_acc"] == 1.0         # the tax row is not a purchase


def test_legacy_answer_without_categories_is_scored_on_extraction_only():
    score, items, why = score_record(RECORD, answer(strip_categories(GOLD)), legacy=True)
    assert score.money_exact and score.exact_match and items is None and why == ""


def test_unparseable_answer():
    score, items, why = score_record(RECORD, '{"shop_name": "7-El', legacy=False)
    assert not score.json_valid and items is None and why == "unparseable"
    tally = CategoryTally()
    tally.add(GOLD["items"], items)
    assert tally.parsed == 0 and tally.summary()["cat_acc"] is None


def test_rescoring_from_cache_needs_no_model(tmp_path, monkeypatch):
    """A cached completion is rescored without generate() ever being called --
    the path taken after a validation label is fixed."""
    ckpt = tmp_path / "joint" / "checkpoint-25"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps(RECORD, ensure_ascii=False) + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    key = eval_joint.prompt_key(eval_joint.build_messages(RECORD, legacy=False))
    path = eval_joint.cache_path(cache, str(ckpt), legacy=False)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": key, "max_new_tokens": 1024,
                                "completion": answer(GOLD), "tokens": 90,
                                "truncated": False}, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("generate() called despite a full cache")
    monkeypatch.setattr(eval_joint, "generate", boom)

    out = tmp_path / "report.md"
    eval_joint.main(["--checkpoints", str(ckpt), "--val", str(val),
                     "--cache-dir", str(cache), "--out", str(out)])
    report = out.read_text(encoding="utf-8")
    assert "joint/checkpoint-25" in report and "**100.0%**" in report


def test_subcategory_offered_where_the_label_has_none_is_split_out():
    gold = [dict(GOLD["items"][1])]                       # s: None in the label
    pred = [{**gold[0], "s": "Household supplies"}]
    tally = CategoryTally()
    tally.add(gold, pred)
    s = tally.summary()
    assert s["sub_precision"] == 0.0                      # counted wrong overall
    assert s["sub_precision_labelled"] is None            # but no labelled case exists


def test_receipts_the_legacy_model_trained_on_are_found_by_money(tmp_path):
    seen = {**GOLD, "shop_name": "CP ALL, 7-Eleven",      # relabelled, re-OCR'd copy
            "items": [{"name": i["name"] if eval_joint.is_tax(i) else i["name"] + "x",
                       "price": i["price"]} for i in GOLD["items"]]}
    path = tmp_path / "real_train.jsonl"
    path.write_text(json.dumps({"input": "old ocr", "target": seen}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    other = {**RECORD, "id": "photo-y",
             "target": {**GOLD, "total_price": "99.00"}}
    assert eval_joint.seen_ids([RECORD, other], path) == {"photo-x"}
