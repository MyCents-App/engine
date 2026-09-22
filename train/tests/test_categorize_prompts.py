"""The categorization prompt contract: positional output, strict validation."""

from __future__ import annotations

import json

from categorize_prompts import (
    CATEGORIES,
    build_messages,
    build_user_message,
    parse_output,
    serialize_target,
)

ITEMS = [{"name": "ชีสโรลไส้กรอก", "name_en": None},
         {"name": "เอโร่ ไข่ไก่", "name_en": "ARO Chicken Egg"}]


def test_user_message_numbers_items_and_brackets_known_english():
    msg = build_user_message("7-Eleven", ITEMS)
    assert msg.splitlines() == [
        "shop: 7-Eleven", "items:",
        "1. ชีสโรลไส้กรอก", "2. เอโร่ ไข่ไก่ [ARO Chicken Egg]",
    ]


def test_unknown_shop_is_spelled_out():
    assert build_user_message(None, ITEMS[:1]).startswith("shop: (unknown)")


def test_messages_round_trip_through_parse():
    labels = [("Groceries", "Snacks & sweets"), ("Groceries", None)]
    msgs = build_messages("7-Eleven", ITEMS, labels)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert parse_output(msgs[-1]["content"], expected=2) == labels


def test_parse_rejects_wrong_length():
    raw = serialize_target([("Groceries", None)])
    assert parse_output(raw, expected=2) is None


def test_parse_rejects_unknown_category():
    assert parse_output('{"items":[{"c":"Snacks","s":null}]}', expected=1) is None


def test_parse_drops_subcategory_from_wrong_category_but_keeps_row():
    raw = '{"items":[{"c":"Groceries","s":"Dental"}]}'
    assert parse_output(raw, expected=1) == [("Groceries", None)]


def test_parse_tolerates_prose_around_the_json():
    raw = 'Sure! {"items":[{"c":"Transport","s":"Ride-hailing"}]} done'
    assert parse_output(raw, expected=1) == [("Transport", "Ride-hailing")]


def test_taxonomy_matches_the_app():
    assert len(CATEGORIES) == 8
    assert all(subs for subs in CATEGORIES.values())
    assert json.loads(serialize_target([("Education", None)])) == {
        "items": [{"c": "Education", "s": None}]}
