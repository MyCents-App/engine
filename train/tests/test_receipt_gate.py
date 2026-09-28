"""The not-a-receipt gate.

Validated against the 84 real Surya OCR outputs in the gold corpus — all 84
must pass, because a false rejection sends a user to retake a photo that was
fine. The non-receipts are the cases that made this necessary: the model
answered all of them with confident, entirely invented money.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.receipt_gate import check, count_prices

GOLD = (pathlib.Path(__file__).resolve().parents[3]
        / "test" / "overfit-test" / "dataset" / "surya")


@pytest.mark.skipif(not GOLD.exists(), reason="gold corpus not present")
def test_every_real_receipt_passes():
    rejected = []
    for path in sorted(GOLD.glob("*.json")):
        text = json.loads(path.read_text(encoding="utf-8")).get("text", "")
        if not check(text).is_receipt:
            rejected.append(path.name)
    assert rejected == [], f"false rejections: {rejected}"


@pytest.mark.parametrize("text", [
    # What the model actually did with these: "The Rain in Spain", 10.00.
    "Chapter 4. The rain in Spain falls mainly on the plain.",
    # And this: four items at 280/180, a confident 940.00 total.
    "MENU\nTom Yum Goong\nPad Thai\nGreen Curry\nMango Sticky Rice",
    "Somchai Jaidee\nSenior Engineer\n+66 81 234 5678\nsomchai@example.com",
    "Page 12\nIn 1984 the population reached 3 million, by 1990 it was 4 million.",
    "EXIT",
    "   \n\n  ",
])
def test_non_receipts_are_rejected(text):
    result = check(text)
    assert not result.is_receipt
    assert result.reason


def test_a_minimal_real_receipt_passes():
    assert check("7-ELEVEN\nwater 10.00\nTOTAL 10.00").is_receipt


def test_one_price_plus_a_total_word_is_enough():
    """A receipt whose only surviving price is the total."""
    assert check("ร้านค้า\nรวม 85.00").is_receipt


def test_one_price_alone_is_not():
    assert not check("something 85.00").is_receipt


def test_single_decimal_place_counts():
    """A real transit receipt in the corpus prints THB 20.0, and a
    two-decimal rule rejected it."""
    assert count_prices("THB 20.0") >= 1


def test_currency_marker_makes_a_bare_integer_an_amount():
    assert count_prices("THB 69") >= 1
    assert count_prices("69 บาท") >= 1
    assert count_prices("69") == 0


def test_document_words_carry_a_receipt_with_no_parseable_amount():
    """A kiosk slip whose total is a bare 69. Rejecting it would send the
    user into a retake loop they cannot win."""
    assert check("Kiosk Payment Receipt\nSID 03249\nSALE\n69\nCUSTOMER COPY").is_receipt


def test_the_reason_is_something_a_user_can_act_on():
    reason = check("MENU\nPad Thai\nGreen Curry").reason
    assert "receipt" in reason.lower() and len(reason) > 40
