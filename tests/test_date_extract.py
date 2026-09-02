"""Receipt date extraction.

The Buddhist-era conversion is the reason this file exists. Reading 2569
literally puts every Thai receipt 543 years in the future, where it sorts to
the top of the history screen forever and falls outside every "this month"
filter — while looking like a perfectly valid date the whole way.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.date_extract import extract_receipt_date

TODAY = date(2026, 9, 2)


class TestBuddhistEra:
    def test_four_digit_buddhist_year_converts(self):
        assert extract_receipt_date("วันที่ 15/08/2569", TODAY) == date(2026, 8, 15)

    def test_two_digit_buddhist_year_converts(self):
        # 69 -> 2569 BE -> 2026 CE, not 1969.
        assert extract_receipt_date("15/08/69", TODAY) == date(2026, 8, 15)

    def test_buddhist_iso_form(self):
        assert extract_receipt_date("2569-08-15", TODAY) == date(2026, 8, 15)

    def test_thai_month_name(self):
        assert extract_receipt_date("15 ส.ค. 2569", TODAY) == date(2026, 8, 15)

    def test_full_thai_month_name(self):
        assert extract_receipt_date("15 สิงหาคม 2569", TODAY) == date(2026, 8, 15)


class TestCommonEra:
    """International chains and card terminals often print CE, so the era is
    detected from the value rather than assumed."""

    def test_common_era_is_left_alone(self):
        assert extract_receipt_date("15/08/2026", TODAY) == date(2026, 8, 15)

    def test_iso_format(self):
        assert extract_receipt_date("2026-08-15 14:32", TODAY) == date(2026, 8, 15)

    def test_english_month(self):
        assert extract_receipt_date("15 Aug 2026", TODAY) == date(2026, 8, 15)

    def test_two_digit_common_era(self):
        assert extract_receipt_date("15/08/26", TODAY) == date(2026, 8, 15)


class TestDayFirst:
    """Thai receipts are day-first. 03/04 is 3 April, never 4 March."""

    def test_day_first_is_assumed(self):
        assert extract_receipt_date("03/04/2026", TODAY) == date(2026, 4, 3)

    @pytest.mark.parametrize("sep", ["/", "-", "."])
    def test_separators(self, sep):
        assert extract_receipt_date(f"15{sep}08{sep}2026", TODAY) == date(2026, 8, 15)


class TestRejectsNonsense:
    """A wrong date is worse than no date: the app defaults to today and the
    user can fix it, but a silently wrong date is never noticed."""

    @pytest.mark.parametrize(
        "text",
        [
            "TAX ID 0105558103019",     # long digit runs
            "TEL 02-123-4567",          # phone number
            "Total 1,234.00",
            "",
            "no date here at all",
            "32/13/2026",               # impossible day and month
            "31/02/2026",               # 31 February
        ],
    )
    def test_returns_none(self, text):
        assert extract_receipt_date(text, TODAY) is None

    def test_future_dates_rejected(self):
        # A receipt cannot be from next year — that is a misparse.
        assert extract_receipt_date("15/08/2030", TODAY) is None

    def test_ancient_dates_rejected(self):
        assert extract_receipt_date("15/08/1985", TODAY) is None


class TestRealReceiptText:
    def test_finds_date_inside_a_receipt(self):
        ocr = (
            "7-ELEVEN\n"
            "สาขา 12345\n"
            "TAX ID 0105558103019\n"
            "วันที่ 02/09/2569 เวลา 14:32\n"
            "มาม่า 18.00\n"
            "รวม 18.00\n"
        )
        assert extract_receipt_date(ocr, TODAY) == date(2026, 9, 2)

    def test_transaction_date_wins_over_a_later_one(self):
        # Receipts print the transaction date near the top; a later date is
        # usually a promo expiry or "member since", not what was spent.
        ocr = "วันที่ 02/09/2569\nสมาชิกหมดอายุ 01/01/2570\n"
        assert extract_receipt_date(ocr, TODAY) == date(2026, 9, 2)

    def test_tax_id_before_the_date_does_not_confuse_it(self):
        ocr = "TAX ID 0105558103019\nDATE 02-09-2569\n"
        assert extract_receipt_date(ocr, TODAY) == date(2026, 9, 2)
