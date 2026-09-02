"""Pull the purchase date out of raw OCR text.

The model does not emit a date: engine/task.md §2 froze the output schema as
{shop_name, items, total_price, basket-wide_discount} and left the date out.
The app and the database both need one, and retraining to add a field is a
poor trade when a receipt's date is a deterministic pattern.

So this is a CODE-layer function, in the spirit of task.md §4 — the model
handles what needs judgement, the code handles what is mechanical.

THE BUDDHIST ERA TRAP
---------------------
Thai receipts overwhelmingly print the Buddhist year: 2569, not 2026. Reading
it literally puts every scanned receipt 543 years in the future, which then
sorts to the top of every history screen forever and silently breaks any
"this month" filter. Conversion is by far the most important thing here.

Both eras appear in the wild — international chains and card terminals often
print CE — so the era is detected from the value rather than assumed.
"""

from __future__ import annotations

import re
from datetime import date

BUDDHIST_OFFSET = 543

# A receipt is never far from "now"; anything outside this window is a
# misparse (a phone number, a product code, a VAT id) rather than a date.
MIN_YEAR = 2000
MAX_FUTURE_DAYS = 2  # tolerate a timezone slip, nothing more

THAI_MONTHS = {
    "ม.ค": 1, "มกราคม": 1, "ก.พ": 2, "กุมภาพันธ์": 2, "มี.ค": 3, "มีนาคม": 3,
    "เม.ย": 4, "เมษายน": 4, "พ.ค": 5, "พฤษภาคม": 5, "มิ.ย": 6, "มิถุนายน": 6,
    "ก.ค": 7, "กรกฎาคม": 7, "ส.ค": 8, "สิงหาคม": 8, "ก.ย": 9, "กันยายน": 9,
    "ต.ค": 10, "ตุลาคม": 10, "พ.ย": 11, "พฤศจิกายน": 11, "ธ.ค": 12, "ธันวาคม": 12,
}

EN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# dd/mm/yyyy or dd-mm-yy, the dominant Thai POS format. Thai receipts are
# day-first; mm/dd is not used, so the ambiguity of 03/04 does not arise.
_NUMERIC = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
# yyyy-mm-dd (ISO), common on card slips and e-receipts.
_ISO = re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b")
# "12 ม.ค. 2569" / "12 Jan 2026"
_TEXTUAL = re.compile(
    r"\b(\d{1,2})\s*([A-Za-z]{3,9}|[ก-๙.]{2,12})\.?\s*(\d{2,4})\b"
)


def _normalize_year(year: int) -> int | None:
    """Two-digit years, and Buddhist -> Common Era."""
    if year < 100:
        # 69 -> 2569 (BE) rather than 1969: Thai POS prints BE two-digit far
        # more often than a 20th-century date appears on a receipt.
        year += 2500 if year >= 50 else 2000
    if year > 2400:                      # unmistakably Buddhist
        year -= BUDDHIST_OFFSET
    return year


def _plausible(d: date, today: date | None = None) -> bool:
    today = today or date.today()
    if d.year < MIN_YEAR:
        return False
    return (d - today).days <= MAX_FUTURE_DAYS


def _build(year: int, month: int, day: int, today: date | None = None) -> date | None:
    year = _normalize_year(year)
    if year is None or not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    try:
        d = date(year, month, day)
    except ValueError:                    # 31 February and friends
        return None
    return d if _plausible(d, today) else None


def _month_number(token: str) -> int | None:
    t = token.strip().lower().rstrip(".")
    if t[:3] in EN_MONTHS:
        return EN_MONTHS[t[:3]]
    for name, num in THAI_MONTHS.items():
        if t.startswith(name.rstrip(".")):
            return num
    return None


def extract_receipt_date(ocr_text: str, today: date | None = None) -> date | None:
    """Best-effort purchase date, or None.

    Returns the FIRST plausible date in the text. Receipts print the
    transaction date near the top; later dates tend to be promotional expiry
    or "member since", which are not what was spent.

    None is a perfectly good answer — the app defaults to today and the user
    can correct it on the review screen. Guessing wrong is worse than not
    guessing, because a wrong date is silent.
    """
    if not ocr_text:
        return None

    for match in _ISO.finditer(ocr_text):
        y, m, d = (int(g) for g in match.groups())
        got = _build(y, m, d, today)
        if got:
            return got

    for match in _NUMERIC.finditer(ocr_text):
        d, m, y = (int(g) for g in match.groups())
        got = _build(y, m, d, today)
        if got:
            return got

    for match in _TEXTUAL.finditer(ocr_text):
        day_s, month_s, year_s = match.groups()
        month = _month_number(month_s)
        if month is None:
            continue
        got = _build(int(year_s), month, int(day_s), today)
        if got:
            return got

    return None
