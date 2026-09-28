"""Is this OCR text actually a receipt?

The model will read anything. Given a paragraph of prose it answered
`shopName: "The Rain in Spain", total: "10.00"`; given a menu with no prices
printed on it, it invented 280.00 for three dishes and 180.00 for a fourth
and returned a confident ฿940 total. `ok: true` both times, `reconciles: true`
for the prose. Nothing downstream could tell those from a real receipt, so a
user who photographed a menu would get a ฿940 expense in their budget.

The tell is not in the model's output — it is in the OCR text. **A receipt has
prices printed on it.** Neither of those inputs contained a single
money-shaped token; every number in the answers was invented. So the check is
deterministic, runs on the OCR text before the model is called, and costs
nothing.

WHICH WAY TO FAIL
-----------------
A blurry real receipt whose prices did not survive OCR gets rejected, and the
user retakes the photo — a few seconds. A menu that gets accepted becomes
money in someone's spending totals that they never spent, and nothing later
in the pipeline will question it. So this errs toward rejecting, and the
threshold is deliberately low (two prices: one item and a total) rather than
tuned to catch every non-receipt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 1,234.50 · 65.00 · 20.0 — a decimal amount. One decimal place is allowed
# because tills print it ("THB 20.0" on a real transit receipt in the gold
# corpus, which a two-decimal rule rejected). Bare integers are NOT a price
# on their own: a menu, a page number and a phone number are all integers.
_PRICE = re.compile(r"\d{1,3}(?:[,\s]\d{3})*[.,]\d{1,2}(?!\d)")

# ...unless a currency marker is sitting next to them. "THB 69" or "69 บาท"
# is an amount in a way that a bare "69" is not.
_CURRENCY_AMOUNT = re.compile(
    r"(?:(?:thb|บาท|฿)\s*\d+|\d+\s*(?:thb|บาท|฿))", re.IGNORECASE
)

# Words that only appear on a transaction record — not on a menu, a page of
# prose or a business card. Two of these carry a document whose amounts did
# not survive OCR (a kiosk slip in the gold corpus prints its total as a bare
# "69", which no amount rule can safely accept on its own).
_DOCUMENT_WORDS = (
    "receipt", "invoice", "customer copy", "merchant copy", "tax invoice",
    "sale", "change due", "cashier", "pos ", "terminal",
    "ใบเสร็จ", "ใบกำกับภาษี", "ใบกํากับภาษี", "เงินทอน", "ขายสินค้า", "สำเนา",
)
MIN_DOCUMENT_WORDS = 2

# Words a Thai or English till prints near the bottom. One of these plus a
# single price is as convincing as two prices.
_TOTAL_WORDS = (
    "total", "subtotal", "grand total", "amount", "balance", "net", "cash",
    "change", "vat", "tax", "receipt", "invoice", "qty",
    "รวม", "ยอดรวม", "ยอดสุทธิ", "สุทธิ", "รวมทั้งสิ้น", "เงินสด", "เงินทอน",
    "ภาษี", "ใบเสร็จ", "ใบกำกับภาษี", "จำนวน", "บาท",
)

# One item and a total. Below this we are looking at something that is not a
# till slip, or at OCR that failed badly enough to be worth reshooting.
MIN_PRICES = 2


@dataclass(frozen=True)
class GateResult:
    is_receipt: bool
    prices: int
    total_word: bool
    document_words: int = 0
    reason: str | None = None

    def as_dict(self) -> dict:
        return {"prices": self.prices, "total_word": self.total_word,
                "document_words": self.document_words}


def count_prices(text: str) -> int:
    return len(_PRICE.findall(text)) + len(_CURRENCY_AMOUNT.findall(text))


def count_document_words(text: str) -> int:
    lowered = text.lower()
    return sum(1 for word in _DOCUMENT_WORDS if word in lowered)


def has_total_word(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _TOTAL_WORDS)


def check(text: str) -> GateResult:
    """Decide before spending a model call on it."""
    prices = count_prices(text)
    total_word = has_total_word(text)
    document_words = count_document_words(text)

    # Two prices, or one price plus a till word. A one-line receipt is real
    # ("7-ELEVEN / water 10.00 / TOTAL 10.00" is three), and so is a receipt
    # whose only surviving price is the total.
    if prices >= MIN_PRICES or (prices >= 1 and total_word):
        return GateResult(True, prices, total_word, document_words)

    # No usable amount, but the page says plainly what it is. Rejecting this
    # sends the user into a retake loop they cannot win: reshooting will not
    # put a decimal point on a receipt that never printed one.
    if document_words >= MIN_DOCUMENT_WORDS:
        return GateResult(True, prices, total_word, document_words)

    if prices == 0:
        reason = (
            "No prices were found in this image. A receipt prints an amount "
            "for each line; this looks like something else — or the photo is "
            "too blurry to read."
        )
    else:
        reason = (
            "Only one amount was found and nothing that looks like a total. "
            "Make sure the whole receipt is in frame, including the bottom."
        )
    return GateResult(False, prices, total_word, document_words, reason)
