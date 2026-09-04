"""Tests for app/stitch.py -- joining several photos of one receipt.

No GPU and no model: the stitcher is pure text, which is the point of it being
its own module. Run with the rest of the CPU tests:

    .venv-test\\Scripts\\python.exe -m pytest tests/ -q

The fixtures are written the way the failure actually looks in production: the
same printed line, OCR'd twice from two photos, comes back with DIFFERENT
character errors each time. Tests that reuse byte-identical text for the
overlap would pass with an exact-match stitcher and tell us nothing about the
one we have.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import stitch  # noqa: E402

# A long Thai/English receipt, as three photos. Page 2 re-shoots the last two
# lines of page 1; page 3 re-shoots the last two lines of page 2. Repeated
# lines carry plausible OCR drift: 0/O, 1/l, dropped tone marks, spacing.
PAGE_1 = """BIG C SUPERCENTER
สาขา บางนา TAX ID 0107536000749
28/08/2026 14:22
นมยูเอชที ด.16          13.00
ชีสโรลไส้กรอก           26.00
มาม่าคัพ ต้มยำ          15.00"""

PAGE_2 = """ชีสโรลไส้กรอก           26.OO
มาม่าคัพ ตัมยำ          15.00
ขนมปังโฮลวีท            42.00
น้ำดื่มสิงห์ 6 ขวด       54.00
ไข่ไก่ เบอร์ 2          89.00"""

PAGE_3 = """น้ำดื่มสิงห์ 6 ขวด      54.00
ไข่ไก่ เบอร์ 2          89.OO
ถุงพลาสติก              2.00
รวม                    241.00
เงินสด                 300.00"""


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# The case this module exists for
# --------------------------------------------------------------------------

def test_three_overlapping_photos_keep_every_item_once():
    result = stitch.stitch([PAGE_1, PAGE_2, PAGE_3])
    body = result.text

    for item in ("ชีสโรลไส้กรอก", "ขนมปังโฮลวีท", "ถุงพลาสติก"):
        assert body.count(item) == 1, f"{item} survived more than once"

    # The overlapped lines appear once each, from whichever photo read them
    # first -- not twice.
    assert len([line for line in lines(body) if "มาม่าคัพ" in line]) == 1
    assert len([line for line in lines(body) if "ไข่ไก่" in line]) == 1

    # Nothing was lost: two seams of two lines each, removed from the 16
    # non-blank lines the three photos produced between them.
    assert result.lines_in == 16
    assert result.duplicate_lines_removed == 4
    assert result.lines_out == 12


def test_seams_are_reported_for_every_join():
    result = stitch.stitch([PAGE_1, PAGE_2, PAGE_3])
    assert [seam.page for seam in result.seams] == [2, 3]
    assert all(seam.overlap_lines == 2 for seam in result.seams)
    assert all(seam.score >= stitch.BLOCK_MATCH for seam in result.seams)


def test_order_is_preserved_so_a_split_item_rejoins():
    result = stitch.stitch([PAGE_1, PAGE_2, PAGE_3])
    body = lines(result.text)
    assert body.index([i for i in body if "ชีสโรล" in i][0]) < \
           body.index([i for i in body if "ถุงพลาสติก" in i][0])
    assert body[0].startswith("BIG C")
    assert body[-1].startswith("เงินสด")


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

def test_single_page_is_returned_unchanged():
    result = stitch.stitch([PAGE_1])
    assert result.text == "\n".join(lines(PAGE_1))
    assert result.seams == []
    assert result.duplicate_lines_removed == 0


def test_disjoint_photos_are_concatenated_not_truncated():
    """Someone who shoots the parts edge to edge, with no overlap at all."""
    tail = """ขนมปังโฮลวีท            42.00
รวม                     81.00"""
    result = stitch.stitch([PAGE_1, tail])
    assert result.seams == []
    assert result.duplicate_lines_removed == 0
    assert "ขนมปังโฮลวีท" in result.text
    assert "BIG C SUPERCENTER" in result.text


def test_the_same_photo_sent_twice_collapses_to_one():
    result = stitch.stitch([PAGE_1, PAGE_1])
    assert result.text == "\n".join(lines(PAGE_1))
    assert result.seams[0].overlap_lines == len(lines(PAGE_1))


def test_a_clipped_leading_line_does_not_defeat_the_seam():
    """The top of a photo usually cuts a line in half.

    A half-height line comes back from OCR as unreadable noise that matches
    nothing. A stitcher insisting the seam start at the new page's very first
    line would find no overlap here and duplicate the whole block; MAX_LEAD_SKIP
    lets the seam begin a line or two in, and the noise goes with it.
    """
    clipped = """~,, .-.
ชีสโรลไส้กรอก           26.00
มาม่าคัพ ตัมยำ          15.00
ขนมปังโฮลวีท            42.00"""
    result = stitch.stitch([PAGE_1, clipped])
    assert result.seams[0].overlap_lines == 3      # the noise line plus two dups
    assert result.text.count("26.00") == 1
    assert result.text.count("15.00") == 1
    assert "~,," not in result.text
    assert "ขนมปังโฮลวีท" in result.text


def test_blank_and_empty_pages_are_ignored():
    result = stitch.stitch(["", PAGE_1, "   \n\n  "])
    assert result.text == "\n".join(lines(PAGE_1))
    assert result.pages == 3


def test_empty_input():
    result = stitch.stitch([])
    assert result.text == ""
    assert result.pages == 0
    assert result.duplicate_lines_removed == 0


# --------------------------------------------------------------------------
# False seams -- the mistake that actually costs something
# --------------------------------------------------------------------------

def test_a_shared_boilerplate_line_is_not_an_overlap():
    """One line in common is not a seam.

    Both halves of this receipt end/start with a rule and a short word. A
    stitcher that accepts any single matching line would delete the second
    page's first real item along with it.
    """
    first = """7-ELEVEN
โค้ก                    20.00
------------"""
    second = """------------
ขนมปัง                  35.00
รวม                     55.00"""
    result = stitch.stitch([first, second])
    assert "ขนมปัง" in result.text
    assert "35.00" in result.text
    # The rule is short enough to fall under MIN_OVERLAP_CHARS, so no seam.
    assert result.seams == []


def test_two_different_items_at_the_boundary_are_both_kept():
    first = """LOTUS
น้ำปลา ตราปลาหมึก        45.00
ซีอิ๊วขาว ฉลากทอง       39.00"""
    second = """ซอสหอยนางรม แม่ครัว      52.00
น้ำมันพืช องุ่น         68.00"""
    result = stitch.stitch([first, second])
    assert result.seams == []
    assert result.lines_out == 5


def test_repeated_price_only_lines_do_not_trigger_a_seam():
    """A column of bare numbers is the worst case for fuzzy line matching."""
    first = "SHOP\n10.00\n20.00\n30.00"
    second = "40.00\n50.00\n60.00"
    result = stitch.stitch([first, second])
    assert result.lines_out == 7


# --------------------------------------------------------------------------
# Normalization and the disabled path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "นมยูเอชที ด.16   13.00",          # spacing differs between the two shots
    "นมยูเอชที ด.16<br>13.00",         # Surya's <br> artifact
    "นมยูเอชท่ี ด.16    13.00",        # tone mark misread
    "H UHT นมยูเอชที ด.16  13.00",     # one shot caught the leading SKU
])
def test_normalization_absorbs_ocr_noise(variant):
    a = stitch._normalize("นมยูเอชที ด.16          13.00")
    b = stitch._normalize(variant)
    assert stitch._similar(a, b) >= stitch.LINE_MATCH


def test_two_different_item_lines_score_below_the_line_threshold():
    """The other half of the same claim: LINE_MATCH must reject non-matches.

    These two share a shop, a price column and a Thai prefix, which is as
    close as two genuinely different lines get.
    """
    a = stitch._normalize("นมยูเอชที ด.16          13.00")
    b = stitch._normalize("นมเปรี้ยว ยาคูลท์       15.00")
    assert stitch._similar(a, b) < stitch.LINE_MATCH


def test_concatenate_keeps_the_duplicates():
    """ENGINE_STITCH_PAGES=0 -- the behaviour to compare the stitcher against."""
    result = stitch.concatenate([PAGE_1, PAGE_2, PAGE_3])
    assert result.lines_out == result.lines_in == 16
    assert result.seams == []
    assert result.text.count("42.00") == 1
    assert result.text.count("15.") == 2       # the duplicates survive
    assert result.text.count("89.") == 2


def test_window_bounds_the_search():
    """A window of 1 can only ever find a one-line overlap."""
    result = stitch.stitch([PAGE_1, PAGE_1], window=1)
    assert result.duplicate_lines_removed <= 1


def test_as_dict_shape():
    body = stitch.stitch([PAGE_1, PAGE_2]).as_dict()
    assert set(body) == {"pages", "duplicate_lines_removed", "seams"}
    assert set(body["seams"][0]) == {"page", "overlap_lines", "score"}
