"""Unit tests for row-grouping/joining logic. No GPU or model needed --
operates on synthetic bounding boxes shaped like real Surya detections.
"""

from receipt_ocr.reconstruct import LineFragment, group_rows, join_row, reconstruct_text


def frag(text, x1, y1, x2, y2, confidence=0.95):
    return LineFragment(text=text, bbox=(x1, y1, x2, y2), confidence=confidence)


def test_single_row_no_split():
    fragments = [frag("Pad Thai", 10, 100, 120, 120)]
    rows = group_rows(fragments)
    assert len(rows) == 1
    assert len(rows[0]) == 1


def test_item_and_price_on_same_row_are_merged():
    # Item name on the left, price far to the right, same visual row --
    # this is the case Surya often splits into two line detections.
    name = frag("Pad Thai", 10, 100, 120, 122)
    price = frag("60.00", 300, 101, 350, 121)
    rows = group_rows([price, name])  # order shouldn't matter
    assert len(rows) == 1
    # group_rows only clusters into rows; left-to-right ordering is join_row's job.
    assert join_row(rows[0]) == "Pad Thai  60.00"


def test_two_separate_rows_stay_separate():
    row1 = frag("Pad Thai", 10, 100, 120, 122)
    row2 = frag("Tom Yum", 10, 140, 120, 162)
    rows = group_rows([row1, row2])
    assert len(rows) == 2


def test_rows_ordered_top_to_bottom():
    bottom = frag("Total", 10, 300, 60, 320)
    top = frag("Merchant", 10, 10, 90, 30)
    rows = group_rows([bottom, top])
    assert [r[0].text for r in rows] == ["Merchant", "Total"]


def test_join_row_uses_double_space_for_wide_gap():
    name = frag("Pad Thai", 0, 0, 80, 20)  # avg char width = 10
    price = frag("60.00", 400, 0, 450, 20)  # huge gap -> double space
    joined = join_row([price, name])
    assert joined == "Pad Thai  60.00"


def test_join_row_uses_single_space_for_normal_gap():
    word1 = frag("Thai", 0, 0, 40, 20)  # avg char width = 10
    word2 = frag("Tea", 45, 0, 75, 20)  # small gap -> single space
    joined = join_row([word1, word2])
    assert joined == "Thai Tea"


def test_reconstruct_text_end_to_end():
    fragments = [
        frag("Merchant Name", 10, 10, 200, 30),
        frag("Pad Thai", 10, 100, 120, 122),
        frag("60.00", 300, 101, 350, 121),
        frag("Tom Yum", 10, 140, 120, 162),
        frag("80.00", 300, 141, 350, 161),
        frag("Total", 10, 300, 60, 320),
        frag("140.00", 300, 301, 360, 321),
    ]
    text, rows = reconstruct_text(fragments)
    lines = text.split("\n")
    assert len(rows) == 4
    assert lines[0] == "Merchant Name"
    assert "Pad Thai" in lines[1] and "60.00" in lines[1]
    assert "Tom Yum" in lines[2] and "80.00" in lines[2]
    assert "Total" in lines[3] and "140.00" in lines[3]


def test_empty_input():
    assert group_rows([]) == []
    text, rows = reconstruct_text([])
    assert text == ""
    assert rows == []
