"""Reconstructs Surya's per-line detections into receipt-shaped rows.

Surya detects text line-by-line. On a receipt, an item name and its price are
usually printed on the same visual row but separated by a gap (whitespace or
dot leaders), so they can come back as two separate line detections. This
module re-merges lines that share a row and orders them left-to-right, so the
downstream extraction model sees "Pad Thai        60.00" as one row rather
than two disconnected fragments in an arbitrary order.

Pure geometry, no language assumptions -- works the same for Thai item names
and English/numeric prices on the same row.
"""

from __future__ import annotations

from dataclasses import dataclass

from receipt_ocr.config import settings


@dataclass
class LineFragment:
    text: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float

    @property
    def y_span(self) -> tuple[float, float]:
        return self.bbox[1], self.bbox[3]

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1e-6)

    @property
    def avg_char_width(self) -> float:
        length = max(len(self.text.strip()), 1)
        return max(self.bbox[2] - self.bbox[0], 1e-6) / length


def _y_overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Fraction of the shorter span that overlaps, in [0, 1]."""
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def group_rows(
    fragments: list[LineFragment],
    y_overlap_threshold: float | None = None,
) -> list[list[LineFragment]]:
    """Cluster fragments into rows using vertical bbox overlap.

    A running row tracks the union of its members' vertical span; a new
    fragment joins the current row if it overlaps that span enough,
    otherwise it starts a new row. Fragments are pre-sorted by vertical
    center so rows come out top-to-bottom.
    """
    if y_overlap_threshold is None:
        y_overlap_threshold = settings.row_y_overlap_threshold

    if not fragments:
        return []

    ordered = sorted(fragments, key=lambda f: (f.y_span[0] + f.y_span[1]) / 2)

    rows: list[list[LineFragment]] = [[ordered[0]]]
    row_spans: list[tuple[float, float]] = [ordered[0].y_span]

    for frag in ordered[1:]:
        span = row_spans[-1]
        if _y_overlap_ratio(frag.y_span, span) >= y_overlap_threshold:
            rows[-1].append(frag)
            lo, hi = span
            row_spans[-1] = (min(lo, frag.y_span[0]), max(hi, frag.y_span[1]))
        else:
            rows.append([frag])
            row_spans.append(frag.y_span)

    return rows


def join_row(row: list[LineFragment], gap_multiplier: float | None = None) -> str:
    """Join one row's fragments left-to-right into a single line of text.

    Uses a single space between normally-spaced fragments, and two spaces
    where the horizontal gap is wide enough to signal a name/price column
    gap (dot leaders, wide whitespace) -- keeps them visually distinguishable
    without inventing a delimiter that could confuse a downstream model.
    """
    if gap_multiplier is None:
        gap_multiplier = settings.row_gap_multiplier

    ordered = sorted(row, key=lambda f: f.x1)
    parts: list[str] = [ordered[0].text.strip()]

    for prev, curr in zip(ordered, ordered[1:]):
        gap = curr.x1 - prev.x2
        char_width = (prev.avg_char_width + curr.avg_char_width) / 2
        separator = "  " if gap > gap_multiplier * char_width else " "
        parts.append(separator)
        parts.append(curr.text.strip())

    return "".join(parts)


def reconstruct_text(
    fragments: list[LineFragment],
) -> tuple[str, list[list[LineFragment]]]:
    """Full pipeline: group into rows, join each row, stack rows top-down."""
    rows = group_rows(fragments)
    row_texts = [join_row(row) for row in rows]
    return "\n".join(row_texts), rows
