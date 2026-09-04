"""Joining several photos of ONE receipt into a single OCR text.

A long receipt does not fit in one frame, so people photograph it in two or
three parts. Those parts almost always **overlap**: you finish the first shot
at some line, then start the second a little above it so nothing falls in the
gap. That is the right instinct -- a gap loses items silently, an overlap does
not -- but it means the naive fix (concatenate the pages' OCR text) hands the
model the same lines twice.

What that costs, concretely:

  * the model lists the duplicated items twice, so sum(items) exceeds the
    printed total and the receipt lands in `overcount`. The user is shown a
    basket they did not buy;
  * the duplicated text is paid for in prompt tokens, and the budget is a few
    thousand. On a three-photo receipt the overlap alone can be the difference
    between fitting and a 413.

So the seam is resolved HERE, deterministically, before the model is called --
the same division of labour task.md section 4 draws everywhere else:
arithmetic and text surgery in code, reading and correcting in the model.

The method is the standard sequence-assembly one: for each new page, find the
largest block of its leading lines that repeats the trailing lines of what we
have so far, and drop that block. Matching is fuzzy because the two photos
were taken from different distances and angles, so the *same* printed line
comes back from OCR with different character errors each time.

When no overlap is found the pages are simply concatenated, which is the
correct answer for someone who photographed the parts edge to edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# --------------------------------------------------------------------------
# Tuning. Every one of these exists to stop a FALSE seam, which is the only
# expensive mistake here: a missed overlap costs tokens and shows the user a
# duplicated line they can delete, while a false one deletes real items and
# nobody ever finds out.
# --------------------------------------------------------------------------

# Two OCR readings of the same printed line, from two photos, typically agree
# on most of their characters. 0.70 sits below that with room to spare, and
# well above what two DIFFERENT item lines from the same shop score -- they
# share a price format and often a word, so they are not at zero either.
LINE_MATCH = 0.70

# The block as a whole has to be a better match than any single line needs to
# be, so a few lucky pairings cannot carry a block that is mostly noise.
#
# 0.87 is measured, not guessed -- see reports/stitch_overlap.md. Over 699 real
# seams (the project's 20 receipts, split into 2-3 overlapping photos, with
# per-character OCR noise on the repeated lines) the WORST true seam scored
# 0.875. Over all 190 pairs of two DIFFERENT receipts, the best false seam
# reached 0.855 -- two 7-Eleven receipts from the same branch and member
# minutes apart, 13 of whose 15 lines really are identical boilerplate. 0.87
# sits in that gap: it kept every true seam and admitted none of the false
# ones. Do not lower it to catch a marginal seam; a missed seam shows the user
# a duplicated line they can delete, a false one silently deletes items.
BLOCK_MATCH = 0.87

# ...and most of its lines must match individually. A high mean produced by
# two perfect lines and four bad ones is not an overlap.
STRONG_FRACTION = 0.70

# Normalized characters the block must contain before it is believed. Without
# it, a lone "TOTAL" or "----------" at the end of page 1 that happens to also
# start page 2 reads as a one-line overlap.
MIN_OVERLAP_CHARS = 12

# The top of a photo very often clips a line in half, so page 2 can begin with
# a fragment that matches nothing. Allow the seam to start a line or two in;
# the skipped fragment is itself part of the overlap region and goes with it.
MAX_LEAD_SKIP = 2

# Longest overlap considered, in lines. Whole receipts in this project's own
# review set run 6-37 lines, so 60 comfortably covers "the second photo
# repeats all of the first" while keeping the search trivial.
DEFAULT_WINDOW = 60

# Normalization for COMPARISON only -- the text handed to the model is always
# the original. Whitespace goes entirely (the same line photographed twice
# comes back with different spacing and different <br> artifacts), as does
# punctuation apart from the decimal point, which carries real information.
_JUNK = re.compile("[^0-9a-z฀-๿.]")
_BR = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)


def _normalize(line: str) -> str:
    return _JUNK.sub("", _BR.sub(" ", line).lower())


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # autojunk pinned off: left on, it treats any character appearing in more
    # than 1% of a 200+ element sequence as junk, which for a line of digits
    # is most of the line. Our lines are short enough that it rarely fires,
    # but "rarely" is not a property to leave up to the input.
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass(frozen=True)
class Seam:
    """Where one page was joined onto the pages before it."""

    page: int            # 1-based index of the page being joined on
    overlap_lines: int   # lines of that page dropped as already-seen
    score: float         # mean similarity across the matched block

    def as_dict(self) -> dict:
        return {"page": self.page, "overlap_lines": self.overlap_lines,
                "score": round(self.score, 3)}


@dataclass(frozen=True)
class StitchResult:
    text: str                       # what to send to the model
    pages: int
    lines_in: int                   # non-blank lines across all pages
    lines_out: int                  # non-blank lines after de-duplication
    seams: list[Seam] = field(default_factory=list)

    @property
    def duplicate_lines_removed(self) -> int:
        return self.lines_in - self.lines_out

    def as_dict(self) -> dict:
        """The diagnostic block returned to the client.

        Worth surfacing rather than keeping internal: when a multi-photo
        receipt comes back wrong, the first question is always whether the
        seam was found, and this answers it without a re-run.
        """
        return {
            "pages": self.pages,
            "duplicate_lines_removed": self.duplicate_lines_removed,
            "seams": [s.as_dict() for s in self.seams],
        }


def _lines(page: str) -> list[str]:
    """Non-blank lines, original text preserved."""
    return [line for line in page.splitlines() if line.strip()]


def _find_overlap(tail: list[str], head: list[str],
                  window: int) -> tuple[int, int, float]:
    """Largest leading block of `head` that repeats the end of `tail`.

    Returns (skip, overlap_lines, score); the caller drops
    `head[:skip + overlap_lines]`. (0, 0, 0.0) means no overlap was found,
    i.e. plain concatenation.

    Largest-first, not best-scoring-first: where both a short block and a long
    one qualify, the long one is the real seam and the short one is merely its
    own suffix. Stopping at the first (largest) acceptable k also keeps the
    scan cheap on the common case of a big overlap.
    """
    for skip in range(0, MAX_LEAD_SKIP + 1):
        candidate = head[skip:]
        limit = min(len(tail), len(candidate), window)
        for k in range(limit, 0, -1):
            block = candidate[:k]
            if sum(len(_normalize(line)) for line in block) < MIN_OVERLAP_CHARS:
                continue
            sims = [_similar(_normalize(a), _normalize(b))
                    for a, b in zip(tail[-k:], block)]
            mean = sum(sims) / k
            strong = sum(1 for s in sims if s >= LINE_MATCH)
            if mean >= BLOCK_MATCH and strong >= STRONG_FRACTION * k:
                return skip, k, mean
    return 0, 0, 0.0


def stitch(pages, window: int = DEFAULT_WINDOW) -> StitchResult:
    """Join per-page OCR text, in CAPTURE ORDER, removing repeated lines.

    Order is load-bearing and is the caller's responsibility: the algorithm
    only ever looks for the start of page N inside the end of pages 1..N-1,
    because that is the only place an overlap can be when the photos were
    taken top to bottom. Shuffled pages produce a merged text with the items
    out of order -- the model will still read it, but the seam will not be
    found and the duplicates survive.
    """
    texts = list(pages)
    per_page = [_lines(t) for t in texts]
    lines_in = sum(len(page) for page in per_page)

    merged: list[str] = []
    seams: list[Seam] = []
    for index, lines in enumerate(per_page):
        if not lines:
            continue
        if not merged:
            merged = list(lines)
            continue
        skip, overlap, score = _find_overlap(merged, lines, window)
        if overlap:
            seams.append(Seam(page=index + 1, overlap_lines=skip + overlap,
                              score=score))
        merged.extend(lines[skip + overlap:])

    return StitchResult(
        text="\n".join(merged),
        pages=len(texts),
        lines_in=lines_in,
        lines_out=len(merged),
        seams=seams,
    )


def concatenate(pages) -> StitchResult:
    """Join without looking for overlaps -- what ENGINE_STITCH_PAGES=0 selects.

    A real code path rather than a branch inside stitch(), so both behaviours
    return the same shape and the endpoint needs no special case.
    """
    per_page = [_lines(text) for text in pages]
    merged = [line for page in per_page for line in page]
    return StitchResult(text="\n".join(merged), pages=len(per_page),
                        lines_in=len(merged), lines_out=len(merged))
