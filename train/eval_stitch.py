"""Measured evaluation of app/stitch.py -- multi-photo overlap removal.

The stitcher has two ways to be wrong and they are not equally bad:

  * a MISSED seam leaves duplicated lines in the prompt. The model lists an
    item twice, the receipt fails to reconcile, and the user deletes a line on
    the confirm screen. Visible, recoverable, annoying.
  * a FALSE seam deletes lines that were never duplicated. Items vanish from a
    receipt the user believes was read correctly, and nothing anywhere reports
    it. Invisible, unrecoverable.

So this measures both, separately, and the thresholds in stitch.py are set
from what it reports -- see reports/stitch_overlap.md for the numbers as of
2026-09-04.

There is no multi-photo dataset (nobody has yet photographed one receipt in
parts for us), so the overlapping photos are SYNTHESIZED from the real OCR
text of the 20 review receipts: cut the line sequence into 2-3 pieces that
repeat each other's edges, then re-noise the repeated lines, since the whole
difficulty is that the same printed line read from two photos comes back
differently. A run therefore answers "does the algorithm recover a receipt it
has been shown in overlapping parts", which is the question, but on simulated
rather than observed capture noise. Replace `receipts()` with real multi-photo
captures when there are any.

The false-seam half needs no simulation at all: it pairs every two DIFFERENT
receipts and asserts the stitcher finds no seam between them. Receipts from
one convenience-store chain share a great deal of boilerplate, which is the
hardest case there is for this and is entirely real.

    .venv\\Scripts\\python.exe eval_stitch.py
    .venv\\Scripts\\python.exe eval_stitch.py --thresholds   # tuning sweep

Stdlib only, and it exits non-zero if a line is ever lost or a false seam is
ever found -- both are regressions, not numbers to note and move past.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import stitch  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW = ROOT.parent / "SuryaOCR" / "data" / "review"

# How the same printed line differs between two photos of it. Per CHARACTER,
# with a low probability -- not per line. Applying a confusion table to every
# digit at once turns a 28-digit transaction ID into an unrelated string,
# which is a simulator artifact and not something Surya does.
P_CHAR = 0.08
CONFUSIONS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
              "8": "B", "B": "8", "6": "b", "2": "Z",
              "ำ": "ั", "ี": "ิ",
              "ื": "ั", "่": "้"}

# (photos, lines each repeats of the one before). Two photos with a two-line
# overlap is the common case; three with five is someone being careful.
SHAPES = ((2, 2), (2, 4), (2, 6), (3, 2), (3, 3), (3, 5))
DRAWS = 5  # independent noise draws per receipt per shape

RULE = "=" * 78


def perturb(line: str, rng: random.Random) -> str:
    out = "".join(CONFUSIONS.get(c, c) if rng.random() < P_CHAR else c for c in line)
    if rng.random() < 0.5:                    # spacing drifts between shots
        out = " ".join(out.split())
    if rng.random() < 0.2 and len(out) > 6:   # an occasional dropped character
        cut = rng.randrange(len(out))
        out = out[:cut] + out[cut + 1:]
    return out


def as_photos(lines: list[str], photos: int, overlap: int,
              rng: random.Random) -> list[str]:
    """One receipt's lines, re-cut into `photos` overlapping frames."""
    step = max(1, (len(lines) - overlap) // photos)
    pages, start = [], 0
    for index in range(photos):
        end = len(lines) if index == photos - 1 else min(len(lines), start + step + overlap)
        block = lines[start:end]
        if index > 0:
            block = [perturb(x, rng) for x in block[:overlap]] + block[overlap:]
        pages.append("\n".join(block))
        start = max(0, end - overlap)
    return pages


def receipts(review: Path) -> list[tuple[str, list[str]]]:
    out = []
    for path in sorted(review.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        text = "".join(data.get("ocrTexts") or [])
        lines = [x for x in text.splitlines() if x.strip()]
        if lines:
            out.append((path.stem, lines))
    return out


def recovery(data, verbose: bool) -> tuple[list[tuple], int, list[float]]:
    """Can a receipt shown in overlapping parts be put back together?"""
    rows, lost_total, scores = [], 0, []
    for photos, overlap in SHAPES:
        exact = dupes = lost = trials = 0
        for draw in range(DRAWS):
            rng = random.Random(1000 * draw + photos * 10 + overlap)
            for stem, lines in data:
                if len(lines) < photos * (overlap + 2):
                    continue        # too short to cut this many ways
                trials += 1
                result = stitch.stitch(as_photos(lines, photos, overlap, rng))
                scores.extend(seam.score for seam in result.seams)
                got = [x for x in result.text.splitlines() if x.strip()]
                if len(got) == len(lines):
                    exact += 1
                elif len(got) > len(lines):
                    dupes += 1
                    if verbose:
                        print(f"  missed seam  {stem:10} {photos}p/{overlap}o "
                              f"{len(lines)} -> {len(got)}")
                else:
                    lost += 1
                    lost_total += 1
                    print(f"  LOST LINES   {stem:10} {photos}p/{overlap}o "
                          f"{len(lines)} -> {len(got)}")
        rows.append((photos, overlap, trials, exact, dupes, lost))
    return rows, lost_total, scores


def false_seams(data, verbose: bool) -> list[tuple]:
    """Two DIFFERENT receipts must never look like two halves of one."""
    found = []
    for index, (stem_a, a) in enumerate(data):
        for stem_b, b in data[index + 1:]:
            result = stitch.stitch(["\n".join(a), "\n".join(b)])
            if result.seams:
                found.append((stem_a, stem_b, result.seams[0]))
                print(f"  FALSE SEAM   {stem_a} + {stem_b}: "
                      f"{result.seams[0].as_dict()}")
    return found


def sweep(data) -> None:
    """The tuning table BLOCK_MATCH was chosen from."""
    true_scores: list[float] = []
    for photos, overlap in SHAPES:
        for draw in range(DRAWS):
            rng = random.Random(1000 * draw + photos * 10 + overlap)
            for _stem, lines in data:
                if len(lines) < photos * (overlap + 2):
                    continue
                for seam in stitch.stitch(as_photos(lines, photos, overlap, rng)).seams:
                    true_scores.append(seam.score)

    # The best score an unrelated pair reaches with the threshold lifted --
    # i.e. how close a false seam can get before BLOCK_MATCH stops it.
    worst: list[float] = []
    for index, (_a_stem, a) in enumerate(data):
        for _b_stem, b in data[index + 1:]:
            best = 0.0
            for k in range(min(len(a), len(b), stitch.DEFAULT_WINDOW), 0, -1):
                sims = [stitch._similar(stitch._normalize(x), stitch._normalize(y))
                        for x, y in zip(a[-k:], b[:k])]
                mean = sum(sims) / k
                strong = sum(1 for s in sims if s >= stitch.LINE_MATCH)
                if strong >= stitch.STRONG_FRACTION * k and mean > best:
                    best = mean
            if best:
                worst.append(best)

    print(f"\ntrue seam scores   n={len(true_scores)}  min={min(true_scores):.3f}  "
          f"median={statistics.median(true_scores):.3f}")
    print(f"false seam scores  n={len(worst)}  max={max(worst):.3f}" if worst
          else "false seam scores  none reachable at any threshold")
    print(f"\n{'threshold':>10} {'true seams kept':>18} {'false seams admitted':>22}")
    for bar in (0.82, 0.85, 0.86, 0.87, 0.88, 0.90, 0.92):
        kept = sum(1 for s in true_scores if s >= bar)
        bad = sum(1 for s in worst if s >= bar)
        mark = "  <-- BLOCK_MATCH" if abs(bar - stitch.BLOCK_MATCH) < 1e-9 else ""
        print(f"{bar:>10.2f} {kept:>10}/{len(true_scores)} "
              f"({100 * kept / len(true_scores):5.1f}%) {bad:>13}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", type=Path, default=DEFAULT_REVIEW,
                    help="folder of review_photos.py output (default: %(default)s)")
    ap.add_argument("--thresholds", action="store_true",
                    help="print the BLOCK_MATCH tuning sweep and stop")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="name every receipt whose seam was missed")
    args = ap.parse_args()

    data = receipts(args.review)
    if not data:
        print(f"No receipts in {args.review}. Run review_photos.py first.")
        return 2
    print(f"{len(data)} receipts from {args.review}\n")

    if args.thresholds:
        sweep(data)
        return 0

    print(RULE)
    print("RECOVERY -- one receipt, shown as overlapping photos")
    print(RULE)
    rows, lost, scores = recovery(data, args.verbose)
    print(f"\n{'photos':>7} {'overlap':>8} {'trials':>7} {'recovered':>10} "
          f"{'dupes left':>11} {'lines lost':>11}")
    totals = [0, 0, 0, 0]
    for photos, overlap, trials, exact, dupes, lost_here in rows:
        print(f"{photos:>7} {overlap:>8} {trials:>7} {exact:>10} {dupes:>11} "
              f"{lost_here:>11}")
        totals = [totals[0] + trials, totals[1] + exact,
                  totals[2] + dupes, totals[3] + lost_here]
    print(f"{'all':>7} {'':>8} {totals[0]:>7} {totals[1]:>10} {totals[2]:>11} "
          f"{totals[3]:>11}   ({100 * totals[1] / totals[0]:.1f}% exact)")
    if scores:
        print(f"\nseam scores: min {min(scores):.3f}, median "
              f"{statistics.median(scores):.3f}  (BLOCK_MATCH={stitch.BLOCK_MATCH})")

    print(f"\n{RULE}")
    print("FALSE SEAMS -- two different receipts must not look like one")
    print(RULE)
    pairs = len(data) * (len(data) - 1) // 2
    bad = false_seams(data, args.verbose)
    print(f"  {len(bad)} false seams across {pairs} unrelated receipt pairs")

    print(f"\n{RULE}")
    print("CONTROL -- ENGINE_STITCH_PAGES=0, i.e. plain concatenation")
    print(RULE)
    rng = random.Random(7)
    duplicated = 0
    for _stem, lines in data:
        if len(lines) < 12:
            continue
        pages = as_photos(lines, 3, 2, rng)
        duplicated += len(stitch.concatenate(pages).text.splitlines()) - len(lines)
    print(f"  {duplicated} duplicated lines would reach the model "
          f"(3 photos, 2-line overlap)")

    if lost or bad:
        print("\nFAIL: lines were lost or a false seam was found.")
        return 1
    print("\nOK: no lines lost, no false seams.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
