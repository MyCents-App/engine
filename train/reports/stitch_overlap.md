# Multi-photo overlap removal — measured behaviour

Evidence for the thresholds in `app/stitch.py`. Reproduce with:

```bash
.venv\Scripts\python.exe eval_stitch.py               # the tables below
.venv\Scripts\python.exe eval_stitch.py --thresholds  # the tuning sweep
```

Run 2026-09-04 against the 20 real receipts in `SuryaOCR/data/review/`
(`BLOCK_MATCH = 0.87`, `LINE_MATCH = 0.70`, `STRONG_FRACTION = 0.70`).

---

## What is being measured

A long receipt does not fit in one frame, so people photograph it in parts and
overlap the parts so nothing falls in the gap. `app/stitch.py` finds that
overlap and drops the repeated lines before the model is prompted.

There is **no multi-photo dataset yet** — nobody has photographed one receipt
in parts for us — so the recovery numbers use overlapping photos synthesized
from the real OCR text of the 20 review receipts: cut the line sequence into
2–3 pieces that repeat each other's edges, then re-noise the repeated lines,
because the entire difficulty is that one printed line read from two photos
comes back differently. Noise is per character at p=0.08, using Surya's real
confusions (0/O, 1/l, 5/S, 8/B, Thai tone marks), plus spacing drift and the
occasional dropped character.

**So the recovery half of this report is on simulated capture noise.** The
false-seam half is not simulated at all — it uses real receipts as they were
captured.

## Recovery — can a receipt shown in parts be put back together?

| photos | overlap | trials | recovered | dupes left | lines lost |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 95 | 95 | 0 | 0 |
| 2 | 4 | 90 | 90 | 0 | 0 |
| 2 | 6 | 75 | 75 | 0 | 0 |
| 3 | 2 | 90 | 89 | 1 | 0 |
| 3 | 3 | 85 | 85 | 0 | 0 |
| 3 | 5 | 45 | 45 | 0 | 0 |
| **all** | | **480** | **479** | **1** | **0** |

**99.8% exact reconstruction, and no line was ever lost.** The single miss
left one duplicated line in the prompt — the cheap failure.

Seam scores over the 699 seams found: min 0.875, median 0.969.

## False seams — the failure that actually costs something

The two ways to be wrong are not equally bad:

- a **missed** seam leaves a duplicated line in the prompt. The model lists the
  item twice, `reconciles` goes false, and the user deletes a line on the
  confirm screen. Visible and recoverable.
- a **false** seam deletes lines that were never duplicated. Items disappear
  from a receipt the user believes was read correctly, and nothing reports it.
  Invisible and unrecoverable.

So the thresholds are set to make the second impossible before making the first
rare. Every pair of two *different* receipts is checked; none may produce a
seam:

```
0 false seams across 190 unrelated receipt pairs
```

## Where BLOCK_MATCH came from

| threshold | true seams kept | false seams admitted |
|---:|---:|---:|
| 0.82 | 699/699 (100.0%) | 2 |
| 0.85 | 699/699 (100.0%) | 1 |
| 0.86 | 699/699 (100.0%) | 0 |
| **0.87** | **699/699 (100.0%)** | **0** |
| 0.88 | 698/699 (99.9%) | 0 |
| 0.90 | 694/699 (99.3%) | 0 |
| 0.92 | 664/699 (95.0%) | 0 |

The worst true seam scores **0.875**; the best false seam reaches **0.855**.
0.87 sits in that gap — it costs nothing measured and closes the only observed
false positive.

That false positive is worth knowing about. It is two 7-Eleven receipts from
the same branch, bought by the same member twelve minutes apart:

```
13 of the 15 lines are byte-identical boilerplate — the branch header, the
tax ID, the POS number, the member-services footer, the points table. Only
three lines differ, and those three are the actual purchase.
```

Nothing about the text says these are two receipts rather than two photos of
one. What rules it out is the contract — `/v1/extract` takes photos of *one*
receipt — not the algorithm. Two receipts posted as one will still merge.

## Control — what happens without stitching

`ENGINE_STITCH_PAGES=0` concatenates the pages instead:

```
72 duplicated lines would reach the model (20 receipts, 3 photos, 2-line overlap)
```

Those are lines the model is asked to extract twice, and prompt tokens paid for
twice against a budget of 3328.

## Caveats

- **Simulated capture noise.** Re-run this against real multi-photo captures as
  soon as there are any, and replace this section. Photographing the same line
  twice at different angles may distort it in ways a per-character confusion
  model does not reproduce — in particular, a line captured at a steep angle
  can lose or gain whole tokens.
- **Line-level, not word-level.** A line split *across* the seam (the top half
  of a character row in one frame, the bottom half in the next) is not
  reassembled; it survives as two junk lines, and correcting them is the
  model's job under task.md Rule 1.
- **Order is assumed.** The search only looks for the start of page N at the
  end of pages 1..N-1. Shuffled photos produce no seam and no de-duplication.
- **No post-model de-duplication.** Duplicate *items* are not removed from the
  model's output, because a receipt may legitimately print the same line twice
  and there is no way to tell that apart from a survived overlap. A survived
  overlap surfaces as `reconcileStatus: "overcount"`, which the confirm screen
  is already meant to flag.
