"""Score categorization checkpoints the way CATEGORIZATION.md section 4 asks.

SUPERSEDED, kept for its metric code
------------------------------------
This scores the OLD two-adapter task: shop + item names in, categories out. The
pipeline is now ONE joint task -- OCR text in, items WITH categories out (see
ANNOTATION.md, prompts.py) -- so this script cannot read the new data and its
numbers are not comparable to anything the joint model produces.

What is still good here: the Tally class and the five metrics, the per-`kind`
breakdown that keeps real receipts apart from synthetic (the only reason the
first run's overfitting was visible), and diagnose(), which is what made the
"11 entries for a 12-item basket" failure legible.

What the joint evaluator has to change: read {id, meta, input, target} records,
prompt with prompts.build_messages(input), and score BOTH halves of one output
-- extraction (shop, item names, prices, total, against eval_metrics.py's
measures) and categorization (c/s, with the metrics below) -- reported
separately, because a regression in either one has to be attributable.

Everything below describes the superseded task.

Generates greedily (do_sample=False, as every published number in this repo
does), runs each completion through categorize_prompts.parse_output -- the
same validator the engine will use at serving time -- and reports the five
metrics broken out by `kind`, with the real receipts kept separate from the
synthetic baskets.

    python eval_categorize.py \
      --checkpoints checkpoints/qwen3.5-2b-categorize/checkpoint-* \
      --val data/categorize_val.jsonl \
      --out reports/categorize_eval.md

Why the real receipts are reported on their own: 279 of the 294 validation
records are synthetic baskets drawn from the same 5,200 catalog products the
training baskets were drawn from, so a model can memorise its way to a good
pooled number while getting worse at receipts. Only the 15 rows of kind
"real" are user-confirmed items off actual receipts, and section 4 is explicit
that they decide whether this ships.

Loss is not what picks the checkpoint. A model can grow more accurate while
growing less confident, which raises eval loss; this script measures the thing
that is actually being bought.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import categorize_prompts

# Order the report's rows so the number that decides the ship sits last and
# alone. Anything not listed keeps its own name and sorts before "real".
KIND_ORDER = ["7-eleven-allonline", "makro-pro", "bigc", "watsons-th",
              "brand", "keyword", "real"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", default="Qwen/Qwen3.5-2B",
                   help="base model, used when a checkpoint is adapter-only")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="checkpoint directories; globs are expanded")
    p.add_argument("--val", type=Path, default=Path("data/categorize_val.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/categorize_eval.md"))
    p.add_argument("--batch", type=int, default=8,
                   help="receipts generated at once (default: %(default)s)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None,
                   help="cap records per kind, for a quick look")
    p.add_argument("--latency", action="store_true",
                   help="also measure per-receipt latency at batch size 1")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_val(path: Path, limit: int | None) -> list[dict]:
    """Each record becomes {kind, messages(prompt turns), gold(list of (c, s))}.

    Gold is read back through parse_output rather than json.loads so the
    reference and the prediction are held to exactly the same contract -- a
    gold label outside the taxonomy would otherwise score a correct prediction
    as wrong.
    """
    if not path.exists():
        sys.exit(f"error: no such file: {path}")

    per_kind: dict[str, int] = {}
    records: list[dict] = []
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        messages = raw["messages"]
        kind = (raw.get("meta") or {}).get("kind", "-")
        if limit is not None and per_kind.get(kind, 0) >= limit:
            continue

        expected = (raw.get("meta") or {}).get("n_items")
        gold = categorize_prompts.parse_output(messages[-1]["content"],
                                               expected or 0)
        if gold is None and expected:
            sys.exit(f"error: {path}:{lineno}: gold labels fail parse_output")
        if gold is None:
            data = json.loads(messages[-1]["content"])
            gold = [(e["c"], e.get("s")) for e in data["items"]]

        per_kind[kind] = per_kind.get(kind, 0) + 1
        records.append({"kind": kind, "prompt_messages": messages[:-1],
                        "gold": gold})

    if not records:
        sys.exit(f"error: {path} produced no records")
    return records


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

class Tally:
    """Counts for one `kind` (or for everything pooled)."""

    def __init__(self) -> None:
        self.receipts = 0
        self.valid = 0
        self.items = 0            # items in receipts that parsed
        self.cat_hits = 0
        self.gold_sub = 0         # items whose gold has a subcategory
        self.sub_hits = 0         # ...and the model got it right
        self.pred_sub = 0         # items where the model offered a subcategory
        self.pred_sub_hits = 0    # ...and it was right
        self.declines = 0         # gold has a subcategory, model said null

    def add(self, gold, pred) -> None:
        self.receipts += 1
        if pred is None:
            return
        self.valid += 1
        for (gold_c, gold_s), (pred_c, pred_s) in zip(gold, pred):
            self.items += 1
            if gold_c == pred_c:
                self.cat_hits += 1
            if gold_s is not None:
                self.gold_sub += 1
                if gold_s == pred_s:
                    self.sub_hits += 1
                elif pred_s is None:
                    self.declines += 1
            if pred_s is not None:
                self.pred_sub += 1
                if pred_s == gold_s:
                    self.pred_sub_hits += 1

    def row(self, name: str) -> str:
        def pct(num, den):
            return f"{num / den:.1%}" if den else "-"
        return (f"| {name} | {self.receipts} | {self.items} "
                f"| {pct(self.valid, self.receipts)} "
                f"| {pct(self.cat_hits, self.items)} "
                f"| {pct(self.sub_hits, self.gold_sub)} "
                f"| {pct(self.pred_sub_hits, self.pred_sub)} "
                f"| {pct(self.declines, self.gold_sub)} |")


def diagnose(completion: str, expected: int) -> str:
    """Say why parse_output rejected a completion.

    Worth the few lines: the interesting rejection is well-formed JSON with the
    wrong number of entries -- the model silently dropping an item off a long
    basket -- and that is indistinguishable from a truncated generation or a
    bad category unless the report names it.
    """
    try:
        start, end = completion.index("{"), completion.rindex("}") + 1
        data = json.loads(completion[start:end])
        entries = data["items"]
    except (ValueError, KeyError, TypeError):
        if "{" not in completion:
            return "no JSON emitted"
        return "malformed or truncated JSON"
    if not isinstance(entries, list):
        return "'items' is not a list"
    if len(entries) != expected:
        return f"emitted {len(entries)} entries for {expected} items"
    for entry in entries:
        if not isinstance(entry, dict):
            return "an entry is not an object"
        if entry.get("c") not in categorize_prompts.CATEGORIES:
            return f"unknown category {entry.get('c')!r}"
    return "rejected for an unknown reason"


def build_prompt(tokenizer, messages: list[dict]) -> str:
    """Byte-identical to app/extraction.py:build_prompt -- same template, same
    add_generation_prompt, same enable_thinking=False. If this drifts, every
    number here stops describing the served model."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_all(model, tokenizer, records, args) -> list[str]:
    """Greedy generation, left-padded so a batch's prompts end together."""
    import torch

    inner = getattr(tokenizer, "tokenizer", tokenizer)
    previous_side = inner.padding_side
    inner.padding_side = "left"   # decoder-only: the answer follows the prompt
    pad_id = inner.pad_token_id or inner.eos_token_id

    completions: list[str] = []
    try:
        for start in range(0, len(records), args.batch):
            chunk = records[start:start + args.batch]
            prompts = [build_prompt(tokenizer, r["prompt_messages"]) for r in chunk]
            batch = inner(text=prompts, return_tensors="pt",
                          padding=True).to(model.device)
            width = batch["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=pad_id)
            for row in out:
                completions.append(
                    inner.decode(row[width:], skip_special_tokens=True))
            done = min(start + args.batch, len(records))
            print(f"\r  generated {done}/{len(records)}", end="", flush=True)
    finally:
        inner.padding_side = previous_side
    print()
    return completions


def measure_latency(model, tokenizer, records, args) -> dict[int, float]:
    """Median seconds per receipt at batch size 1, for 1-, 5- and 12-item
    baskets. Section 4 budgets ~1s: the categorize call is on the user's
    critical path, and a batched throughput number would hide that."""
    import torch

    inner = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = inner.pad_token_id or inner.eos_token_id
    out: dict[int, float] = {}

    for target in (1, 5, 12):
        sized = [r for r in records if len(r["gold"]) == target][:5]
        if not sized:
            continue
        timings = []
        for record in sized:
            prompt = build_prompt(tokenizer, record["prompt_messages"])
            batch = inner(text=prompt, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            started = time.time()
            with torch.no_grad():
                model.generate(**batch, max_new_tokens=args.max_new_tokens,
                               do_sample=False, pad_token_id=pad_id)
            torch.cuda.synchronize()
            timings.append(time.time() - started)
        out[target] = statistics.median(timings)
    return out


def score(checkpoint: str, records, args) -> dict:
    from unsloth import FastLanguageModel  # noqa: I001
    import torch

    print(f"\n{checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint).replace("\\", "/"),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    started = time.time()
    completions = generate_all(model, tokenizer, records, args)
    elapsed = time.time() - started

    overall = Tally()
    by_kind: dict[str, Tally] = {}
    failures: list[dict] = []
    for record, completion in zip(records, completions):
        pred = categorize_prompts.parse_output(completion, len(record["gold"]))
        overall.add(record["gold"], pred)
        by_kind.setdefault(record["kind"], Tally()).add(record["gold"], pred)
        if pred is None and len(failures) < 8:
            failures.append({"kind": record["kind"],
                             "expected_items": len(record["gold"]),
                             "why": diagnose(completion, len(record["gold"])),
                             "completion": completion[:200]})

    latency = measure_latency(model, tokenizer, records, args) if args.latency else {}

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"checkpoint": str(checkpoint), "overall": overall,
            "by_kind": by_kind, "seconds": elapsed, "latency": latency,
            "failures": failures}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

HEADER = ("| kind | receipts | items | valid | cat acc | sub acc (given) "
          "| sub prec | decline |\n|---|---|---|---|---|---|---|---|")


def render(results: list[dict], records) -> str:
    lines = ["# Categorization eval", ""]
    lines.append(f"{len(records)} validation receipts, greedy decoding "
                 f"(`do_sample=False`), scored through "
                 f"`categorize_prompts.parse_output`.")
    lines.append("")
    lines.append("Targets from CATEGORIZATION.md section 4: valid >= 99%, "
                 "category accuracy >= 90% on synthetic and **the real-receipt "
                 "number is the one that decides**, sub accuracy >= 70%, "
                 "sub precision >= 90%.")
    lines.append("")

    lines.append("## Checkpoint comparison")
    lines.append("")
    lines.append("| checkpoint | valid | cat acc (all) | **cat acc (real)** "
                 "| sub prec | gen seconds |\n|---|---|---|---|---|---|")
    for result in results:
        overall, real = result["overall"], result["by_kind"].get("real")

        def pct(num, den):
            return f"{num / den:.1%}" if den else "-"
        real_cell = pct(real.cat_hits, real.items) if real else "-"
        lines.append(
            f"| {Path(result['checkpoint']).name} "
            f"| {pct(overall.valid, overall.receipts)} "
            f"| {pct(overall.cat_hits, overall.items)} "
            f"| **{real_cell}** "
            f"| {pct(overall.pred_sub_hits, overall.pred_sub)} "
            f"| {result['seconds']:.0f} |")
    lines.append("")

    for result in results:
        lines.append(f"## {Path(result['checkpoint']).name}")
        lines.append("")
        lines.append(HEADER)
        kinds = sorted(result["by_kind"],
                       key=lambda k: (KIND_ORDER.index(k) if k in KIND_ORDER
                                      else len(KIND_ORDER), k))
        for kind in kinds:
            label = f"**{kind}**" if kind == "real" else kind
            lines.append(result["by_kind"][kind].row(label))
        lines.append(result["overall"].row("_all_"))
        lines.append("")
        if result["latency"]:
            spread = ", ".join(f"{n} items: {s:.2f}s"
                               for n, s in sorted(result["latency"].items()))
            lines.append(f"Latency at batch 1 (median): {spread}")
            lines.append("")
        if result["failures"]:
            lines.append("Completions that failed `parse_output`:")
            lines.append("")
            for failure in result["failures"]:
                lines.append(f"- `{failure['kind']}`, "
                             f"{failure['expected_items']} items expected -- "
                             f"**{failure['why']}**")
            lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)

    checkpoints: list[str] = []
    for pattern in args.checkpoints:
        expanded = sorted(glob.glob(pattern))
        checkpoints.extend(expanded or [pattern])
    checkpoints = [c for c in checkpoints if Path(c, "adapter_config.json").exists()]
    if not checkpoints:
        sys.exit("error: no checkpoint directory with an adapter_config.json")
    # checkpoint-50 must sort before checkpoint-338; lexically it does not.
    checkpoints.sort(key=lambda c: (0, int(Path(c).name.split("-")[-1]))
                     if Path(c).name.split("-")[-1].isdigit() else (1, 0))

    records = load_val(args.val, args.limit)
    spread: dict[str, int] = {}
    for record in records:
        spread[record["kind"]] = spread.get(record["kind"], 0) + 1
    print(f"{len(records)} receipts: "
          + ", ".join(f"{k}={v}" for k, v in sorted(spread.items())))
    print(f"{len(checkpoints)} checkpoint(s) to score")

    results = [score(checkpoint, records, args) for checkpoint in checkpoints]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results, records), encoding="utf-8")
    print(f"\nwrote {args.out}")

    print("\n  checkpoint            valid   cat acc   cat acc (real)")
    for result in results:
        overall, real = result["overall"], result["by_kind"].get("real")

        def pct(num, den):
            return f"{num / den:.1%}" if den else "-"
        real_cell = pct(real.cat_hits, real.items) if real else "-"
        print(f"  {Path(result['checkpoint']).name:22s}"
              f"{pct(overall.valid, overall.receipts):>7s}"
              f"{pct(overall.cat_hits, overall.items):>10s}"
              f"{real_cell:>17s}")


if __name__ == "__main__":
    main()
