"""Score joint checkpoints: OCR text in, items with prices AND categories out.

    .venv\\Scripts\\python.exe eval_joint.py \\
        --checkpoints checkpoints/qwen3.5-2b-joint/checkpoint-* \\
        --legacy checkpoints/qwen3.5-2b-qlora/checkpoint-550

Reads data/val.jsonl ({id, meta, input, target}, written by build_dataset.py),
generates greedily (do_sample=False, as every published number in this repo
does) and scores the two halves of each answer separately, because a
regression in either one has to be attributable (ANNOTATION.md section 7):

  extraction      eval_metrics.score_example on the answer with c/s stripped --
                  money exact, count, total, price recall, name quality, shop.
                  The same measures behind checkpoint-550's published numbers.
  categorization  c/s scored over items PAIRED to the gold items by price and
                  name, never by position. A model that drops item 3 of 12
                  would otherwise be marked wrong on items 3-11 by a shift,
                  and the category numbers would describe the count error.

--legacy scores checkpoint-550 on the same receipts with the prompt it was
trained on (prompts_legacy.py). Its extraction numbers are the bar the joint
model must clear -- section 7 keeps two adapters as the fallback if it cannot.

But checkpoint-550 was trained on an earlier labelling of many of these same
photos (38 of the 73 validation receipts: it scores 95% money-exact on those
and 63% on the rest). Pass --legacy-train with that training file,

    --legacy-train D:/Documents/train/data/real_train.jsonl

and the report adds every row again on only the receipts the legacy
checkpoint never saw. That second table is the fair comparison.

Completions are cached per checkpoint under --cache-dir, keyed by a hash of
each prompt. Fixing a validation LABEL after the hand-check changes no prompt,
so a re-run rescores the saved completions without touching the GPU; changing
the OCR text or the prompt regenerates exactly the records it affects.

Loss does not pick the checkpoint -- a model can grow more accurate while
growing less confident. This measures what is actually being bought.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from rapidfuzz import fuzz

import eval_metrics
import postprocess
import prompts
import prompts_legacy
from categorize_prompts import CATEGORIES

ROOT = Path(__file__).resolve().parent
TAX_NAMES = {"vat", "tax"}
NAME_THRESHOLD = eval_metrics.DEFAULT_NAME_SIM_THRESHOLD


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="joint checkpoints (globs expanded)")
    p.add_argument("--legacy", nargs="*", default=[],
                   help="extraction-only checkpoints, prompted with prompts_legacy.py "
                        "and scored on extraction only")
    p.add_argument("--val", type=Path, default=ROOT / "data" / "val.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "joint_eval.md")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "reports" / "joint_eval")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="the longest gold answer is ~860 tokens; serving's 768 "
                        "would truncate it (default: %(default)s)")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="first N records only")
    p.add_argument("--regenerate", action="store_true",
                   help="ignore cached completions")
    p.add_argument("--legacy-train", type=Path, default=None,
                   help="the real-receipt JSONL the --legacy checkpoint was trained "
                        "on. Receipts found in it are split out, and every row is "
                        "also reported on the rest: a checkpoint scored on its own "
                        "training receipts is not a bar anything else can be held to")
    p.add_argument("--normalize", action="store_true",
                   help="apply postprocess.normalize_prediction before scoring -- "
                        "the deployed pipeline's view rather than the raw model's")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Scoring -- pure functions, no model, covered by tests/test_eval_joint.py
# --------------------------------------------------------------------------

def is_tax(item) -> bool:
    return isinstance(item, dict) and str(item.get("name", "")).strip().lower() in TAX_NAMES


def strip_categories(target: dict) -> dict:
    """The extraction half: the contract checkpoint-550 was scored against."""
    out = {k: v for k, v in target.items() if k != "items"}
    items = target.get("items")
    if isinstance(items, list):
        out["items"] = [{"name": i.get("name"), "price": i.get("price")}
                        if isinstance(i, dict) else i for i in items]
    ordered = {k: out[k] for k in ("shop_name", "items", "basket-wide_discount",
                                   "total_price") if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return ordered


def pair_items(gold: list, pred: list, threshold: float = NAME_THRESHOLD):
    """[(gold_index, pred_index)]: same price first, most similar name winning;
    then whatever is left, by name alone above the threshold.

    Price leads because it is the part that is either right or wrong. The
    second pass catches an item whose price the model misread but whose name
    it got -- its category is still worth scoring.
    """
    def name(item):
        return " ".join(str(item.get("name", "")).lower().split())

    def price(item):
        return postprocess.parse_price_strict(item.get("price"))

    gold_ok = [(i, g) for i, g in enumerate(gold) if isinstance(g, dict)]
    pred_ok = [(j, p) for j, p in enumerate(pred) if isinstance(p, dict)]
    pairs, used_g, used_p = [], set(), set()

    candidates = []
    for i, g in gold_ok:
        for j, p in pred_ok:
            gp, pp = price(g), price(p)
            if gp is not None and pp is not None and abs(gp - pp) < 1e-6:
                candidates.append((fuzz.ratio(name(g), name(p)), i, j))
    for _, i, j in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
        if i not in used_g and j not in used_p:
            pairs.append((i, j)); used_g.add(i); used_p.add(j)

    candidates = []
    for i, g in gold_ok:
        if i in used_g:
            continue
        for j, p in pred_ok:
            if j in used_p:
                continue
            sim = fuzz.ratio(name(g), name(p))
            if sim >= threshold:
                candidates.append((sim, i, j))
    for _, i, j in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
        if i not in used_g and j not in used_p:
            pairs.append((i, j)); used_g.add(i); used_p.add(j)

    return sorted(pairs)


class CategoryTally:
    """Category counts for one group of receipts. Tax rows are kept apart:
    they must carry c null, and counting them as a correct 'null' would flatter
    the accuracy of the items that matter."""

    def __init__(self) -> None:
        self.receipts = self.parsed = 0
        self.gold_items = self.paired = 0
        self.cat_hits = 0
        self.gold_sub = self.sub_hits = self.declines = 0
        self.pred_sub = self.pred_sub_hits = 0
        self.pred_sub_labelled = self.pred_sub_labelled_hits = 0
        self.pred_items = self.invalid = 0
        self.tax_rows = self.tax_clean = 0
        self.confusion: Counter = Counter()
        self.per_category: dict = defaultdict(lambda: [0, 0])   # gold c -> [hits, n]

    def add(self, gold_items: list, pred_items: list | None) -> None:
        self.receipts += 1
        goods = [g for g in gold_items if not is_tax(g)]
        self.gold_items += len(goods)
        if pred_items is None:
            return
        self.parsed += 1

        for p in pred_items:
            if not isinstance(p, dict):
                continue
            if is_tax(p):
                self.tax_rows += 1
                self.tax_clean += p.get("c") is None and p.get("s") is None
                continue
            self.pred_items += 1
            c, s = p.get("c"), p.get("s")
            valid = (c is None and s is None) or (
                c in CATEGORIES and (s is None or s in CATEGORIES[c]))
            self.invalid += not valid

        gold_goods = [g for g in gold_items if not is_tax(g)]
        pred_goods = [p for p in pred_items if not is_tax(p)]
        for gi, pj in pair_items(gold_goods, pred_goods):
            g, p = gold_goods[gi], pred_goods[pj]
            self.paired += 1
            gc_, pc = g.get("c"), p.get("c")
            self.per_category[gc_][1] += 1
            if gc_ == pc:
                self.cat_hits += 1
                self.per_category[gc_][0] += 1
            else:
                self.confusion[(gc_, pc)] += 1
            gs, ps = g.get("s"), p.get("s")
            if gs is not None:
                self.gold_sub += 1
                if ps == gs:
                    self.sub_hits += 1
                elif ps is None:
                    self.declines += 1
            if ps is not None:
                self.pred_sub += 1
                self.pred_sub_hits += (ps == gs and pc == gc_)
                if gs is not None:
                    self.pred_sub_labelled += 1
                    self.pred_sub_labelled_hits += (ps == gs and pc == gc_)

    @staticmethod
    def _rate(num, den):
        return num / den if den else None

    def summary(self) -> dict:
        r = self._rate
        return {
            "paired_share": r(self.paired, self.gold_items),
            "cat_acc": r(self.cat_hits, self.paired),
            "sub_acc_given": r(self.sub_hits, self.gold_sub),
            "sub_precision": r(self.pred_sub_hits, self.pred_sub),
            # Precision where the gold label HAS a subcategory. The plain figure
            # also counts every subcategory offered where the label left it
            # null -- partly model guessing, partly labels left incomplete --
            # so the gap between the two is how much the labels decide it.
            "sub_precision_labelled": r(self.pred_sub_labelled_hits, self.pred_sub_labelled),
            "decline_rate": r(self.declines, self.gold_sub),
            "taxonomy_valid": r(self.pred_items - self.invalid, self.pred_items),
            "tax_row_clean": r(self.tax_clean, self.tax_rows),
        }


def score_record(record: dict, completion: str, legacy: bool,
                 normalize: bool = False) -> tuple[eval_metrics.ExampleScore, list | None, str]:
    """(extraction score, predicted items or None, why money_exact failed or '')."""
    gold = record["target"]
    pred, _ = postprocess.extract_json(completion)
    if normalize and pred is not None:
        pred = postprocess.normalize_prediction(pred)

    # Score the extraction half on c/s-stripped JSON on both sides, so the
    # numbers mean what they meant for checkpoint-550 and a category never
    # decides whether the money was right.
    if isinstance(pred, dict):
        raw = json.dumps(strip_categories(pred), ensure_ascii=False)
    else:
        raw = completion
    score = eval_metrics.score_example(raw, strip_categories(gold), record["input"])

    items = pred.get("items") if isinstance(pred, dict) else None
    items = items if isinstance(items, list) else None

    why = ""
    if not score.json_valid:
        why = "unparseable"
    elif not score.money_exact:
        gold_n, pred_n = len(gold["items"]), len(items or [])
        if pred_n != gold_n:
            why = f"{pred_n} items for {gold_n}"
        elif not score.price_multiset_exact:
            why = "a price is wrong"
        else:
            why = "total is wrong"
    return score, (None if legacy else items), why


# --------------------------------------------------------------------------
# Generation, with a per-prompt cache
# --------------------------------------------------------------------------

def build_messages(record: dict, legacy: bool) -> list[dict]:
    return (prompts_legacy if legacy else prompts).build_messages(record["input"])


def prompt_key(messages: list[dict]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, checkpoint: str, legacy: bool) -> Path:
    parts = Path(checkpoint).parts[-2:]
    name = "__".join(parts) + ("__legacy" if legacy else "")
    return cache_dir / f"{name}.completions.jsonl"


def load_cache(path: Path, max_new_tokens: int) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("max_new_tokens") == max_new_tokens:
            out[row["key"]] = row
    return out


def apply_template(tokenizer, messages: list[dict]) -> str:
    """Byte-identical to app/extraction.py:build_prompt."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


def generate(checkpoint: str, prompts_by_key: dict[str, list], args) -> dict[str, dict]:
    """Greedy completions for the given prompts, longest first so a batch pads
    little and an out-of-memory shows up at once rather than at the end."""
    from unsloth import FastLanguageModel  # noqa: I001
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint).replace("\\", "/"),
        max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    previous_side, inner.padding_side = inner.padding_side, "left"
    pad_id = inner.pad_token_id or inner.eos_token_id
    eos_ids = {inner.eos_token_id, inner.convert_tokens_to_ids("<|im_end|>")}

    texts = {k: apply_template(tokenizer, m) for k, m in prompts_by_key.items()}
    order = sorted(texts, key=lambda k: -len(texts[k]))
    out: dict[str, dict] = {}
    started = time.time()
    try:
        for start in range(0, len(order), args.batch):
            keys = order[start:start + args.batch]
            batch = inner(text=[texts[k] for k in keys], return_tensors="pt",
                          padding=True).to(model.device)
            width = batch["input_ids"].shape[1]
            with torch.no_grad():
                generated = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=pad_id)
            for key, output in zip(keys, generated):
                new = output[width:].tolist()
                # A finished row is padded out to the batch's longest; count up
                # to its first end token. No end token at all = cut off.
                end = next((i for i, t in enumerate(new) if t in eos_ids), None)
                out[key] = {"key": key, "max_new_tokens": args.max_new_tokens,
                            "completion": inner.decode(new[:end], skip_special_tokens=True),
                            "tokens": len(new) if end is None else end + 1,
                            "truncated": end is None}
            print(f"\r  generated {min(start + args.batch, len(order))}/{len(order)}",
                  end="", flush=True)
    finally:
        inner.padding_side = previous_side
        print()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seconds = time.time() - started
    for row in out.values():
        row["batch_seconds_per_receipt"] = seconds / max(1, len(out))
    return out


# --------------------------------------------------------------------------
# One checkpoint
# --------------------------------------------------------------------------

def evaluate(checkpoint: str, legacy: bool, records: list[dict], args) -> dict:
    label = "/".join(Path(checkpoint).parts[-2:]) + (" (legacy prompt)" if legacy else "")
    print(f"\n{label}")
    messages = {r["id"]: build_messages(r, legacy) for r in records}
    keys = {rid: prompt_key(m) for rid, m in messages.items()}

    path = cache_path(args.cache_dir, checkpoint, legacy)
    cache = {} if args.regenerate else load_cache(path, args.max_new_tokens)
    missing = {keys[rid]: messages[rid] for rid in messages if keys[rid] not in cache}
    if missing:
        print(f"  {len(missing)} to generate, {len(records) - len(missing)} cached")
        cache.update(generate(checkpoint, missing, args))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in cache.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        print(f"  all {len(records)} cached -- rescoring only")

    scores: list[eval_metrics.ExampleScore] = []
    by_kind: dict[str, list] = defaultdict(list)
    cats_all, cats_by_kind = CategoryTally(), defaultdict(CategoryTally)
    misses, examples, per_record = [], [], []
    truncated = 0
    for record in records:
        row = cache[keys[record["id"]]]
        score, items, why = score_record(record, row["completion"], legacy, args.normalize)
        kind = record.get("meta", {}).get("kind", "-")
        scores.append(score)
        by_kind[kind].append(score)
        per_record.append((record["id"], score, items, record["target"]["items"]))
        truncated += bool(row.get("truncated"))
        if not legacy:
            cats_all.add(record["target"]["items"], items)
            cats_by_kind[kind].add(record["target"]["items"], items)
        if why:
            misses.append((record["id"], why, record["target"]["shop_name"]))
        examples.append({"id": record["id"], "kind": kind, "why": why,
                         "truncated": row.get("truncated"),
                         "completion": row["completion"], **asdict(score)})

    with (args.cache_dir / f"{path.stem.replace('.completions', '')}_examples.jsonl"
          ).open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    seconds = [cache[keys[r["id"]]].get("batch_seconds_per_receipt") for r in records]
    seconds = [s for s in seconds if s is not None]
    return {"label": label, "legacy": legacy,
            "extraction": eval_metrics.aggregate(scores),
            "extraction_by_kind": {k: eval_metrics.aggregate(v) for k, v in by_kind.items()},
            "categories": None if legacy else cats_all,
            "categories_by_kind": None if legacy else dict(cats_by_kind),
            "truncated": truncated, "misses": misses, "per_record": per_record,
            "seconds": sum(seconds) / len(seconds) if seconds else None}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


COLUMNS = ("| checkpoint | MONEY EXACT | count exact | total exact | price recall "
           "| shop exact | name ok | JSON valid | truncated | **cat acc** | sub acc "
           "| sub prec | sub prec (labelled) | taxonomy valid | items paired | s/receipt |")


def row(result: dict, ext: dict, cats) -> str:
    c = cats.summary() if cats else {}
    seconds = "-" if result["seconds"] is None else f"{result['seconds']:.2f}"
    return (f"| {result['label']} | **{pct(ext.get('money_exact_rate'))}** "
            f"| {pct(ext.get('item_count_exact_rate'))} "
            f"| {pct(ext.get('total_price_exact_rate'))} "
            f"| {pct(ext.get('item_price_only_recall_mean'))} "
            f"| {pct(ext.get('shop_name_exact_rate'))} "
            f"| {pct(ext.get('name_acceptable_rate'))} "
            f"| {pct(ext.get('json_validity_rate'))} "
            f"| {result['truncated']} "
            f"| **{pct(c.get('cat_acc'))}** | {pct(c.get('sub_acc_given'))} "
            f"| {pct(c.get('sub_precision'))} | {pct(c.get('sub_precision_labelled'))} "
            f"| {pct(c.get('taxonomy_valid'))} "
            f"| {pct(c.get('paired_share'))} "
            f"| {seconds} |")


def render(results: list[dict], records: list[dict], args) -> str:
    kinds = Counter(r.get("meta", {}).get("kind", "-") for r in records)
    sep = "|---" * (COLUMNS.count("|") - 1) + "|"
    lines = ["# Joint model evaluation", "",
             f"{len(records)} receipts from `{args.val.name}` "
             f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}), greedy "
             f"decoding, max_new_tokens={args.max_new_tokens}"
             f"{', post-processed (--normalize)' if args.normalize else ', raw model output'}.",
             "",
             "Extraction columns are `eval_metrics.py`'s, on the answer with c/s "
             "stripped, so they compare directly with checkpoint-550's. Category "
             "columns score only items paired to a gold item by price and name "
             "(`items paired` is that share): a dropped item costs money-exact, "
             "not the categories of every item after it.",
             "", "Targets (CATEGORIZATION.md section 4): category accuracy >= 90%, "
             "sub accuracy >= 70%, **sub precision >= 90%** -- a wrong subcategory "
             "is worse than null. Extraction must not fall below the legacy row.",
             "", "## Checkpoints", "", COLUMNS, sep]
    for result in results:
        lines.append(row(result, result["extraction"], result["categories"]))
    joint = [r for r in results if not r["legacy"]]
    if joint:
        best_money = max(joint, key=lambda r: r["extraction"].get("money_exact_rate") or 0)
        best_cat = max(joint, key=lambda r: r["categories"].summary().get("cat_acc") or 0)
        lines += ["", f"Best money exact: **{best_money['label']}**. "
                      f"Best category accuracy: **{best_cat['label']}**."]

    seen = getattr(args, "seen", None)
    if seen:
        unseen = {r["id"] for r in records} - seen
        lines += ["", f"## On the {len(unseen)} receipts the legacy checkpoint never "
                      f"trained on", "",
                  f"{len(seen)} of these {len(records)} receipts are in "
                  f"`{args.legacy_train.name}`, which the --legacy checkpoint was "
                  f"trained on (matched by total and price multiset). Its row above "
                  f"is inflated by them; **this table is the fair comparison.**",
                  "", COLUMNS, sep]
        for result in results:
            ext, cats = subset(result, unseen)
            lines.append(row(result, ext, cats))
        if joint:
            fair = max(joint, key=lambda r: subset(r, unseen)[0].get("money_exact_rate") or 0)
            lines += ["", f"Best money exact here: **{fair['label']}**."]

    for result in results:
        lines += ["", f"## {result['label']}", ""]
        if len(result["extraction_by_kind"]) > 1:
            lines += ["By kind:", "", COLUMNS, sep]
            for kind, ext in sorted(result["extraction_by_kind"].items()):
                cats = (result["categories_by_kind"] or {}).get(kind)
                lines.append(row({**result, "label": kind}, ext, cats))
            lines.append("")
        cats = result["categories"]
        if cats:
            lines += ["| gold category | items | accuracy |", "|---|---|---|"]
            for category in list(CATEGORIES) + [None]:
                hits, n = cats.per_category.get(category, (0, 0))
                if n:
                    lines.append(f"| {category or 'null'} | {n} | {pct(hits / n)} |")
            if cats.confusion:
                lines += ["", "Most common category errors (gold -> predicted):", ""]
                for (g, p), n in cats.confusion.most_common(8):
                    lines.append(f"- {g or 'null'} -> {p or 'null'}: {n}")
            lines.append("")
        if result["misses"]:
            reasons = Counter(why.split(" for ")[0] if " for " not in why
                              else "item count wrong" for _, why, _ in result["misses"])
            lines += [f"Not money-exact: {len(result['misses'])} -- "
                      + ", ".join(f"{k} {v}" for k, v in reasons.most_common()), ""]
            for rid, why, shop in result["misses"][:20]:
                lines.append(f"- `{rid}` {shop} -- {why}")
            if len(result["misses"]) > 20:
                lines.append(f"- ... and {len(result['misses']) - 20} more "
                             f"(see the _examples.jsonl beside the cache)")
    return "\n".join(lines) + "\n"


def money_key(target: dict) -> tuple:
    """Total plus every non-tax price. Survives re-OCR and relabelled names,
    which is what the old project's copies of these receipts differ by."""
    return (target.get("total_price"),
            tuple(sorted(str(i.get("price")) for i in target.get("items") or []
                         if isinstance(i, dict) and not is_tax(i))))


def seen_ids(records: list[dict], path: Path) -> set[str]:
    trained = {money_key(json.loads(line)["target"])
               for line in path.open(encoding="utf-8") if line.strip()}
    return {r["id"] for r in records if money_key(r["target"]) in trained}


def subset(result: dict, keep: set[str]) -> tuple[dict, CategoryTally | None]:
    rows = [r for r in result["per_record"] if r[0] in keep]
    tally = None
    if not result["legacy"]:
        tally = CategoryTally()
        for _, _, items, gold in rows:
            tally.add(gold, items)
    return eval_metrics.aggregate([r[1] for r in rows]), tally


def expand(patterns: list[str]) -> list[str]:
    found = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(pattern)) or [pattern])
    found = [c for c in found if Path(c, "adapter_config.json").exists()]

    def step(c):
        tail = Path(c).name.split("-")[-1]
        return (0, int(tail)) if tail.isdigit() else (1, 0)
    return sorted(found, key=step)


def main(argv=None) -> None:
    args = parse_args(argv)
    joint, legacy = expand(args.checkpoints), expand(args.legacy)
    if not joint and not legacy:
        sys.exit("error: no checkpoint directory with an adapter_config.json")

    records = [json.loads(l) for l in args.val.open(encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} receipts, {len(joint)} joint + {len(legacy)} legacy checkpoint(s)")

    args.seen = seen_ids(records, args.legacy_train) if args.legacy_train else None
    if args.seen is not None:
        print(f"{len(args.seen)} of them are in {args.legacy_train.name}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate(c, True, records, args) for c in legacy]
    results += [evaluate(c, False, records, args) for c in joint]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results, records, args), encoding="utf-8")
    print(f"\nwrote {args.out}")
    for result in results:
        c = result["categories"].summary() if result["categories"] else {}
        print(f"  {result['label']:48s} money {pct(result['extraction'].get('money_exact_rate')):>6s}"
              f"   cat {pct(c.get('cat_acc')):>6s}   sub prec {pct(c.get('sub_precision')):>6s}")


if __name__ == "__main__":
    main()
