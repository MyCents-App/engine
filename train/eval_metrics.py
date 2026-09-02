"""Phase 2: the shared evaluation harness, reused unchanged by baseline_eval.py (Phase 3) and
the fine-tuned-model eval (Phase 5) so the two are directly comparable.

Scores one model's raw completions against gold targets from the same eval JSONL (data/val.jsonl
or data/test.jsonl, produced by data_prep.py), producing per-example diagnostics and aggregate
metrics: JSON validity, shop_name accuracy, item-level precision/recall/F1 (bipartite name+price
matching), total_price / basket-wide_discount accuracy, and a reconciliation-pass-rate business-
validity signal via postprocess.reconcile.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rapidfuzz import fuzz

import postprocess
from postprocess import parse_price_strict

DEFAULT_NAME_SIM_THRESHOLD = 60.0  # 0-100; a knob to revisit once real Phase 3 outputs are seen


# --------------------------------------------------------------------------------------
# Per-example scoring
# --------------------------------------------------------------------------------------

@dataclass
class ExampleScore:
    json_valid: bool
    schema_valid: bool
    parse_error: str | None
    schema_errors: list[str] = field(default_factory=list)

    exact_match: bool = False

    shop_name_exact: bool = False
    shop_name_similarity: float = 0.0  # 0-100

    item_precision: float = 0.0
    item_recall: float = 0.0
    item_f1: float = 0.0
    item_price_only_recall: float = 0.0  # of gold items, fraction whose price appears somewhere in pred, name ignored

    # ---- primary objective ----------------------------------------------------------
    # The stated goal for this model is: get the COUNT of items and their PRICES exactly
    # right; names only need to be recognizable. These three are the headline metrics and
    # every one of them ignores item names entirely.
    item_count_exact: bool = False        # len(pred items) == len(gold items)
    price_multiset_exact: bool = False    # same count AND same multiset of prices
    money_exact: bool = False             # price_multiset_exact AND total_price_exact

    # ---- name quality (the "acceptable is fine" axis) --------------------------------
    # Measured only over price-matched pairs, so it reports name quality on the lines the
    # model already got right rather than being dragged down by price errors.
    name_quality_mean: float = 0.0        # 0-100 mean similarity over price-matched pairs
    name_acceptable_rate: float = 0.0     # fraction of those pairs scoring >= threshold
    name_matched_pairs: int = 0

    total_price_exact: bool = False

    discount_gold_present: bool = False
    discount_pred_present: bool = False
    discount_value_exact: bool | None = None  # None unless both gold and pred have it

    reconciliation_status: str = "unparseable"
    reconciliation_pass: bool = False

    latency_s: float | None = None
    output_tokens: int | None = None


def _norm_name(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _match_items(
    gold_items: list[dict], pred_items: list[dict], name_sim_threshold: float
) -> tuple[float, float, float]:
    """Greedy bipartite matching: a pair (gold_i, pred_j) is a valid candidate match only if
    their prices are exactly equal (strict NN.DD parse) AND their (lowercased) name similarity
    clears name_sim_threshold. Candidates are assigned highest-similarity-first, each item used
    at most once. Returns (precision, recall, f1)."""
    if not pred_items:
        return 0.0, 0.0, 0.0
    if not gold_items:
        return 0.0, 0.0, 0.0

    candidates = []
    for gi, g in enumerate(gold_items):
        g_price = parse_price_strict(g.get("price")) if isinstance(g, dict) else None
        g_name = _norm_name(g.get("name", "")) if isinstance(g, dict) else ""
        for pi, p in enumerate(pred_items):
            if not isinstance(p, dict):
                continue
            p_price = parse_price_strict(p.get("price"))
            if g_price is None or p_price is None or abs(g_price - p_price) > 1e-6:
                continue
            p_name = _norm_name(p.get("name", ""))
            sim = fuzz.ratio(g_name, p_name)
            if sim >= name_sim_threshold:
                candidates.append((sim, gi, pi))

    candidates.sort(key=lambda t: t[0], reverse=True)
    used_gold, used_pred = set(), set()
    tp = 0
    for _, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        tp += 1

    precision = tp / len(pred_items)
    recall = tp / len(gold_items)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _price_only_recall(gold_items: list[dict], pred_items: list[dict]) -> float:
    """Of gold items, the fraction whose exact price appears among (as-yet-unclaimed) predicted
    items, ignoring the name entirely. Isolates "got the number right" from "named it well"."""
    if not gold_items:
        return 0.0
    pred_prices = []
    for p in pred_items:
        if isinstance(p, dict):
            v = parse_price_strict(p.get("price"))
            if v is not None:
                pred_prices.append(v)

    matched = 0
    for g in gold_items:
        if not isinstance(g, dict):
            continue
        g_price = parse_price_strict(g.get("price"))
        if g_price is None:
            continue
        for i, pp in enumerate(pred_prices):
            if abs(pp - g_price) <= 1e-6:
                matched += 1
                del pred_prices[i]
                break

    return matched / len(gold_items)


def _price_multiset(items: list) -> list[float] | None:
    """Sorted list of every item price. None if ANY price is missing or unparseable -- a
    prediction with an unreadable price has not got the numbers right, so it must not be
    able to score as a match by accident."""
    vals = []
    for it in items:
        if not isinstance(it, dict):
            return None
        v = parse_price_strict(it.get("price"))
        if v is None:
            return None
        vals.append(round(v, 2))
    return sorted(vals)


def _name_quality(
    gold_items: list[dict], pred_items: list[dict], threshold: float
) -> tuple[float, float, int]:
    """Name similarity measured ONLY over price-matched pairs.

    Rationale: the objective treats names as secondary to prices, so name quality should be
    reported on the lines whose price the model already got right. Pairing is greedy on exact
    price equality (same rule as _price_only_recall); when several gold lines share a price,
    the pairing picks the highest-similarity survivor so the score is not penalised by an
    arbitrary tie-break. Returns (mean_similarity, fraction_at_or_above_threshold, n_pairs).
    """
    remaining = [p for p in pred_items if isinstance(p, dict)]
    sims = []
    for g in gold_items:
        if not isinstance(g, dict):
            continue
        g_price = parse_price_strict(g.get("price"))
        if g_price is None:
            continue
        cands = [
            (i, p) for i, p in enumerate(remaining)
            if (pv := parse_price_strict(p.get("price"))) is not None
            and abs(pv - g_price) <= 1e-6
        ]
        if not cands:
            continue
        g_name = _norm_name(g.get("name", ""))
        best_i, best_sim = None, -1.0
        for i, p in cands:
            sim = fuzz.ratio(g_name, _norm_name(p.get("name", "")))
            if sim > best_sim:
                best_i, best_sim = i, sim
        sims.append(best_sim)
        del remaining[best_i]

    if not sims:
        return 0.0, 0.0, 0
    return (
        sum(sims) / len(sims),
        sum(1 for s in sims if s >= threshold) / len(sims),
        len(sims),
    )


def score_example(
    raw_completion: str,
    gold: dict,
    ocr_text: str,
    name_sim_threshold: float = DEFAULT_NAME_SIM_THRESHOLD,
    latency_s: float | None = None,
    output_tokens: int | None = None,
    normalize_pred: bool = False,
) -> ExampleScore:
    pred, parse_error = postprocess.extract_json(raw_completion)
    json_valid = pred is not None

    # Off by default so the headline numbers describe the MODEL's raw output. Turn it on to
    # measure the deployed pipeline, where task.md Section 4 puts price formatting in the code
    # layer: it reformats parseable numbers to NN.DD and never changes a value.
    if normalize_pred and pred is not None:
        pred = postprocess.normalize_prediction(pred)

    if not json_valid:
        return ExampleScore(
            json_valid=False,
            schema_valid=False,
            parse_error=parse_error,
            reconciliation_status="unparseable",
            reconciliation_pass=False,
            latency_s=latency_s,
            output_tokens=output_tokens,
        )

    schema_valid, schema_errors = postprocess.is_schema_valid(pred)

    exact_match = pred == gold

    gold_shop = gold.get("shop_name", "")
    pred_shop = pred.get("shop_name", "") if isinstance(pred.get("shop_name"), str) else ""
    shop_name_exact = pred_shop.strip() == gold_shop.strip()
    shop_name_similarity = fuzz.ratio(gold_shop.strip().lower(), pred_shop.strip().lower())

    gold_items = gold.get("items", []) if isinstance(gold.get("items"), list) else []
    pred_items = pred.get("items", []) if isinstance(pred.get("items"), list) else []
    precision, recall, f1 = _match_items(gold_items, pred_items, name_sim_threshold)
    price_only_recall = _price_only_recall(gold_items, pred_items)

    gold_prices = _price_multiset(gold_items)
    pred_prices = _price_multiset(pred_items)
    item_count_exact = len(gold_items) == len(pred_items)
    price_multiset_exact = (
        gold_prices is not None and pred_prices is not None and gold_prices == pred_prices
    )
    name_quality_mean, name_acceptable_rate, name_matched_pairs = _name_quality(
        gold_items, pred_items, name_sim_threshold
    )

    gold_total = parse_price_strict(gold.get("total_price"))
    pred_total = parse_price_strict(pred.get("total_price")) if isinstance(pred.get("total_price"), str) else None
    total_price_exact = (
        gold_total is not None and pred_total is not None and abs(gold_total - pred_total) <= 1e-6
    )

    discount_gold_present = "basket-wide_discount" in gold
    discount_pred_present = "basket-wide_discount" in pred
    discount_value_exact = None
    if discount_gold_present and discount_pred_present:
        gv = parse_price_strict(gold.get("basket-wide_discount"))
        pv = parse_price_strict(pred.get("basket-wide_discount")) if isinstance(pred.get("basket-wide_discount"), str) else None
        discount_value_exact = gv is not None and pv is not None and abs(gv - pv) <= 1e-6

    recon = postprocess.reconcile(pred, ocr_text)

    return ExampleScore(
        json_valid=True,
        schema_valid=schema_valid,
        parse_error=None,
        schema_errors=schema_errors,
        exact_match=exact_match,
        shop_name_exact=shop_name_exact,
        shop_name_similarity=shop_name_similarity,
        item_precision=precision,
        item_recall=recall,
        item_f1=f1,
        item_price_only_recall=price_only_recall,
        item_count_exact=item_count_exact,
        price_multiset_exact=price_multiset_exact,
        money_exact=price_multiset_exact and total_price_exact,
        name_quality_mean=name_quality_mean,
        name_acceptable_rate=name_acceptable_rate,
        name_matched_pairs=name_matched_pairs,
        total_price_exact=total_price_exact,
        discount_gold_present=discount_gold_present,
        discount_pred_present=discount_pred_present,
        discount_value_exact=discount_value_exact,
        reconciliation_status=recon.status,
        reconciliation_pass=recon.passes,
        latency_s=latency_s,
        output_tokens=output_tokens,
    )


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(scores: list[ExampleScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}

    discount_tp = sum(1 for s in scores if s.discount_gold_present and s.discount_pred_present)
    discount_fp = sum(1 for s in scores if not s.discount_gold_present and s.discount_pred_present)
    discount_fn = sum(1 for s in scores if s.discount_gold_present and not s.discount_pred_present)
    discount_precision = discount_tp / (discount_tp + discount_fp) if (discount_tp + discount_fp) else 0.0
    discount_recall = discount_tp / (discount_tp + discount_fn) if (discount_tp + discount_fn) else 0.0
    discount_f1 = (
        2 * discount_precision * discount_recall / (discount_precision + discount_recall)
        if (discount_precision + discount_recall) > 0
        else 0.0
    )
    both_present = [s for s in scores if s.discount_value_exact is not None]

    latencies = [s.latency_s for s in scores if s.latency_s is not None]
    tokens = [s.output_tokens for s in scores if s.output_tokens is not None]

    return {
        "n": n,
        "json_validity_rate": _mean([float(s.json_valid) for s in scores]),
        "schema_valid_rate": _mean([float(s.schema_valid) for s in scores]),
        "exact_match_rate": _mean([float(s.exact_match) for s in scores]),
        "shop_name_exact_rate": _mean([float(s.shop_name_exact) for s in scores]),
        "shop_name_similarity_mean": _mean([s.shop_name_similarity for s in scores]),
        "item_precision_mean": _mean([s.item_precision for s in scores]),
        "item_recall_mean": _mean([s.item_recall for s in scores]),
        "item_f1_mean": _mean([s.item_f1 for s in scores]),
        "item_price_only_recall_mean": _mean([s.item_price_only_recall for s in scores]),
        "item_count_exact_rate": _mean([float(s.item_count_exact) for s in scores]),
        "price_multiset_exact_rate": _mean([float(s.price_multiset_exact) for s in scores]),
        "money_exact_rate": _mean([float(s.money_exact) for s in scores]),
        # Name metrics average over examples that HAVE at least one price-matched pair; an
        # example with no matched line has no name signal to contribute and would otherwise
        # be silently scored as "names are terrible" when the real failure was the prices.
        "name_quality_mean": _mean([s.name_quality_mean for s in scores if s.name_matched_pairs]),
        "name_acceptable_rate": _mean(
            [s.name_acceptable_rate for s in scores if s.name_matched_pairs]
        ),
        "total_price_exact_rate": _mean([float(s.total_price_exact) for s in scores]),
        "discount_precision": discount_precision,
        "discount_recall": discount_recall,
        "discount_f1": discount_f1,
        "discount_value_exact_rate": _mean([float(s.discount_value_exact) for s in both_present]),
        "discount_value_exact_n": len(both_present),
        "reconciliation_pass_rate": _mean([float(s.reconciliation_pass) for s in scores]),
        "reconciliation_status_breakdown": dict(Counter(s.reconciliation_status for s in scores)),
        "mean_latency_s": _mean(latencies) if latencies else None,
        "mean_output_tokens": _mean(tokens) if tokens else None,
        "mean_tokens_per_s": (
            _mean([t / l for t, l in zip(tokens, latencies) if l > 0])
            if latencies and tokens and len(latencies) == len(tokens)
            else None
        ),
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

# Ordered by what the model is actually for: the money columns lead, name quality follows,
# and the older name-coupled item metrics stay for continuity with the Phase 3 tables.
_TABLE_COLUMNS = [
    ("n", "n", "{:.0f}"),
    ("money_exact_rate", "MONEY EXACT", "{:.1%}"),
    ("price_multiset_exact_rate", "Prices exact", "{:.1%}"),
    ("item_count_exact_rate", "Count exact", "{:.1%}"),
    ("total_price_exact_rate", "Total exact", "{:.1%}"),
    ("item_price_only_recall_mean", "Price recall", "{:.1%}"),
    ("name_quality_mean", "Name sim", "{:.1f}"),
    ("name_acceptable_rate", "Name ok", "{:.1%}"),
    ("json_validity_rate", "JSON valid", "{:.1%}"),
    ("schema_valid_rate", "Schema valid", "{:.1%}"),
    ("shop_name_exact_rate", "Shop exact", "{:.1%}"),
    ("shop_name_similarity_mean", "Shop sim", "{:.1f}"),
    ("item_f1_mean", "Item F1", "{:.1%}"),
    ("exact_match_rate", "Exact match", "{:.1%}"),
    ("discount_f1", "Discount F1", "{:.1%}"),
    ("reconciliation_pass_rate", "Recon pass", "{:.1%}"),
    ("mean_latency_s", "s/receipt", "{:.2f}"),
    ("mean_tokens_per_s", "tok/s", "{:.1f}"),
]


def format_comparison_table(results: dict[str, dict]) -> str:
    """results: {model_name: aggregate_dict}. Returns a Markdown table, one row per model."""
    header = "| model | " + " | ".join(label for _, label, _ in _TABLE_COLUMNS) + " |"
    sep = "|---|" + "---|" * len(_TABLE_COLUMNS)
    lines = [header, sep]
    for name, agg in results.items():
        cells = [name]
        for key, _, fmt in _TABLE_COLUMNS:
            v = agg.get(key)
            cells.append(fmt.format(v) if v is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(results: dict[str, dict], path: Path) -> None:
    fieldnames = ["model"] + [key for key, _, _ in _TABLE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, agg in results.items():
            row = {"model": name}
            for key, _, _ in _TABLE_COLUMNS:
                row[key] = agg.get(key)
            writer.writerow(row)


def write_examples_jsonl(path: Path, scores: list[ExampleScore]) -> None:
    """Per-example diagnostics for debugging one model's run in detail."""
    with open(path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
