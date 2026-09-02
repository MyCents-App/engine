# Phase 3 - Zero-shot baseline comparison

Eval set: `data/val.jsonl` (160 rows, stratified). Prompt: the shared system prompt in
`prompts.py`. Scoring: `eval_metrics.py`. Decoding: greedy (`do_sample=False`).

## Methodology note - the two row types are NOT equivalent

**Local model rows (the first five)** are real automated benchmarks: each checkpoint was
downloaded, loaded in 4-bit (bitsandbytes NF4) on the local RTX 5070, and run through
`model.generate()` over all 160 val rows by `baseline_eval.py`. The `s/receipt` and
`tok/s` columns are measured on that GPU. These are reproducible: re-run
`baseline_eval.py --only <key>`.

**The `claude-sonnet-5-zeroshot` row is not an API benchmark.** No Anthropic API key is
configured on this machine and no API call was made. It was produced by the orchestrating
Claude Code session (Sonnet 5) reading 20 receipts' raw OCR text from
`reports/claude_baseline_inputs_only.jsonl` (a deliberately target-free file) and writing
the JSON extractions by hand into `reports/claude_baseline_completions.jsonl`, which were
then scored by the same harness. It is a genuine blind zero-shot data point - the gold
answers were not visible when the completions were written - but it is **not** rerunnable
as a script, has no sampling/temperature control, has no latency numbers (hence `-`), and
covers only 20 rows vs 160. An earlier 20-row sample was discarded after its gold targets
were read by accident; `sample_claude_baseline.py` excludes those rows by reconstruction.

Treat it as a rough upper-bound reference for what a frontier model does on this task,
not as a directly comparable benchmark entry. **The local-model rows are what the Phase 4
base-model decision rests on.**

## Reading the table

The leading columns are the stated objective for this model -- get the number of items
and their prices exactly right; names only need to be recognizable. All three ignore
item names completely.

- **MONEY EXACT** - `Prices exact` AND the receipt total is also exact. The headline number.
- **Prices exact** - the multiset of item prices matches gold exactly (so the item count
  matches too). Order-insensitive; duplicate prices must appear the right number of times.
- **Count exact** - the item count matches, prices aside.
- **Name sim / Name ok** - name similarity over price-matched lines only, so name quality is
  reported on lines the model already priced correctly rather than being dragged down by
  price errors. `Name ok` is the fraction scoring >= 60/100.
- `Item F1` and `Exact match` are the older name-coupled metrics, kept for continuity. They
  can look flattering: dropping one line of a 3-line receipt still scores 80% Item F1.

Every row is scored by the CURRENT `eval_metrics.py`, re-run over each model's saved raw
completions, so no row is frozen at an older metric set.

## Results

| model | n | MONEY EXACT | Prices exact | Count exact | Total exact | Price recall | Name sim | Name ok | JSON valid | Schema valid | Shop exact | Shop sim | Item F1 | Exact match | Discount F1 | Recon pass | s/receipt | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 160 | 19.4% | 23.1% | 38.8% | 72.5% | 66.8% | 77.5 | 81.7% | 98.1% | 74.4% | 27.5% | 76.6 | 55.0% | 0.0% | 27.5% | 28.7% | 8.46 | 29.1 |
| gemma-4-e4b-it | 160 | 28.7% | 31.9% | 63.7% | 81.2% | 78.9% | 85.8 | 90.3% | 100.0% | 92.5% | 35.6% | 82.0 | 70.2% | 3.1% | 38.0% | 49.4% | 25.57 | 16.9 |
| typhoon2-qwen2.5-7b | 160 | 32.5% | 37.5% | 66.2% | 79.4% | 77.9% | 78.2 | 80.3% | 99.4% | 63.7% | 34.4% | 77.6 | 64.7% | 5.6% | 39.2% | 47.5% | 5.57 | 34.1 |
| qwen3.5-9b | 160 | 26.2% | 30.0% | 56.9% | 76.2% | 78.3% | 84.2 | 88.8% | 100.0% | 96.9% | 23.8% | 76.4 | 68.3% | 0.0% | 29.9% | 55.0% | 10.02 | 22.4 |
| llama-sea-lion-v3-8b-it | 160 | 19.4% | 23.8% | 43.1% | 73.8% | 78.5% | 86.9 | 89.8% | 99.4% | 70.0% | 16.9% | 63.0 | 65.6% | 0.0% | 30.1% | 33.8% | 5.91 | 34.8 |
| claude-sonnet-5-zeroshot (n=20, reference only) | 20 | 70.0% | 70.0% | 85.0% | 95.0% | 91.5% | 87.2 | 82.8% | 100.0% | 100.0% | 85.0% | 94.6 | 79.0% | 20.0% | 87.5% | 90.0% | - | - |
