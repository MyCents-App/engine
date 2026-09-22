# Categorization with the same model — the plan and the recipe

> ## ⚠ SUPERSEDED IN PART — read this first (22 Sep 2026, evening)
>
> **The two-adapter design in §1, §3 and §5 is no longer the plan.** Extraction
> and categorization were merged into **one task, one adapter, one call**:
> raw Surya OCR text in, one JSON object out carrying the shop, the items with
> prices, and a `c`/`s` category pair per item.
>
> **Current plan of record: [`ANNOTATION.md`](ANNOTATION.md)** — dataset
> contract, field rules, the taxonomy, and §7 on what the merge costs.
> The prompt lives in [`prompts.py`](prompts.py).
>
> ### Why it changed
>
> The two-adapter plan was trained and it overfitted. One run, 2 epochs,
> 2,693 synthetic baskets: pooled validation loss fell to 0.045 and looked
> healthy while **real-receipt loss climbed 0.2713 → 0.4443, monotonically,
> from the very first checkpoint.** §3's own warning — "more epochs memorise
> catalog rows and lose the real receipts" — was right, and 2 epochs was
> already too many. The final checkpoint was the worst of seven.
>
> Scoring also surfaced a structural failure the loss curve hid: on 11-12
> item baskets the model emits well-formed JSON that closes correctly and
> contains **one entry too few**. Not truncation (138 tokens of a 1024 cap),
> not batching (it reproduces at batch 1). It miscounts.
>
> Both point the same way: the synthetic baskets cannot teach what a real
> receipt looks like, and 15 real validation receipts cannot tell you whether
> anything has. The fix is ~200 hand-annotated real receipts plus synthesised
> OCR/JSON pairs, and a **real-only validation set**.
>
> ### What is still valid here
>
> - **§2's honesty about the data** — the catalog has 0 Transport, 0 Bills &
>   Utilities and 1 Entertainment row. Still true, still the biggest gap.
> - **§4's metrics and targets** — valid ≥99%, category accuracy ≥90% on
>   synthetic with the real number deciding, sub accuracy ≥70%, sub precision
>   ≥90%. Unchanged; the joint evaluator must also score extraction.
> - **§6's backend plan** — stages 1-4 stay as deterministic lookups in front
>   of the model, and a model decline falls to user review. Unchanged, except
>   that categories now arrive with the extraction draft instead of needing a
>   second `/v1/categorize` call.
>
> ### What is superseded
>
> - **§1's "why a separate adapter, not one model for both tasks"** — reversed.
> - **§3's recipe** — `train_qlora.py` now exists and is committed; train from
>   base `Qwen/Qwen3.5-2B` on the joint data. Keep `--max-seq-length 2048` or
>   higher: the joint system prompt alone is **904 tokens** (the two separate
>   prompts were ~330 each), and a 14-item receipt runs ~1,637 end to end.
> - **§5's adapter swapping** — one adapter, no `set_adapter` per call. Note
>   that serving `ENGINE_MAX_NEW_TOKENS` (768) may need raising: the joint
>   output is ~33 tokens per item versus ~12 before, so 768 caps at roughly
>   23 items.
> - **`data/categorize_*.jsonl`** — the input side is a clean item list, not
>   OCR text, so it cannot train the joint task. Kept only as reference for
>   synthesising.
> - **`eval_categorize.py`** — scores the old shop+items→categories task. Its
>   metric code is reusable; its input handling is not.

---

Status (22 Sep 2026): **dataset built, prompt fixed, nothing trained yet.**
This replaces stage 5 (the char n-gram Naive Bayes) of the backend's
pipeline with a second QLoRA adapter on the same Qwen3.5-2B base that does
extraction. Stages 1-4 (catalog, brand, keyword) stay in the backend; they are
lookups, deterministic, and faster than any model.

---

## 0. If you are the person with the GPU — start here

Follow this file top to bottom; §3 (training) and §4 (evaluation) are the
parts that happen on your machine. Before you start you need three things:

1. **The dataset — two files you will be sent**, `categorize_train.jsonl`
   (8 MB) and `categorize_val.jsonl` (0.9 MB). Put them in
   `engine/train/data/` (create the folder; it is gitignored). You do not
   need database access. Sanity check before training:

   ```bash
   wc -l train/data/categorize_*.jsonl     # 2693 train, 294 val
   ```

   If those counts ever need to change (the catalog grew, the gold receipts
   got labelled), ask for a fresh pair — they are built by
   `server/scripts/build_categorization_sft.py` on the Mac that has the DB.
2. **Your existing `train_qlora.py`** — the one that produced
   `checkpoint-550` for extraction. It is not in git. The command in §3
   assumes the same flags; if yours are named differently, the only things
   that must change are the data paths and the output directory. The
   prompt is already inside every record, so the script never needs to
   know about it.
3. **The unsloth virtualenv**, not the Surya one. They cannot share an
   interpreter (`surya-ocr` pins `transformers<5`, unsloth needs 5.5).

What to hand back when done:

- the checkpoint directory (`checkpoints/qwen3.5-2b-categorize/checkpoint-N`,
  the one you picked, not all of them), and
- `train/reports/categorize_eval.md` with the §4 table. The number that
  decides whether this ships is **category accuracy on the real receipts**,
  reported separately from the synthetic baskets.

One thing you will notice: the validation set has only 15 real receipts. The
team is labelling the 84 gold receipts (§2, last paragraph) to fix that. If
the labels are ready before you train, ask for a rebuilt `categorize_val.jsonl`
and use it; if not, train now anyway — the same checkpoint can be
re-evaluated on the bigger set later, the training data does not change.

---

## 1. What a receipt goes through, after this lands

```
 phone                          engine (GPU box)                    backend
 ─────                          ────────────────                    ───────
 1. photos ──POST /api/v1/engine/extract──▶ proxy ──▶ /v1/extract
                                            Surya OCR per page
                                            stitch pages (overlap removed)
                                            Qwen + EXTRACTION adapter
                                              → shopName, items[{name, price}]
                                            postprocess (prices, date, tax)
 2. ◀── draft JSON, verbatim ────────────────────────────────────────
 3. ML Kit on the phone fills nameEn for Thai names; user confirms
 4. ──POST /api/v1/receipts/categorize {shopName, items[{name, nameEn, price}]}──▶
                                                                      per item, first hit wins:
                                                                        1 catalog exact   (trusted rows, 0013)
                                                                        2 catalog fuzzy
                                                                        3 brand / merchant
                                                                        4 keyword
                                                                      the items still unresolved, ONE call:
 5.                          /v1/categorize ◀──{shop, items[{name, name_en}]}──
                                            Qwen + CATEGORIZATION adapter
                                              → {"items":[{"c","s"}, ...]} positional
                                            parse_output(): wrong length / unknown
                                            category → null (never repaired)
 6.                                       ──▶ each item: (category, subcategory) or
                                                                        6 user review   (model declined or failed)
                                                                      store expense_items, name_en from
                                                                      catalog name_alt ▷ phone's nameEn
 7. ◀── categorized receipt ─────────────────────────────────────────
 8. review screen: user fixes what needs fixing, confirms
 9. ──PATCH items, POST confirm──▶                                    learning loop → products (0013 trust)
```

Two things do not change: Qwen keeps emitting the printed (Thai) text, and
categorization keeps running on the Thai `name` (with `name_en` as an
optional hint in brackets). Translation stays a display layer.

**Why one call per receipt, not per item.** A 2B model gets "น้ำแข็ง" right
only if it knows whether it is on a Suki Teenoi receipt or a 7-Eleven one.
The shop and the other items are that context. It is also 1 generation
instead of N — the extract call already costs 3-6 s; this adds one more of
~1 s.

**Why a separate adapter, not one model for both tasks.** Two prompts, two
output schemas, two eval suites, and the extraction checkpoint's 67.6% is
hard-won — retraining it jointly risks that number for no gain. Both
adapters sit on the same 4-bit base, loaded once; PEFT swaps them per call.

---

## 2. The data (already built)

```bash
cd server && uv run python scripts/build_categorization_sft.py
# -> engine/train/data/categorize_train.jsonl   2,693 receipts / 16,279 items
# -> engine/train/data/categorize_val.jsonl       294 receipts /  1,754 items
```

`data/` is gitignored (8 MB of repeated system prompt). The two files are
sent to the GPU box by hand; the script needs the database and runs here.

What is in it, and what is not:

| Source | Rows | Becomes | Caveat |
|---|---|---|---|
| `products` (5,200) | item → category, 24% subcategory | baskets of 1-12 items "at" that retailer | **7-Eleven rows (3,934) have no subcategory** |
| `keywords` (216 curated, w ≥ 0.7) | keyword → category + sub | baskets of 1-6 at an unknown shop | only source of Transport / Bills / Entertainment |
| `brands` (83 single-category) | shop → category | a visit: shop name + 1-4 items of that category | teaches shop context |
| `expense_items` (55, 15 receipts) | real, user-confirmed | **validation only, whole receipts** | the only real receipts with labels |

Products and keywords are split 90/10 *before* baskets are built, so no
validation item was seen in training. `name_en` is present on ~50% of
training items (a catalog hit on the backend supplies it; a miss does not).

**The holes, honestly.** The catalog has 0 Transport, 0 Bills & Utilities
and 1 Entertainment rows. Those three categories exist in training only
through ~230 keyword/brand items. Expect the model to be weak on them, and
expect stages 3-4 (brand, keyword) to keep catching most of them anyway —
that is why they stay in front of the model. And no 7-Eleven training row
has a subcategory, so the model will answer `null` for most convenience
store items; the backend's per-category subcategory classifier can still
fill some in afterwards, as it does today.

**The one thing worth doing before training:** hand-label the 84 gold
receipts (`test/overfit-test/dataset/gold/*.json`, 303 items, real Surya
OCR text) with category + subcategory and add them to `categorize_val`.
That turns the validation number from "synthetic baskets plus 15 real
receipts" into a real held-out set. Two people, one afternoon.

---

## 3. Training (on the GPU box)

Same stack and same shape as the extraction fine-tune — unsloth QLoRA on
`Qwen/Qwen3.5-2B`, 4-bit base, LoRA on the attention and MLP projections.
`train_qlora.py` is the existing script; it reads `messages` JSONL and
applies the tokenizer's chat template, so the new files drop in unchanged.

```bash
cd engine/train
source .venv/bin/activate          # the unsloth env, NOT the Surya one

python train_qlora.py \
  --base Qwen/Qwen3.5-2B \
  --train data/categorize_train.jsonl \
  --val   data/categorize_val.jsonl \
  --out   checkpoints/qwen3.5-2b-categorize \
  --max-seq-length 2048 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 2 --lr 2e-4 --warmup-ratio 0.03 \
  --batch 4 --grad-accum 4 \
  --eval-steps 100 --save-steps 100
```

Notes that matter:

- **Sequence length 2048 is enough.** The system prompt is ~330 tokens, a
  12-item basket ~250, the answer ~150. Nothing is truncated; check the
  data-prep log says so.
- **Loss on the assistant turn only** (the script already masks the prompt
  for extraction; keep that). Otherwise the model spends its capacity
  learning to reproduce the taxonomy list.
- **2 epochs, not 5.** Baskets are random samples of the same 5,200
  products, so the set is already 3× oversampled per product
  (`BASKETS_PER_PRODUCT`). More epochs memorise catalog rows and lose the
  real receipts in validation. Watch `val/real` separately (see §4); when
  it stops improving, stop.
- **VRAM**: ~6 GB at batch 4 × 2048 in 4-bit. Fits the same card the
  extraction run used. If it does not, `--batch 2 --grad-accum 8`.
- **Time**: ~2,700 examples × 2 epochs ≈ 340 optimizer steps at
  effective batch 16. Roughly 25-40 minutes on the workstation.
- Pick the checkpoint by the **real-receipt** accuracy in §4, not by loss.

If `train_qlora.py` hardcodes the extraction prompt or dataset path, the
only change needed is the data path and the output dir — it never sees the
prompt text, which is already inside each `messages` record.

---

## 4. Evaluation — what "good" means

Write `eval_categorize.py` next to `eval_metrics.py`. For each validation
record: build the prompt, generate greedily (`do_sample=False`, as every
published number does), `parse_output()`, then score:

| Metric | Definition | Target |
|---|---|---|
| **valid** | `parse_output` returned a list (right length, known categories) | ≥ 99% |
| **category acc** | per item, `c` matches | ≥ 90% on synthetic, **the real number is the one on the 15 (→84) real receipts** |
| **sub acc (given)** | per item with a gold subcategory, `s` matches | ≥ 70% |
| **sub precision** | when the model gives an `s`, it is right | ≥ 90% — a wrong subcategory is worse than null |
| **decline rate** | `s` = null where gold has one | report, don't target |

Report the four `kind`s separately (7-Eleven / Makro / Big C / Watsons /
keyword / **real**). The number that decides whether this ships is category
accuracy on real receipts versus what stage 5 gets on the same items today:
run `server/scripts/measure_pipeline_coverage.py` for the baseline. Stage 5
currently resolves what stages 1-4 leave; the model is only ever asked the
same leftovers, so compare on those.

Also report **latency**: seconds per receipt at 1, 5 and 12 items, on the
serving GPU, 4-bit. Budget is ~1 s; the categorize endpoint is on the
user's critical path.

---

## 5. Serving — the engine change

`train/app/categorize.py` (new) and one endpoint in `train/app/api.py`:

```
POST /v1/categorize          X-API-Key required, same as /v1/extract
{ "shopName": "7-Eleven",
  "items": [ {"name": "ชีสโรลไส้กรอก", "nameEn": null}, ... ] }      ≤ 100 items

200 { "ok": true,
      "items": [ {"category": "Groceries", "subcategory": "Snacks & sweets"},
                 {"category": "Food & Dining", "subcategory": null} ],
      "engine": {"model_seconds": 0.9, "adapter": "categorize"} }
200 { "ok": false, "error": "model output unusable" }   parse_output → None
```

Positional, one entry per input item; the backend trusts the order, never
the names. `ok: false` is a decline — the backend treats it exactly like
stage 5 declining today (→ user review), never as a server error.

Loading both adapters on one base (in `extraction.load_model`):

```python
_model, _tokenizer = FastLanguageModel.from_pretrained(settings.checkpoint, ...)  # extraction, as now
_model.load_adapter(settings.categorize_checkpoint, adapter_name="categorize")
# per call: _model.set_adapter("default") for extract, "categorize" for categorize
```

Generation is already serialised on the GPU (`max_queue`); adapter switching
happens inside that lock, so the two never interleave. Add
`ENGINE_CATEGORIZE_CHECKPOINT` to `config.py`, `/ready` reports
`categorize_loaded`, and `require_api_key` guards the route. Prompting goes
through `categorize_prompts.build_messages` and the tokenizer's chat
template — **the same module the data builder used** — so serving and
training cannot drift.

---

## 6. Backend — swapping stage 5

`server/app/services/categorization/pipeline.py`:

- `categorize_one` stays for stages 1-4. Add `categorize_batch(names, shop,
  user_id)` that runs 1-4 per item, collects the leftovers, and makes ONE
  `POST /v1/categorize` (via `engine_proxy`, which already holds the key)
  for them. Fill `Match(step=5, method="qwen_categorize", confidence=0.85)`
  for each answer, subcategory from the model when given.
- On `ok: false`, timeout, or 5xx: leftovers fall to stage 6 exactly as a
  stage-5 decline does now. The engine being off must never block a
  receipt from being saved.
- `repositories/receipts.create_and_categorize` calls the batch function
  instead of the per-item loop.
- `classifier.py` stays as a fallback behind a setting
  (`CATEGORIZE_WITH_ENGINE=1`) until the model has beaten it on real
  receipts; then it can go.
- The subcategory fallback classifier (`predict_subcategory`) keeps running
  after the model, only where the model said `null`.

Nothing changes for the app. `pipelineStep: 5` means "the model", as it
means "the classifier" today.

---

## 7. Checklist

- [ ] Hand-label the 84 gold receipts; rebuild `categorize_val.jsonl` with them (§2)
- [ ] `scp` `data/` to the GPU box; train (§3); pick by real-receipt accuracy (§4)
- [ ] Record the eval table in `train/reports/categorize_eval.md`, with the stage-5 baseline beside it
- [ ] Engine: `/v1/categorize`, second adapter, `/ready.categorize_loaded` (§5); tests
- [ ] Backend: `categorize_batch` behind `CATEGORIZE_WITH_ENGINE` (§6); tests with a stub engine
- [ ] Measure `measure_pipeline_coverage.py` before/after and write the delta in `server/db/HANDOVER.md`
- [ ] Retire `classifier.py` when the model wins on real receipts
