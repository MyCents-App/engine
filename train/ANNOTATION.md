# Annotating real receipt photos — the format

Raw Surya OCR text in, one JSON object out: shop, items with prices, and a
category per item. **One task, one model, one call** — the extraction and
categorization jobs are merged rather than split across two adapters.

Status (22 Sep 2026): written after the first categorization run overfitted to
synthetic baskets — real-receipt loss rose from 0.27 to 0.44 across 2 epochs
while pooled loss flatlined at 0.045 and looked healthy. Real annotated
receipts, and a real-only validation set, are the fix.

---

## 1. The record

One **joint** task: raw Surya OCR text in, one JSON object out carrying the
shop, the items with prices, AND each item's category. This replaces the
earlier two-adapter plan (extraction adapter + categorization adapter) with a
single call — see §7 for what that costs.

JSONL, UTF-8, Thai left as Thai. One record per line:

```json
{
  "id": "2026-09-22-malapizong-001",
  "meta": { "kind": "real" },
  "input": "ทานที่ร้าน<br>A206\nหม่าล่าปีชง2\nวันที่ : 14/11/2025 เวลา : 17:54:37 เวลา : 17:54:37\n16  60.00\n1 ซุปผสม  9.00\nยอดสุทธิ :  69.00",
  "target": {
    "shop_name": "หม่าล่าปิซง2",
    "items": [
      { "name": "16",     "price": "60.00", "c": null,           "s": null },
      { "name": "ซุปผสม", "price": "9.00",  "c": "Food & Dining", "s": "Restaurants" }
    ],
    "total_price": "69.00"
  }
}
```

Read the first item against the `input` above. OCR reduced that line to
`16  60.00` — the price survived, the name did not, so the name stays `"16"`.
Writing the real name because you can see it on the photo is what teaches the
model to invent names out of nothing.

Contrast that with the shop name: `หม่าล่าปีชง2` **is** in the OCR text, just
misread, so correcting it to `หม่าล่าปิซง2` is exactly the job. Repairing
garbled text is what this model is for; inventing absent text is not.

`c` is `null` on that item only because this example is a general-retailer
case. See §3, "When OCR destroyed the name" — at a single-category merchant
the shop supplies the category even with no name.

`input` is the OCR text exactly as Surya produced it — `<br>` artifacts,
duplicated lines, character errors and all. Do not clean it. Correcting the
noise is what the model is being trained to do, so a cleaned input teaches it
nothing. For a receipt shot across several photos, `input` is the **stitched**
text (pages joined, overlap removed), because that is what the model is
prompted with at serving time.

`meta.kind` is the **one required field**: `"real"` or `"synthetic"`. The
evaluator reports the two separately, and that separation is the whole reason
the first run's failure was visible — pooled validation loss sat at 0.045 and
looked healthy while real-receipt loss climbed from 0.27 to 0.44.

`id` is **optional**. Nothing in the training path reads it; the model never
sees it, and the tooling falls back to `file:line` labels when it is absent.
Two reasons to set it anyway, and the photo's filename stem does the job:

- **The train/validation split is derived from it.** Splitting by line index
  means regenerating the file reshuffles the split, and a validation receipt
  can cross into training — which inflates the numbers with nothing to flag
  it. Hashing a stable `id` survives re-labelling, added receipts and fixed
  rows. Hashing `input` instead gives the same property until the day anything
  is re-OCR'd, at which point that receipt changes sides.
- **Error analysis.** When the evaluator names the twelve worst receipts you
  will want to open those photos, and `real.jsonl:47` does not say which image.
  That is most of the work after a training run.

Every money value is a **string**, `NN.DD` — two decimals, no `฿`, no
thousands separator, no sign. `"7"` and `"7.0"` and `"1,250.00"` are all
wrong; write `"7.00"`, `"7.00"`, `"1250.00"`.

### What is deliberately NOT in the target

- **`receipt_date`** — `app/date_extract.py` parses it deterministically,
  Buddhist years included, with its own test suite. Replacing tested code
  with a 2B model's guess is a downgrade.
- **`name_en`** — translation is not the engine's job (RECEIPT_API.md,
  "Translation is not the engine's job"). The app fills it with ML Kit and the
  backend overrides from the product catalog.
- **`tax_amount` / `tax_added_on_top`** — when tax sits on top of the listed
  prices, add one item named `"vat"` with `"c": null, "s": null`. `_split_tax`
  in the API lifts it back out. The existing reconcile logic is written and
  tested against that shape.
- **`photos` / `annotator` / `notes`** — provenance. Keep them wherever you
  track the work; they are not model targets. `id` rides along for
  traceability and is not shown to the model.

---

## 2. Field rules

These are not style preferences — they are the contract the model is trained
against. Getting them wrong teaches the model the wrong thing, and it will
reproduce the mistake confidently.

Everything below describes fields of `target`.

### shop_name
As printed, OCR errors corrected, **in the receipt's own language**. Do not
translate. `"7-Eleven"`, `"แม็คโคร"`, `"CP ALL, 7-Eleven"`.

### items[].name
As printed, OCR errors corrected, original language kept. Never translate.
This is the text categorization runs on, so an English name here silently
changes what the model learns — and it must be recoverable from `input`
(see "When OCR destroyed the name").

### items[].price — the one people get wrong
**Line totals, not unit prices.**

| Printed on receipt | Annotate as |
|---|---|
| `2 ChickN'Roll ALC   78.00` | one item, `"78.00"` |
| `mineral-water  3   30.00` | one item, `"30.00"` |
| Buy-one-get-one on a 45.00 item | one item, `"45.00"` — do **not** add the free one |
| `Shampoo 120.00` then `  discount -20.00` naming it | one item, `"100.00"` — the discount is folded in, no separate line |

### basket-wide_discount
A discount **not tied to any specific item** (a coupon, a member discount off
the whole bill). Put the amount here and **leave every item price as printed**
— do not subtract it yourself. The code layer spreads it pro-rata later.

**Omit the key entirely when there is no basket discount.** Not `null`, not
`"0.00"` — the key is simply absent, which is how the contract has always
worked and how the model was trained.

If the discount names a specific item, it is not a basket discount — fold it
into that item's price per the table above.

### VAT — a `"vat"` item, not a field
- Prices **exclude** VAT, so it is added to reach the total → add one item
  `{"name": "vat", "price": "3.01", "c": null, "s": null}`
- Prices **already include** VAT, nothing added on top → **no vat item**. The
  total already contains it.
- No VAT printed → no vat item.

`c` is `null` on that row because tax is not a purchase: categorized as one,
every per-category spending total is wrong by the tax. `_split_tax` in the API
lifts it onto `taxAmount` before the response goes out.

Nearly every Thai receipt prints a `TAX ID#` boilerplate line. That is not VAT.

### total_price
The receipt's **final printed total**. Not your computed sum — if they
disagree, the printed one wins and you note the discrepancy.

---

## 3. Categories — exact spelling, copy-paste them

Eight categories, 47 subcategories. `c` must be one of the eight **spelled
exactly as below** (or `null`), including the `&` and the capitalisation, and
`s` must belong to that same category. A typo makes the row unusable.

| Category | Subcategories |
|---|---|
| `Food & Dining` | Restaurants · Fast food · Coffee & café · Street food · Food delivery · Bakery & desserts · Drinks & beverages |
| `Groceries` | Fresh produce · Meat & seafood · Dairy & eggs · Snacks & sweets · Instant & frozen · Condiments & cooking · Household supplies |
| `Transport` | Public transit · Ride-hailing · Fuel & parking · Vehicle maintenance · Tolls & expressway |
| `Shopping` | Clothing & shoes · Electronics & gadgets · Beauty & cosmetics · Home & furniture · Books & stationery · Gifts |
| `Health & Wellness` | Pharmacy & medicine · Personal care · Doctor & clinic · Dental · Fitness & gym · Supplements & vitamins |
| `Bills & Utilities` | Electricity · Water · Internet & phone · Rent & housing · Insurance · Subscriptions |
| `Entertainment` | Movies & cinema · Streaming services · Games & gaming · Events & concerts · Hobbies · Social & nightlife |
| `Education` | Tuition & fees · Courses & workshops · Books & materials · Software & tools |

Note `Coffee & café` carries an accent. `Books & stationery` (Shopping) and
`Books & materials` (Education) are different subcategories of different
categories.

### `s`: prefer `null` to a guess

A wrong subcategory is worse than none — the target is 90% precision when one
is given, versus 70% recall. If you hesitate, write `null`. The backend has a
separate subcategory classifier that runs afterwards on the nulls.

### When OCR destroyed the name, `c` is `null` too

Real receipts lose item names outright. A line that Surya read as

```
16  60.00
```

has no recoverable name — the price survived and the name did not. Write
`name` as whatever is actually there.

**Never recover the name from the photo when the OCR does not contain it.**
The model only ever sees `input`; a target naming something absent from the
input trains it to hallucinate. This is the one case where the photo must not
win.

Be clear about what "unrecoverable" means, though — it is narrow. Repairing a
garbled name (`หม่าล่าปีชง2` → `หม่าล่าปิซง2`) and reconstructing missing Thai
vowels and tone marks (`ชสโรลไสกรอ` → `ชีสโรลไส้กรอก`) are the **main thing
this model is being trained to do**, and a name that appears elsewhere in the
text, or that OCR'd cleanly where the item repeats, is recoverable too.
"Recoverable" is judged against the whole OCR text, not one line. Only a name
with no surviving characters anywhere is gone.

### ...but the category often survives the name

`c` depends on the shop, not on the item:

- **Single-category merchant** — restaurant, pharmacy, cinema, petrol station:
  give `c` anyway. An unnamed line on a hotpot receipt is `Food & Dining` with
  near-certainty, and nulling it throws away a reliable label. Leave `s` null.
- **General retailer** — 7-Eleven, Makro, Big C, Watsons: an unnamed item could
  be Groceries, Food & Dining, Health & Wellness or Shopping. `c: null`.

That split mirrors the backend's stage 3, which already maps 83
single-category brands straight to a category without looking at the item.

Never infer a category from a **price alone**. The shop is context; the price
is not. And a `null` is not a hole in the data — it is the model learning to
decline, which routes the item to user review, the same path a stage-5 decline
takes today. `c` is nullable for that reason and for the `"vat"` row.

### The shop is context, and it changes the answer

The same product is a different category at a different shop. This is the
whole reason the model is called once per receipt with the shop name rather
than once per item.

| Item | At | Category |
|---|---|---|
| น้ำแข็ง (ice) | Suki Teenoi (restaurant) | `Food & Dining` |
| น้ำแข็ง (ice) | 7-Eleven | `Groceries` |
| Bottled water | Restaurant | `Food & Dining` / Drinks & beverages |
| Bottled water | Makro | `Groceries` / Drinks & beverages |

Annotate what the **spend** was, not what the object is.

### The categories we are starved of

The synthetic catalog has **0 Transport, 0 Bills & Utilities and 1
Entertainment** rows. If a real receipt is a BTS trip, a parking fee, a phone
top-up, a cinema ticket or a utility bill, it is worth far more than another
convenience-store basket. Prioritise those photos.

---

## 4. Check your own work before handing it over

Every annotation must satisfy this, within 3% (`postprocess.RECONCILIATION_TOLERANCE`):

```
sum(item prices) - basket-wide_discount = total_price
```

The vat row, when present, is one of the item prices — so it is already inside
the sum.

If it doesn't balance, something is misread — usually a unit price entered
where a line total belongs, or a basket discount subtracted from items as
well as recorded. Fix it or put the receipt aside with a note; do not round a
number to force the balance.

Also check: every `c` is one of the eight strings above or `null`, every `s`
belongs to its own category or is `null`, and every price matches
`^\d+\.\d{2}$`.

---

## 5. The workflow, and how the 200 are split

Do not annotate from blank files. On real held-out receipts the extraction
checkpoint scores 61.8% money-exact, 80.8% price recall and 79.4% total-exact
(`reports/phase5_real_test_comparison.md`), so most of what you need is
already there — bootstrap from its own output and correct it:

```
train\run_demo.bat                                   # engine up; ngrok not needed
cd train
.venv\Scripts\python.exe review_photos.py --no-pause # photos -> draft JSON
.venv\Scripts\python.exe make_annotations.py         # drafts -> skeletons
   ... fill category + subcategory, fix what is wrong ...
.venv\Scripts\python.exe validate_annotations.py     # gate before the builder
```

`make_annotations.py` never overwrites an existing annotation, so re-running
it after shooting more photos only adds what is new. It also lists the
receipts the engine could not reconcile — check those against the photo first,
they are where its numbers are wrong.

`validate_annotations.py` prints the category distribution every time. Watch
the three starved rows: **Transport, Bills & Utilities and Entertainment have
0 / 0 / 1 rows in the synthetic catalog**, so real receipts are the only thing
that will ever teach them. Knowing you have none of those while there is still
time to go photograph some is the point of printing it.

### The split — 200 real, and why not all of them train

| | Receipts | ~Items | Role |
|---|---|---|---|
| Real — train | 130 | ~800 | shifts the distribution toward real receipts |
| **Real — validation** | **70** | **~430** | never trained on; **this picks the checkpoint** |
| Synthetic — train | 1,000 | ~6,000 | teaches the taxonomy and the output format |

Split **by receipt, not by item**, and hold the validation 70 back completely.
Putting all 200 into training leaves the checkpoint choice resting on the 15
real receipts already in `categorize_val.jsonl`, which is the problem that
produced this document.

Mirror CATEGORIZATION.md §2's discipline while you are at it: products and
keywords are split 90/10 *before* baskets are built, so no validation item is
ever seen in training. The same applies here — if the same shop and basket
appears twice, keep both sides on the same side of the split.

### On cutting the synthetic set to 1,000

Lower **baskets per product**, do not drop products. The existing 2,693
records resample the same 5,200 catalog items about 3× (`BASKETS_PER_PRODUCT`
in `build_categorization_sft.py`), and that repetition is what the first run
memorised. `BASKETS_PER_PRODUCT=1` gives ~1,000 records with the full catalog
still covered. Dropping products instead would lose taxonomy coverage that
the real receipts cannot replace.

**Do not throw the synthetic baskets away entirely.** They teach the output
format, and the first run's structural failure — emitting 11 entries for a
12-item basket — is something more data of *any* kind helps. What they cannot
teach is what a real receipt looks like. That is the gap the 200 fill.

Real items will be roughly 17% of the training mix (~800 of ~6,800). If
real-validation loss still climbs, the cheap next move is repeating the real
training receipts 2× to reach ~29%, rather than annotating another 200.

---

## 6. What happens to the data

```
real.jsonl        ──┐
                    ├──▶ builder ──▶ prompts.build_messages(input, target)
synthetic.jsonl   ──┘                  ──▶ messages JSONL ──▶ train_qlora.py
```

`prompts.build_messages(ocr_text, target)` already takes exactly an OCR string
and a target dict, so these records map onto it directly — the builder's only
job is reading the file, splitting real into train/validation, and serialising.

**The validation split is real-only.** Synthetic rows go entirely to training;
roughly 70 real receipts are held back and never trained on. This is the
safeguard that makes synthesising OCR text safe: if the generated noise does
not match Surya's real error distribution, a real-only validation set shows it
immediately, where a mixed one would hide it until production.

### Synthesising the input

Generate the noise from **observed** Surya output, not from imagination. The
artifacts that actually occur are visible in any real OCR dump: `<br>` tags,
whole lines duplicated (`เวลา : 17:54:37 เวลา : 17:54:37`), item names
swallowed into a leading quantity (`16  60.00`), and Thai character
confusions (ี/ิ, ช/ซ, ๐/0). Reproduce those. Invented noise trains the model
to repair errors it will never see and leaves the real ones unlearned.

Draw synthetic shop and item names from the real catalog even when the noise
is approximated — that way the taxonomy coverage is genuine, which is the
whole reason to keep synthetic data at all.

---

## 7. What merging the two tasks costs

Worth knowing, because it is not free. One call instead of two, one adapter and
one prompt module is the gain. Against that:

1. **Failure coupling.** With two adapters a categorization failure lost only
   the categories and the user still got a confirmable draft. One combined
   JSON means a malformed generation loses everything, extraction included.
   Mitigated in the code layer by validating the two halves independently —
   a bad category is nulled while a good price survives — but the coupling is
   real and that mitigation has to exist.
2. **Longer output, and the model already miscounts.** The first run emitted
   11 entries for a 12-item basket with a *short* output. Adding `c` and `s`
   per item makes the output longer and a dropped item now costs the price as
   well as the category. Long receipts (10+ items) are worth
   over-representing in the data for exactly this reason.
3. **Extraction quality is back in play.** `checkpoint-550` stops being
   untouchable. Keep it on disk: if joint extraction regresses against
   `reports/phase5_real_test_comparison.md`, falling back to two adapters must
   stay possible.

---

## 8. Labelling with a large model

Generating the labels with an LLM instead of by hand is legitimate — it is
distillation, and it is how most of a set this size gets built. The prompt is
`LLM_ANNOTATION_PROMPT.md`, generated by `make_llm_prompt.py` so its taxonomy
cannot drift from `categorize_prompts.CATEGORIES`. Regenerate it rather than
editing it, and make sure everyone labelling uses the same text: two people
running two slightly different prompts produces label noise indistinguishable
from real disagreement, and it does not average out.

### The loop

```
train
un_demo.bat                                      # engine up
.venv\Scripts\python.exe review_photos.py --no-pause   # photos -> drafts, with ocrTexts

   for each receipt, to the LLM:
     LLM_ANNOTATION_PROMPT.md  +  the photo(s)  +  ocrTexts from the draft
   save its answer as data/llm/<same stem>.json

.venv\Scripts\python.exe make_annotations.py --merge-llm data/llm
.venv\Scripts\python.exe validate_annotations.py
.venv\Scripts\python.exe make_annotations.py --collect data/real.jsonl
```

One receipt per request. Batching several into one prompt blurs them together,
and the cost of a separate request is nothing next to re-labelling.

**The labeller never writes `input`.** It is shown the OCR text and asked for
`target` only; `--merge-llm` splices the exact text off the draft. Large models
are unreliable at echoing long noisy text verbatim — they tidy `<br>` artifacts,
normalise spacing, drop a duplicated line — and any drift there trains the model
on input the OCR engine does not produce. That corruption is invisible
afterwards: the pair looks well-formed and the model just learns to expect text
it will never be given. Asking for labels only removes the failure mode
entirely, and shortens the answer.

**Give the labeller the OCR text as well as the photo.** This is not optional.
The model being trained only ever sees OCR text, so a target containing a name
that OCR destroyed teaches it to invent names. A labeller shown only the photo
will do exactly that, confidently, on every receipt where OCR dropped a line —
and those are common.

### Three rules that keep this honest

1. **The validation set must be human-verified.** If training and validation
   are both LLM-labelled, the eval measures agreement with the labeller, not
   correctness — a systematic labelling error scores as success. Hand-check the
   ~70 real validation receipts yourself. That is the one part not worth
   automating.
2. **Measure the labeller before trusting it.** Hand-check ~30 of its outputs
   and count how often it is right. That number is the ceiling on what the 2B
   can learn: a labeller at 85% cannot produce a 95% student. If it is weak on
   something specific — the line-total rule is the usual one — you will see it
   in 30 receipts and can fix the prompt before spending the other 170.
3. **Run `validate_annotations.py` over everything it produces.** An LLM
   invents plausible category names (`Coffee & cafe` without the accent,
   `Food and Dining`), writes `"7.0"`, and quietly drops an item from a long
   receipt. The validator catches all of those, and `_needs_review: true` is
   the labeller's own admission that a human should look.

### Its errors are systematic, not random

Hand-annotation errors scatter and partly cancel. A model's errors correlate:
if it misreads quantity lines, it misreads them the same way on every receipt,
and the student learns that rule perfectly. This is why the sample check in
rule 2 matters more than the raw volume — 200 receipts labelled by something
you have not measured is 200 copies of the same unknown bias.
