# Annotating real receipt photos — the format

One file per receipt. One annotation serves **both** models: the extraction
adapter learns shop/items/prices/total from it, the categorization adapter
learns category/subcategory from it. Do not annotate the same receipt twice
in two formats.

Status (22 Sep 2026): written after the first categorization run overfitted to
synthetic baskets (real-receipt loss rose from 0.27 to 0.44 across 2 epochs
while synthetic loss flatlined). Real annotated receipts are the fix.

---

## 1. The file

`data/annotations/<receipt_id>.json`, UTF-8, Thai left as Thai.

```json
{
  "receipt_id": "2026-09-22-7eleven-001",
  "photos": ["IMG_4821.jpg", "IMG_4822.jpg"],

  "shop_name": "7-Eleven",
  "receipt_date": "2026-09-22",

  "items": [
    {
      "name": "ชีสโรลไส้กรอก",
      "name_en": "Cheese sausage roll",
      "price": "26.00",
      "category": "Food & Dining",
      "subcategory": "Bakery & desserts"
    },
    {
      "name": "H UHT นมยูเอชที ด.16",
      "name_en": null,
      "price": "13.00",
      "category": "Groceries",
      "subcategory": "Dairy & eggs"
    }
  ],

  "basket_discount": null,
  "tax_amount": null,
  "tax_added_on_top": false,
  "total_price": "39.00",

  "annotator": "your-name",
  "notes": ""
}
```

Every money value is a **string**, `NN.DD` — two decimals, no `฿`, no
thousands separator, no sign. `"7"` and `"7.0"` and `"1,250.00"` are all
wrong; write `"7.00"`, `"7.00"`, `"1250.00"`.

`photos` lists the image filenames in **capture order, top of the receipt
first**. Order is load-bearing for multi-photo receipts.

---

## 2. Field rules

These are not style preferences — they are the contracts the two models are
trained against (`prompts.py` and `categorize_prompts.py`). Getting them
wrong teaches the model the wrong thing.

### shop_name
As printed, OCR errors corrected, **in the receipt's own language**. Do not
translate. `"7-Eleven"`, `"แม็คโคร"`, `"CP ALL, 7-Eleven"`.

### receipt_date
`YYYY-MM-DD`. Thai receipts print the **Buddhist year** — 2569 is 2026.
Convert it. `null` if no date is printed.

### items[].name
As printed, OCR errors corrected, original language kept. Never translate
into `name`. This is the text the categorizer matches on, so an English
translation here silently changes which category the model learns.

### items[].name_en
Optional, `null` when you don't know it. Only fill it when you're confident.
Roughly half the synthetic training items have one; it's a hint, not a
requirement.

### items[].price — the one people get wrong
**Line totals, not unit prices.**

| Printed on receipt | Annotate as |
|---|---|
| `2 ChickN'Roll ALC   78.00` | one item, `"78.00"` |
| `mineral-water  3   30.00` | one item, `"30.00"` |
| Buy-one-get-one on a 45.00 item | one item, `"45.00"` — do **not** add the free one |
| `Shampoo 120.00` then `  discount -20.00` naming it | one item, `"100.00"` — the discount is folded in, no separate line |

### basket_discount
A discount **not tied to any specific item** (a coupon, a member discount off
the whole bill). Put the amount here and **leave every item price as printed**
— do not subtract it yourself. The code layer spreads it pro-rata later.
`null` when there isn't one.

If the discount names a specific item, it is not a basket discount — fold it
into that item's price per the table above.

### tax_amount / tax_added_on_top
- Receipt prints VAT and the item prices **exclude** it (so VAT is added to
  reach the total) → `tax_amount: "3.01"`, `tax_added_on_top: true`
- Receipt prints VAT but prices **already include** it (the "VAT included"
  line, total unchanged) → `tax_amount: "3.01"`, `tax_added_on_top: false`
- No VAT printed → `tax_amount: null`, `tax_added_on_top: false`

Do **not** add a `"vat"` row to `items[]`. The builder does that for the
extraction target when `tax_added_on_top` is true. Tax is not a purchase and
must never carry a category.

Nearly every Thai receipt prints a `TAX ID#` boilerplate line. That is not VAT.

### total_price
The receipt's **final printed total**. Not your computed sum — if they
disagree, the printed one wins and you note the discrepancy.

---

## 3. Categories — exact spelling, copy-paste them

Eight categories, 47 subcategories. `category` must be one of the eight
**spelled exactly as below**, including the `&` and the capitalisation.
A typo makes the row unusable.

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

### subcategory: prefer `null` to a guess

A wrong subcategory is worse than none — the target is 90% precision when one
is given, versus 70% recall. If you hesitate, write `null`. The backend has a
separate subcategory classifier that runs afterwards on the nulls.

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
sum(item prices) - basket_discount + (tax_amount if tax_added_on_top) = total_price
```

If it doesn't balance, something is misread — usually a unit price entered
where a line total belongs, or a basket discount subtracted from items as
well as recorded. Fix it or put the receipt aside with a note; do not round a
number to force the balance.

Also check: every `category` is one of the eight strings above, every
`subcategory` belongs to its own category, every price matches `^\d+\.\d{2}$`.

---

## 5. The workflow, and how the 200 are split

Do not annotate from blank files. The extraction checkpoint is 67.6%
money-exact, so bootstrap from its own output and correct it:

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

## 6. What happens to the file

Nothing in this repo reads `data/annotations/` yet. A builder has to convert
these files into the two `messages` JSONL formats, the same way
`server/scripts/build_categorization_sft.py` does today:

```
annotations/*.json
  ├─▶ extraction:     prompts.build_messages(ocr_text, target)
  │     target = {shop_name, items[{name, price}],
  │               basket-wide_discount?, total_price}
  │     (the "vat" row is injected here when tax_added_on_top)
  │     ...needs the Surya OCR text for each photo as the user turn
  │
  └─▶ categorization: categorize_prompts.build_messages(shop_name, items, labels)
        items  = [{name, name_en}]
        labels = [(category, subcategory), ...]   positional
```

The extraction side needs the receipt's OCR text as the prompt, so run each
photo through `/v1/extract-text`'s OCR path (or `ocr_service.py`) and store the
text beside the annotation — the model is prompted with OCR text, never with
the clean names you typed.
