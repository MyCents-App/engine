"""Emit the prompt for labelling receipt PHOTOS with a large model.

    .venv\\Scripts\\python.exe make_llm_prompt.py                 # to stdout
    .venv\\Scripts\\python.exe make_llm_prompt.py -o llm_prompt.md

The labeller is shown the photo only, and returns `target`. The OCR text that
becomes `input` is produced separately by our own Surya pipeline and paired on
by `make_annotations.py --merge-llm`, so the labeller is never asked to
transcribe or reproduce it.

That means a target name can be more complete than the OCR text it is paired
with -- the photo shows what a garbled line really said. This is deliberate:
the gold standard for a name is what the receipt prints, not what the OCR
engine managed to recover, and mapping the lossy text onto the truth is the
denoising job the model exists to do. validate_annotations.py reports how often
a name is not literally present in the input, so the size of what the model is
being asked to reconstruct stays visible rather than becoming a surprise.

The taxonomy is injected from categorize_prompts.CATEGORIES rather than typed
out, so the labelling prompt, the trained model's system prompt, and the
validator can never disagree about what the eight categories are. Regenerate
after any taxonomy change; do not hand-edit the output.

Everyone labelling must use the SAME prompt text. Label noise from two people
running two slightly different prompts is indistinguishable from real
disagreement in the data, and it does not average out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from categorize_prompts import CATEGORIES

TAXONOMY = "\n".join(
    f"- {category}: {', '.join(subs)}" for category, subs in CATEGORIES.items()
)

PROMPT = f"""You are reading Thai/English point-of-sale receipts to build a supervised training set. You will be given **the photo(s) of one receipt**.

You will also be told the receipt's **id** (its photo filename, e.g. `photo-7`).

Read the receipt and return exactly one JSON object describing what it says. Nothing else.

Your answer becomes the ground truth a model is trained towards, so read carefully and prefer accuracy over speed. You do not need to transcribe the receipt's raw text — our OCR output is attached to your labels automatically afterwards.

---

## Output format

```json
{{
  "id": "photo-7",
  "target": {{
    "shop_name": "7-Eleven",
    "items": [
      {{ "name": "ชีสโรลไส้กรอก", "price": "26.00", "c": "Food & Dining", "s": "Bakery & desserts" }},
      {{ "name": "H UHT นมยูเอชที ด.16", "price": "13.00", "c": "Groceries", "s": "Dairy & eggs" }}
    ],
    "total_price": "39.00"
  }},
  "_flags": [],
  "_needs_review": false
}}
```

- `id` is **copied verbatim from the id you were given**. Do not invent it, do not renumber, do not count receipts yourself. It is what pairs your labels with the right OCR text, and a single off-by-one silently misaligns every record after it.
- All money values are **strings** in `NN.DD` form: two decimals, no currency symbol, no thousands separator, no sign. `"7"`, `"7.0"` and `"1,250.00"` are all wrong. Write `"7.00"` and `"1250.00"`.
- `_flags` is a list of short strings describing anything you were unsure about — a smudged price, an item you could not read, a total that does not add up. `_needs_review` is `true` when a human should look. Both are stripped before training, so use them freely: they cost nothing, and they are how a hard receipt gets a second pair of eyes instead of a confident guess.

---

## Field rules

### shop_name
The merchant, as printed, **in the receipt's own language**. Never translate. `"7-Eleven"`, `"แม็คโคร"`, `"หม่าล่าปิซง2"`.

### items[].name
The item as printed, **in the receipt's own language**. Never translate — the categorization is matched on this text downstream, so an English name here changes the meaning of the data.

Transcribe what the receipt says. If a line is genuinely illegible, give your best reading, add a `_flag`, and set `"c"` to `null` rather than inventing a category for something you cannot identify.

### items[].price — read this twice
**Line totals, not unit prices.** This is the rule most often got wrong.

| Printed on the receipt | Correct annotation |
|---|---|
| `2 ChickN'Roll ALC   78.00` | ONE item at `"78.00"` |
| `mineral-water  3   30.00` | ONE item at `"30.00"` |
| `ข้าว  x2  @25.00   50.00` | ONE item at `"50.00"` |
| Buy-one-get-one on a 45.00 item | ONE item at `"45.00"` — do NOT add the free one |
| `Shampoo 120.00` then a `-20.00` discount naming Shampoo | ONE item at `"100.00"` — fold it in, no separate discount line |

Never split a quantity line into several items. Never add an item for a discount that names a specific product. One entry per printed item line.

### basket-wide_discount
A discount **not tied to any specific item** — a coupon, a member discount, a percentage off the whole bill.

- Put the amount in `"basket-wide_discount"`.
- **Leave every item price as printed.** Do not subtract it yourself; our code spreads it across the items afterwards.
- **Omit the key entirely** when there is no basket discount. Not `null`, not `"0.00"` — absent.

### VAT
- Prices **exclude** VAT and it is added to reach the total → add one final item `{{"name": "vat", "price": "3.01", "c": null, "s": null}}`.
- Prices **already include** VAT with nothing added on top → **no vat item at all**. The total already contains it.
- No VAT printed → no vat item.

The tax row always has `"c": null`. Tax is not a purchase; categorised as one, every per-category spending total is wrong by the tax amount.

Note: nearly every Thai receipt prints a `TAX ID#` / `เลขประจำตัวผู้เสียภาษี` boilerplate line. **That is not VAT.** Ignore it.

### total_price
The receipt's **final printed total** (`ยอดสุทธิ`, `รวมทั้งสิ้น`, `Total`, `Net`). Not your computed sum. If the receipt shows both a subtotal and a net total, use the net total. If the printed total disagrees with the items, the printed total wins and you flag it.

---

## Categories

`c` must be exactly one of these eight strings, or `null`. `s` must be one of that same category's subcategories, or `null`.

{TAXONOMY}

Spell them **exactly** — including `&`, the capitalisation, and the accent in `Coffee & café`. A typo makes the row unusable. Note that `Books & stationery` (Shopping) and `Books & materials` (Education) are different subcategories of different categories.

### The shop decides the category

The same product means different things in different places. Judge what the **spending** was, not what the object is.

| Item | Where | Category |
|---|---|---|
| น้ำแข็ง (ice) | a restaurant / hotpot place | `Food & Dining` |
| น้ำแข็ง (ice) | 7-Eleven | `Groceries` |
| Bottled water | a restaurant | `Food & Dining` / `Drinks & beverages` |
| Bottled water | Makro | `Groceries` / `Drinks & beverages` |
| Paracetamol | a pharmacy | `Health & Wellness` / `Pharmacy & medicine` |

If the whole receipt is from a restaurant, nearly every food line is `Food & Dining`, not `Groceries`.

### Prefer `null` to a guess

- Unsure of the **subcategory** → `"s": null`. A wrong subcategory is worse than none: we would rather have nothing than the wrong one.
- The tax row → always `"c": null`.
- An item you genuinely **cannot identify** from the photo → `"c": null` and `"s": null`, plus a `_flag`. Never infer a category from a price alone.

A `null` is not a failure. Our pipeline routes those items to a human, which is the correct outcome. A confident wrong answer is far more expensive than an honest blank.

---

## Check your work before answering

Verify this balances, within about 3%:

```
sum(all item prices, vat row included) - basket-wide_discount = total_price
```

If it does not balance, something is misread — nine times out of ten it is a unit price where a line total belongs, or a basket discount subtracted from the items as well as recorded. Fix it.

**Do not round a number to force the balance.** If you cannot make it balance honestly, leave your best reading, add a `_flag` saying so, and set `"_needs_review": true`.

Also confirm before answering:
- `id` is exactly the id you were given
- every `price` matches `NN.DD`
- every `c` is one of the eight exact strings or `null`
- every `s` belongs to the category beside it, or is `null`
- there is exactly **one entry per printed item line** — you have neither merged, dropped, nor duplicated a line
- names are in the receipt's own language, untranslated

Output only the JSON object. No explanation, no markdown fences.
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write to this file instead of stdout")
    args = p.parse_args(argv)

    if args.out:
        args.out.write_text(PROMPT, encoding="utf-8")
        print(f"wrote {args.out} ({len(PROMPT)} chars)", file=sys.stderr)
    else:
        sys.stdout.write(PROMPT)


if __name__ == "__main__":
    main()
