# Per-receipt scorecard

20 receipts. Money is scored exact-match -- a price is either the printed number or it is wrong. Names are scored on how closely they resemble the label.

| Receipt | Total amount exact | Line price accuracy | Item names acceptable | Item count exact | Exact on money | Shop name acceptable | Schema-valid JSON |
|---|---|---|---|---|---|---|---|
| photo-1 | Yes | 2/2 (100%) | 2/2 (100%) | Yes | Yes | Exact | Valid |
| photo-2 | Yes | 3/3 (100%) | 3/3 (100%) | Yes | Yes | Exact | Valid |
| photo-3 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Exact | Valid |
| photo-4 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Yes | Valid |
| photo-5 | Yes | 1/2 (50%) | 0/1 (0%) | No | No | Exact | Valid |
| photo-6 | Yes | 3/3 (100%) | 3/3 (100%) | Yes | Yes | Exact | Valid |
| photo-7 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Exact | Valid |
| photo-8 | Yes | 3/4 (75%) | 2/3 (67%) | Yes | No | Exact | Valid |
| photo-9 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Yes | Valid |
| photo-10 | Yes | 1/1 (100%) | 0/1 (0%) | Yes | Yes | Exact | Valid |
| photo-11 | Yes | 2/2 (100%) | 2/2 (100%) | Yes | Yes | Exact | Valid |
| photo-12 | Yes | 2/2 (100%) | 2/2 (100%) | Yes | Yes | Exact | Valid |
| photo-13 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Exact | Valid |
| photo-14 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Exact | Valid |
| photo-15 | Yes | 2/2 (100%) | 2/2 (100%) | Yes | Yes | No | Valid |
| photo-16 | Yes | 2/2 (100%) | 2/2 (100%) | Yes | Yes | Yes | Valid |
| photo-17 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | No | Valid |
| photo-18 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Yes | Valid |
| photo-19 | Yes | 1/1 (100%) | 1/1 (100%) | Yes | Yes | Exact | Valid |
| photo-20 | Yes | 2/2 (100%) | 2/2 (100%) | No | No | No | Valid |
| **All 20** | **20/20 (100%)** | **32/34 (94%)** | **29/32 (91%)** | **18/20 (90%)** | **17/20 (85%)** | **17/20 (85%)** | **20/20 (100%)** |

### How "item names acceptable" is calculated

Two names are compared character by character, and the overlap is expressed as a percentage:

    score = 100 x 2 x (characters the two names share) / (length of both names added together)

`Toast` vs `Roast` share o, a, s, t -- 4 characters out of 5 + 5 -- so they score
(2 x 4) / 10 = 80%. A name counts as **acceptable at 60% or above**.

Two rules matter as much as the formula:

1. **A name is only scored if its price already matched.** Gold and predicted lines are paired
   on the exact same price first; names are then compared within those pairs. A line whose price
   is wrong is counted as a price error and is not scored again as a name error.
2. **Shop names are compared word by word instead**, because the gold labels carry branch and
   legal-entity decoration the app does not need. Predicting "H&M" for
   "H&M (Mega Bangna, Branch No. 00013)" scores 16% character by character purely for being
   shorter, but 100% word by word, since every word it gave is in the gold name. Item names
   cannot use this rule: most are Thai, which puts no spaces between words.


### What each column means

- **Total amount exact** -- the receipt's grand total matches the label exactly.
- **Line price accuracy** -- of the labelled item lines, how many the pipeline produced at exactly the right price.
- **Item names acceptable** -- of those price-matched lines, how many names cleared the 60% bar.
- **Item count exact** -- the pipeline returned the same number of lines as the label.
- **Exact on money** -- every line price *and* the total right, with no extra or missing line. The strictest column.
- **Shop name acceptable** -- the merchant name is the right merchant, allowing for dropped branch and legal-entity decoration.
- **Schema-valid JSON** -- the output parsed and obeyed the field contract.
