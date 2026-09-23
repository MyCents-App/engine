# Joint model evaluation

73 receipts from `val.jsonl` (real=73), greedy decoding, max_new_tokens=1024, raw model output.

Extraction columns are `eval_metrics.py`'s, on the answer with c/s stripped, so they compare directly with checkpoint-550's. Category columns score only items paired to a gold item by price and name (`items paired` is that share): a dropped item costs money-exact, not the categories of every item after it.

Targets (CATEGORIZATION.md section 4): category accuracy >= 90%, sub accuracy >= 70%, **sub precision >= 90%** -- a wrong subcategory is worse than null. Extraction must not fall below the legacy row.

## Checkpoints

| checkpoint | MONEY EXACT | count exact | total exact | price recall | shop exact | name ok | JSON valid | truncated | **cat acc** | sub acc | sub prec | sub prec (labelled) | taxonomy valid | items paired | s/receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b-qlora/checkpoint-550 (legacy prompt) | **79.5%** | 87.7% | 94.5% | 92.0% | 32.9% | 85.5% | 100.0% | 0 | **-** | - | - | - | - | - | 3.51 |
| qwen3.5-2b-joint/checkpoint-25 | **43.8%** | 52.1% | 84.9% | 89.1% | 69.9% | 83.5% | 97.3% | 0 | **82.0%** | 62.4% | 36.6% | 64.3% | 100.0% | 93.8% | 5.43 |
| qwen3.5-2b-joint/checkpoint-50 | **65.8%** | 71.2% | 90.4% | 88.3% | 76.7% | 85.7% | 98.6% | 1 | **86.5%** | 56.8% | 32.2% | 57.3% | 100.0% | 93.8% | 5.40 |
| qwen3.5-2b-joint/checkpoint-75 | **63.0%** | 68.5% | 91.8% | 92.0% | 75.3% | 85.3% | 100.0% | 0 | **89.2%** | 68.6% | 48.5% | 74.6% | 99.3% | 96.2% | 5.12 |
| qwen3.5-2b-joint/checkpoint-100 | **68.5%** | 75.3% | 93.2% | 92.8% | 76.7% | 86.6% | 100.0% | 0 | **92.9%** | 68.8% | 48.5% | 74.8% | 100.0% | 97.7% | 5.01 |
| qwen3.5-2b-joint/checkpoint-125 | **67.1%** | 78.1% | 91.8% | 90.5% | 76.7% | 84.9% | 100.0% | 0 | **91.6%** | 68.1% | 46.5% | 72.3% | 100.0% | 95.8% | 4.69 |
| qwen3.5-2b-joint/checkpoint-150 | **65.8%** | 71.2% | 90.4% | 90.5% | 80.8% | 86.0% | 100.0% | 0 | **93.1%** | 69.1% | 51.6% | 79.0% | 100.0% | 94.6% | 4.55 |
| qwen3.5-2b-joint/checkpoint-175 | **64.4%** | 78.1% | 91.8% | 88.8% | 79.5% | 87.9% | 100.0% | 0 | **91.9%** | 70.4% | 50.0% | 77.2% | 100.0% | 94.6% | 4.56 |
| qwen3.5-2b-joint/checkpoint-200 | **65.8%** | 76.7% | 91.8% | 89.3% | 79.5% | 86.8% | 100.0% | 0 | **92.7%** | 73.5% | 53.2% | 81.3% | 99.6% | 95.4% | 4.58 |
| qwen3.5-2b-joint/checkpoint-225 | **65.8%** | 76.7% | 91.8% | 89.7% | 79.5% | 87.4% | 100.0% | 0 | **92.7%** | 75.0% | 53.7% | 82.3% | 100.0% | 95.4% | 4.56 |
| qwen3.5-2b-joint/checkpoint-231 | **65.8%** | 78.1% | 91.8% | 89.5% | 79.5% | 86.7% | 100.0% | 0 | **93.1%** | 74.3% | 53.4% | 81.5% | 100.0% | 95.4% | 4.55 |

Best money exact: **qwen3.5-2b-joint/checkpoint-100**. Best category accuracy: **qwen3.5-2b-joint/checkpoint-231**.

## On the 35 receipts the legacy checkpoint never trained on

38 of these 73 receipts are in `real_train.jsonl`, which the --legacy checkpoint was trained on (matched by total and price multiset). Its row above is inflated by them; **this table is the fair comparison.**

| checkpoint | MONEY EXACT | count exact | total exact | price recall | shop exact | name ok | JSON valid | truncated | **cat acc** | sub acc | sub prec | sub prec (labelled) | taxonomy valid | items paired | s/receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b-qlora/checkpoint-550 (legacy prompt) | **62.9%** | 77.1% | 88.6% | 85.0% | 31.4% | 82.9% | 100.0% | 0 | **-** | - | - | - | - | - | 3.51 |
| qwen3.5-2b-joint/checkpoint-25 | **40.0%** | 51.4% | 80.0% | 89.1% | 80.0% | 83.9% | 100.0% | 0 | **77.1%** | 62.0% | 39.6% | 62.9% | 100.0% | 92.9% | 5.43 |
| qwen3.5-2b-joint/checkpoint-50 | **65.7%** | 71.4% | 82.9% | 85.5% | 80.0% | 87.8% | 97.1% | 1 | **84.2%** | 66.7% | 41.9% | 67.7% | 100.0% | 89.8% | 5.40 |
| qwen3.5-2b-joint/checkpoint-75 | **65.7%** | 71.4% | 88.6% | 92.8% | 77.1% | 86.2% | 100.0% | 0 | **88.2%** | 72.9% | 53.7% | 76.1% | 98.4% | 93.7% | 5.12 |
| qwen3.5-2b-joint/checkpoint-100 | **65.7%** | 71.4% | 88.6% | 92.5% | 80.0% | 85.9% | 100.0% | 0 | **91.1%** | 70.0% | 53.3% | 75.4% | 100.0% | 96.9% | 5.01 |
| qwen3.5-2b-joint/checkpoint-125 | **68.6%** | 74.3% | 85.7% | 91.8% | 80.0% | 86.8% | 100.0% | 0 | **90.0%** | 73.6% | 53.5% | 76.8% | 100.0% | 94.5% | 4.69 |
| qwen3.5-2b-joint/checkpoint-150 | **62.9%** | 65.7% | 82.9% | 90.8% | 85.7% | 88.6% | 100.0% | 0 | **94.0%** | 69.6% | 55.8% | 77.4% | 100.0% | 91.3% | 4.55 |
| qwen3.5-2b-joint/checkpoint-175 | **62.9%** | 74.3% | 85.7% | 89.8% | 82.9% | 89.3% | 100.0% | 0 | **91.6%** | 69.0% | 55.1% | 76.6% | 100.0% | 93.7% | 4.56 |
| qwen3.5-2b-joint/checkpoint-200 | **62.9%** | 68.6% | 85.7% | 88.3% | 82.9% | 91.4% | 100.0% | 0 | **93.2%** | 81.2% | 62.9% | 84.8% | 100.0% | 92.9% | 4.58 |
| qwen3.5-2b-joint/checkpoint-225 | **62.9%** | 68.6% | 85.7% | 88.3% | 82.9% | 91.4% | 100.0% | 0 | **93.2%** | 81.2% | 62.9% | 84.8% | 100.0% | 92.9% | 4.56 |
| qwen3.5-2b-joint/checkpoint-231 | **62.9%** | 68.6% | 85.7% | 88.3% | 82.9% | 91.4% | 100.0% | 0 | **93.2%** | 81.2% | 62.9% | 84.8% | 100.0% | 92.9% | 4.55 |

Best money exact here: **qwen3.5-2b-joint/checkpoint-125**.

## qwen3.5-2b-qlora/checkpoint-550 (legacy prompt)

Not money-exact: 15 -- item count wrong 9, a price is wrong 6

- `photo-11` 7-Eleven -- a price is wrong
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-132` UNIQLO -- 4 items for 5
- `photo-142` new balance -- a price is wrong
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-172` 7-Eleven -- 3 items for 2
- `photo-187` % Arabica -- 3 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-196` ยาหนึ่ง -- a price is wrong
- `photo-203` 7-Eleven -- a price is wrong
- `photo-33` 7-Eleven -- a price is wrong
- `photo-4` KFC -- 2 items for 1
- `photo-66` Seoul wang -- 5 items for 4
- `photo-72` Ayar House Bar & Restaurant -- 2 items for 4

## qwen3.5-2b-joint/checkpoint-25

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 87 | 77.0% |
| Groceries | 125 | 90.4% |
| Transport | 3 | 0.0% |
| Shopping | 18 | 83.3% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 0.0% |
| null | 4 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 20
- Groceries -> Food & Dining: 11
- Shopping -> Groceries: 3
- null -> Groceries: 3
- Transport -> Shopping: 2
- Transport -> Food & Dining: 1
- null -> Health & Wellness: 1
- Health & Wellness -> Shopping: 1

Not money-exact: 41 -- item count wrong 33, a price is wrong 5, unparseable 2, total is wrong 1

- `photo-100` 7-Eleven -- 13 items for 4
- `photo-101` Retro Teahouse -- 2 items for 1
- `photo-11` 7-Eleven -- 5 items for 4
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- unparseable
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-124` UNIQLO -- 2 items for 1
- `photo-130` CP Axtra PCL -- 19 items for 23
- `photo-132` UNIQLO -- a price is wrong
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- unparseable
- `photo-141` 麻麻 - MAMA ม่าม่า -- 6 items for 5
- `photo-142` new balance -- 4 items for 3
- `photo-146` ZARA -- 2 items for 1
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-16` King Kong Skewer -- 14 items for 13
- `photo-160` 7-Eleven -- 3 items for 2
- `photo-161` 7-Eleven -- 2 items for 1
- `photo-162` 7-Eleven -- 2 items for 1
- `photo-168` 7-Eleven -- a price is wrong
- ... and 21 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-50

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 83 | 75.9% |
| Groceries | 127 | 98.4% |
| Transport | 2 | 100.0% |
| Shopping | 21 | 81.0% |
| Health & Wellness | 5 | 60.0% |
| Entertainment | 1 | 100.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 20
- Shopping -> Groceries: 4
- null -> Groceries: 4
- Health & Wellness -> Groceries: 2
- Groceries -> Food & Dining: 2
- null -> Food & Dining: 1

Not money-exact: 25 -- item count wrong 20, a price is wrong 4, unparseable 1

- `photo-100` 7-Eleven -- 6 items for 4
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 5 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-126` CP Axtra PCL -- 21 items for 20
- `photo-129` CP Axtra PCL -- 27 items for 26
- `photo-130` CP Axtra PCL -- 20 items for 23
- `photo-132` UNIQLO -- 4 items for 5
- `photo-140` MUJI -- 4 items for 2
- `photo-141` 麻麻 - MAMA ม่าม่า -- unparseable
- `photo-142` new balance -- 5 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-150` Mr. Bao MALATANG -- 2 items for 3
- `photo-16` King Kong Skewer -- 12 items for 13
- `photo-168` 7-Eleven -- a price is wrong
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-196` ยาหนึ่ง -- 1 items for 2
- `photo-4` KFC -- 2 items for 1
- ... and 5 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-75

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 89 | 96.6% |
| Groceries | 126 | 91.3% |
| Transport | 2 | 100.0% |
| Shopping | 21 | 85.7% |
| Health & Wellness | 6 | 33.3% |
| Entertainment | 1 | 0.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Groceries -> Food & Dining: 11
- Health & Wellness -> Groceries: 4
- Food & Dining -> Groceries: 3
- Shopping -> Groceries: 3
- null -> Food & Dining: 3
- null -> Groceries: 2
- Entertainment -> Food & Dining: 1

Not money-exact: 27 -- item count wrong 23, a price is wrong 3, total is wrong 1

- `photo-100` 7-Eleven -- 5 items for 4
- `photo-101` Retro Teahouse -- 2 items for 1
- `photo-11` 7-Eleven -- a price is wrong
- `photo-113` BONCHON CHICKEN -- 3 items for 2
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 4 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-126` CP Axtra PCL -- 21 items for 20
- `photo-129` CP Axtra PCL -- 27 items for 26
- `photo-130` CP Axtra PCL -- 20 items for 23
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 4 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-16` King Kong Skewer -- 12 items for 13
- `photo-160` 7-Eleven -- a price is wrong
- `photo-161` 7-Eleven -- 2 items for 1
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-198` กุ้งเผา ปลาเผา ป.ทะเลเผา -- 3 items for 2
- ... and 7 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-100

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 89 | 95.5% |
| Groceries | 129 | 95.3% |
| Transport | 2 | 100.0% |
| Shopping | 22 | 100.0% |
| Health & Wellness | 6 | 66.7% |
| Entertainment | 1 | 0.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Groceries -> Food & Dining: 5
- Food & Dining -> Groceries: 4
- null -> Groceries: 3
- null -> Food & Dining: 2
- Health & Wellness -> Groceries: 2
- Groceries -> Shopping: 1
- Entertainment -> Food & Dining: 1

Not money-exact: 23 -- item count wrong 18, a price is wrong 5

- `photo-101` Retro Teahouse -- 2 items for 1
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 5 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-126` CP Axtra PCL -- 21 items for 20
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- a price is wrong
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 5 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 3 items for 5
- `photo-16` King Kong Skewer -- 12 items for 13
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-187` % Arabica -- 3 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-198` กุ้งเผา ปลาเผา ป.ทะเลเผา -- 4 items for 2
- `photo-33` 7-Eleven -- 3 items for 4
- `photo-4` KFC -- 2 items for 1
- `photo-63` 7-Eleven -- 2 items for 3
- ... and 3 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-125

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 90 | 91.1% |
| Groceries | 124 | 96.0% |
| Transport | 2 | 100.0% |
| Shopping | 21 | 90.5% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 100.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 8
- Groceries -> Food & Dining: 4
- Shopping -> Groceries: 2
- null -> Groceries: 2
- null -> Food & Dining: 2
- Groceries -> Health & Wellness: 1
- Health & Wellness -> Groceries: 1
- null -> Shopping: 1

Not money-exact: 24 -- item count wrong 16, a price is wrong 7, total is wrong 1

- `photo-100` 7-Eleven -- a price is wrong
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- 1 items for 2
- `photo-116` Starbucks Coffee -- 5 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-126` CP Axtra PCL -- 21 items for 20
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- 19 items for 23
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 4 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-16` King Kong Skewer -- a price is wrong
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-187` % Arabica -- 3 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-33` 7-Eleven -- total is wrong
- `photo-4` KFC -- 2 items for 1
- `photo-58` เฮียง ลูกชิ้นปลา -- a price is wrong
- `photo-63` 7-Eleven -- 2 items for 3
- ... and 4 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-150

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 90 | 95.6% |
| Groceries | 122 | 95.1% |
| Transport | 2 | 100.0% |
| Shopping | 21 | 90.5% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 100.0% |
| null | 4 | 0.0% |

Most common category errors (gold -> predicted):

- Groceries -> Food & Dining: 6
- null -> Groceries: 4
- Food & Dining -> Groceries: 4
- Shopping -> Groceries: 2
- Health & Wellness -> Groceries: 1

Not money-exact: 25 -- item count wrong 21, a price is wrong 4

- `photo-100` 7-Eleven -- a price is wrong
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- 1 items for 2
- `photo-116` Starbucks Coffee -- 3 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-126` CP Axtra PCL -- 21 items for 20
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- 19 items for 23
- `photo-132` UNIQLO -- 4 items for 5
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 5 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 3 items for 5
- `photo-150` Mr. Bao MALATANG -- 2 items for 3
- `photo-16` King Kong Skewer -- 14 items for 13
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-33` 7-Eleven -- 3 items for 4
- `photo-4` KFC -- 2 items for 1
- `photo-42` 7-Eleven -- 2 items for 3
- ... and 5 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-175

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 87 | 88.5% |
| Groceries | 125 | 98.4% |
| Transport | 2 | 100.0% |
| Shopping | 20 | 90.0% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 100.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 10
- null -> Groceries: 3
- Shopping -> Groceries: 2
- null -> Food & Dining: 1
- Health & Wellness -> Groceries: 1
- Groceries -> Food & Dining: 1
- Groceries -> Shopping: 1
- null -> Shopping: 1

Not money-exact: 26 -- item count wrong 16, a price is wrong 9, total is wrong 1

- `photo-100` 7-Eleven -- a price is wrong
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 3 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-126` CP Axtra PCL -- 21 items for 20
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- 19 items for 23
- `photo-132` UNIQLO -- 4 items for 5
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 4 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-150` Mr. Bao MALATANG -- 2 items for 3
- `photo-16` King Kong Skewer -- a price is wrong
- `photo-168` 7-Eleven -- a price is wrong
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-33` 7-Eleven -- total is wrong
- `photo-4` KFC -- 2 items for 1
- ... and 6 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-200

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 89 | 94.4% |
| Groceries | 125 | 96.0% |
| Transport | 2 | 100.0% |
| Shopping | 20 | 90.0% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 100.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 5
- Groceries -> Food & Dining: 4
- null -> Groceries: 3
- Shopping -> Groceries: 2
- null -> Food & Dining: 1
- Health & Wellness -> Groceries: 1
- Groceries -> Shopping: 1
- null -> Shopping: 1

Not money-exact: 25 -- item count wrong 17, a price is wrong 8

- `photo-100` 7-Eleven -- a price is wrong
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 3 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- 21 items for 23
- `photo-132` UNIQLO -- 4 items for 5
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 4 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-150` Mr. Bao MALATANG -- 2 items for 3
- `photo-16` King Kong Skewer -- a price is wrong
- `photo-168` 7-Eleven -- a price is wrong
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-33` 7-Eleven -- 3 items for 4
- `photo-4` KFC -- 2 items for 1
- `photo-42` 7-Eleven -- 2 items for 3
- ... and 5 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-225

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 89 | 93.3% |
| Groceries | 125 | 96.8% |
| Transport | 2 | 100.0% |
| Shopping | 20 | 90.0% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 100.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 6
- Groceries -> Food & Dining: 3
- null -> Groceries: 3
- Shopping -> Groceries: 2
- null -> Food & Dining: 1
- Health & Wellness -> Groceries: 1
- Groceries -> Shopping: 1
- null -> Shopping: 1

Not money-exact: 25 -- item count wrong 17, a price is wrong 8

- `photo-100` 7-Eleven -- a price is wrong
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 3 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- 21 items for 23
- `photo-132` UNIQLO -- 4 items for 5
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 4 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-150` Mr. Bao MALATANG -- 2 items for 3
- `photo-16` King Kong Skewer -- a price is wrong
- `photo-168` 7-Eleven -- a price is wrong
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-33` 7-Eleven -- 3 items for 4
- `photo-4` KFC -- 2 items for 1
- `photo-42` 7-Eleven -- 2 items for 3
- ... and 5 more (see the _examples.jsonl beside the cache)

## qwen3.5-2b-joint/checkpoint-231

| gold category | items | accuracy |
|---|---|---|
| Food & Dining | 89 | 94.4% |
| Groceries | 125 | 96.8% |
| Transport | 2 | 100.0% |
| Shopping | 20 | 90.0% |
| Health & Wellness | 6 | 83.3% |
| Entertainment | 1 | 100.0% |
| null | 5 | 0.0% |

Most common category errors (gold -> predicted):

- Food & Dining -> Groceries: 5
- Groceries -> Food & Dining: 3
- null -> Groceries: 3
- Shopping -> Groceries: 2
- null -> Food & Dining: 1
- Health & Wellness -> Groceries: 1
- Groceries -> Shopping: 1
- null -> Shopping: 1

Not money-exact: 25 -- item count wrong 16, a price is wrong 9

- `photo-100` 7-Eleven -- a price is wrong
- `photo-11` 7-Eleven -- a price is wrong
- `photo-115` TURTLE -- a price is wrong
- `photo-116` Starbucks Coffee -- 3 items for 2
- `photo-121` Moshi Moshi -- 6 items for 5
- `photo-129` CP Axtra PCL -- a price is wrong
- `photo-130` CP Axtra PCL -- 21 items for 23
- `photo-132` UNIQLO -- 4 items for 5
- `photo-137` ร้านอาหารสุกี้ตี๋น้อย -- 4 items for 3
- `photo-140` MUJI -- 4 items for 2
- `photo-142` new balance -- 4 items for 3
- `photo-15` หมูกระทะเฮียเบิร์ด -- 4 items for 5
- `photo-16` King Kong Skewer -- a price is wrong
- `photo-168` 7-Eleven -- a price is wrong
- `photo-172` 7-Eleven -- 4 items for 2
- `photo-19` TSB-GO -- 1 items for 2
- `photo-33` 7-Eleven -- 3 items for 4
- `photo-4` KFC -- 2 items for 1
- `photo-42` 7-Eleven -- 2 items for 3
- `photo-58` เฮียง ลูกชิ้นปลา -- a price is wrong
- ... and 5 more (see the _examples.jsonl beside the cache)
