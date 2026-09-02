# Final results - pooled over all 68 held-out REAL receipts (real_val + real_test)

Note: real_val (34 rows) was used for checkpoint selection, so the pooled figure is mildly
optimistic; real_test (34 rows) alone is the clean number. Both are reported separately in
phase5_real_test_comparison.md.

| model | n | MONEY EXACT | Prices exact | Count exact | Total exact | Price recall | Name sim | Name ok | JSON valid | Schema valid | Shop exact | Shop sim | Item F1 | Exact match | Discount F1 | Recon pass | s/receipt | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B zero-shot (raw output) | 68 | 48.5% | 51.5% | 61.8% | 79.4% | 81.9% | 59.6 | 58.1% | 98.5% | 86.8% | 10.3% | 53.1 | 45.4% | 0.0% | 0.0% | 70.6% | - | - |
| Qwen3.5-2B zero-shot + code-layer normalization | 68 | 51.5% | 52.9% | 61.8% | 86.8% | 83.9% | 59.5 | 58.8% | 98.5% | 94.1% | 10.3% | 53.1 | 46.7% | 0.0% | 0.0% | 70.6% | - | - |
| qwen3.5-2b-qlora ckpt-550 (raw output) | 68 | 64.7% | 64.7% | 76.5% | 82.4% | 82.9% | 75.7 | 73.7% | 98.5% | 92.6% | 66.2% | 85.5 | 62.4% | 10.3% | 0.0% | 86.8% | - | - |
| qwen3.5-2b-qlora ckpt-550 + code-layer normalization | 68 | 67.6% | 67.6% | 76.5% | 88.2% | 86.8% | 73.8 | 70.2% | 98.5% | 98.5% | 66.2% | 85.5 | 62.4% | 10.3% | 0.0% | 86.8% | - | - |
