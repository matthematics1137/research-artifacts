# Test 3, w13 extension (2026-11-01 19:14:22 EST)

Pre-registration: `research/prereg/2026-11-01-four-resources-tests-1-3.md`, test 3. Plan: `w13_predictions-plan.md` (sha256 0167248eaa06f2f1). Predictions: `w13_predictions.md` (written 2026-11-01 19:14:14 EST, sha256 8fba305681f92702), before final2's per-task w13 rows were read. Bench set `20261101T191124` (desktop queue step `x-t3-w13-bench`). Code: `tools/four_resources/test3_w13.py`.

## Verdicts

- **w13 mode 2 ms/token within ±10% (13.97 ms pooled at fill 0; 15.02 at 28K)**: refuted. fill 0: predicted 12.12 ms against 13.97 (-13.2%); fill 28000: predicted 13.79 ms against 15.02 (-8.2%).
- **Kill: the error exceeds 20% on any task (w13 mode 2)**: not triggered. fill 0: code -13.8%, math -10.5%, prose -14.9%; fill 28000: code -14.6%, math -11.6%, prose -3.3%.
- **The drafted-versus-undrafted gain difference: sign and size within 2x**: holds. fill 0: predicted D +21.5 points (drafted +24.4%, undrafted +2.9%) against measured +15.4 (drafted +20.0%, undrafted +4.6%); fill 28000: predicted D +22.5 points (drafted +25.2%, undrafted +2.7%) against measured +16.7 (drafted +21.4%, undrafted +4.7%).
- **Acceptance within ±0.03 per task**: not run. needs the private logging patch (deferred).

## w13 mode 2, predicted against measured (ms per token, final2 pooled over three rounds)

| fill | task | measured | predicted | error |
|---|---|---|---|---|
| 0 | code | 11.43 | 9.85 | -13.8% |
| 0 | math | 10.81 | 9.68 | -10.5% |
| 0 | prose | 19.67 | 16.74 | -14.9% |
| 0 | pooled | 13.97 | 12.12 | -13.2% |
| 28000 | code | 12.44 | 10.63 | -14.6% |
| 28000 | math | 11.74 | 10.37 | -11.6% |
| 28000 | prose | 20.45 | 19.78 | -3.3% |
| 28000 | pooled | 15.02 | 13.79 | -8.2% |

## Other arms and declared variants (pooled, error against final2)

- Fill 0: w13s1 -10.3%; adopted MTP (composed for the gain test) -7.0%; variants for mode 2: base_w9 -6.8%, adopted_counts -10.9%, idle_per_pass -3.7%.
- Fill 28000: w13s1 -8.0%; adopted MTP (composed for the gain test) -5.1%; variants for mode 2: base_w9 -3.6%, adopted_counts -7.2%, idle_per_pass +0.8%.

## Why the drafted gain is larger

Whole-model verify-pass matrix time over a one-row pass (census role mix), by dispatch: w11b M=3 1.86, M=4 1.90, M=5 1.92; sk1 M=3 1.47, M=4 1.48, M=5 1.52; sk2 M=3 1.30, M=4 1.31, M=5 1.35. Dispatch, from the patches: the adopted path leaves the int8 kernel for the fp16 path from three rows; w13 routes 3-16 rows to split-K inside rafatxf's shape envelope, and with EXL3_GEMV_SK=2 outside it too.

