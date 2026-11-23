# Paper 6 analysis (2026-11-01T02:13:17-0800)

Journals: `/opt/p8/repository/testsuite/paper6/work/journals`; items `/opt/p8/repository/testsuite/paper6/work/longctx_200.jsonl` (sha256 4a801e99693a70d9); permutation 100000 draws, seed 20260919; TOST margin +-10.0 points.

## Flags

- qwen38-27b__IQ3_XXS__k8v4/longctx: 103 of 200 items missing
- qwen38-27b__Q4_K_XL__q4_0/longctx: 10 of 200 items missing
- qwen38-27b__Q4_K_XL__q4_0: --fit chose a different layer count on a restart
- co-primaries use 190/200 items (complete cases across the 8 cells)

## Co-primary outcomes (long context, Holm across the two)

| outcome | estimate (points) | p | Holm p | decision |
|---|---|---|---|---|
| M = I_hybrid - I_GQA != 0 (sign-flip permutation) | +3.68 [-2.79, +10.16] | 0.341 | 0.341 | no rejection |
| I_hybrid equivalent to 0 within +-10.0 (TOST) | +3.68 (90% CI +0.34, +7.03) | 0.00106 | 0.00211 | equivalent within the specified margin (Holm) |

Pre-registered predictions: I_hybrid equivalent to 0, and I_GQA >= I_hybrid.

## Estimands — longctx (primary)

| model | D(Q2) | D(IQ3) | D(Q4) | I = D(Q2)-D(Q4) | TOST p | perm p (I) |
|---|---|---|---|---|---|---|
| qwen38-27b (hybrid) | +1.50 [-0.70, +3.70] | -1.00 [-2.97, +0.97] | -2.11 [-5.38, +1.17] | +3.68 [-0.31, +7.68] | 0.00106 | 0.118 |
| qwen3-8b (gqa) | +0.00 [-3.13, +3.13] | - | +1.00 [-2.95, +4.95] | -1.00 [-5.84, +3.84] | 0.000158 | 0.839 |

M = I_hybrid - I_GQA: +3.68 [-2.79, +10.16] points, n = 190, sign-flip p = 0.341

## Estimands — gsm8k (secondary)

| model | D(Q2) | D(IQ3) | D(Q4) | I = D(Q2)-D(Q4) | TOST p | perm p (I) |
|---|---|---|---|---|---|---|
| qwen38-27b (hybrid) | -1.00 [-4.45, +2.45] | +0.00 [+0.00, +0.00] | - | - | - | - |
| qwen3-8b (gqa) | - | - | - | - | - | - |

M = I_hybrid - I_GQA: - points, n = 0, sign-flip p = -

## Per-cell accuracy

| cell | long-context | 95% Wilson | GSM8K | 95% Wilson | layers | resident | missing (lc/gsm) | trunc (lc/gsm) |
|---|---|---|---|---|---|---|---|---|
| qwen38-27b__Q2_K_XL__f16 | 78.5 (157/200) | 72.3-83.6 | 95.0 (95/100) | 88.8-97.8 | 63 | False | 0/0 | 0/0 |
| qwen38-27b__Q2_K_XL__q8_0 | 79.5 (159/200) | 73.4-84.5 | 96.0 (96/100) | 90.2-98.4 | 66 | True | 0/0 | 1/0 |
| qwen38-27b__Q2_K_XL__q4_0 | 77.0 (154/200) | 70.7-82.3 | 96.0 (96/100) | 90.2-98.4 | 66 | True | 0/0 | 1/0 |
| qwen38-27b__Q2_K_XL__k8v4 | 78.5 (157/200) | 72.3-83.6 | 95.0 (95/100) | 88.8-97.8 | 66 | True | 0/0 | 2/0 |
| qwen38-27b__IQ3_XXS__f16 | 80.0 (160/200) | 73.9-85.0 | 97.0 (97/100) | 91.5-99.0 | 55 | False | 0/0 | 29/0 |
| qwen38-27b__IQ3_XXS__q8_0 | 80.5 (161/200) | 74.5-85.4 | 96.0 (96/100) | 90.2-98.4 | 60 | False | 0/0 | 29/0 |
| qwen38-27b__IQ3_XXS__q4_0 | 81.0 (162/200) | 75.0-85.8 | 97.0 (97/100) | 91.5-99.0 | 62 | False | 0/0 | 24/0 |
| qwen38-27b__IQ3_XXS__k8v4 | 81.4 (79/97) | 72.6-87.9 | - (0/0) | --- | 61 | False | 103/100 | 12/0 |
| qwen38-27b__Q4_K_XL__f16 | 89.5 (179/200) | 84.5-93.0 | - (0/0) | --- | 34 | False | 0/100 | 0/0 |
| qwen38-27b__Q4_K_XL__q8_0 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen38-27b__Q4_K_XL__q4_0 | 91.6 (174/190) | 86.8-94.8 | - (0/0) | --- | 36 | False | 10/100 | 0/0 |
| qwen38-27b__Q4_K_XL__k8v4 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__Q2_K_XL__f16 | 72.5 (145/200) | 65.9-78.2 | - (0/0) | --- | 37 | True | 0/100 | 50/0 |
| qwen3-8b__Q2_K_XL__q8_0 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__Q2_K_XL__q4_0 | 72.5 (145/200) | 65.9-78.2 | - (0/0) | --- | 37 | True | 0/100 | 53/0 |
| qwen3-8b__Q2_K_XL__k8v4 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__IQ3_XXS__f16 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__IQ3_XXS__q8_0 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__IQ3_XXS__q4_0 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__IQ3_XXS__k8v4 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__Q4_K_XL__f16 | 77.0 (154/200) | 70.7-82.3 | - (0/0) | --- | 37 | True | 0/100 | 13/0 |
| qwen3-8b__Q4_K_XL__q8_0 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |
| qwen3-8b__Q4_K_XL__q4_0 | 76.0 (152/200) | 69.6-81.4 | - (0/0) | --- | 37 | True | 0/100 | 15/0 |
| qwen3-8b__Q4_K_XL__k8v4 | - (0/0) | --- | - (0/0) | --- | None | None | 200/100 | 0/0 |

## Secondary — each KV type vs f16 (longctx)

| model | tier | kv | n | acc f16 | acc kv | diff (points) | lost | gained | exact McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| qwen38-27b | Q2_K_XL | q8_0 | 200 | 78.5 | 79.5 | +1.00 [-0.39, +2.39] | 0 | 2 | 0.5 |
| qwen38-27b | Q2_K_XL | q4_0 | 200 | 78.5 | 77.0 | -1.50 [-3.70, +0.70] | 4 | 1 | 0.375 |
| qwen38-27b | Q2_K_XL | k8v4 | 200 | 78.5 | 78.5 | +0.00 [-1.40, +1.40] | 1 | 1 | 1 |
| qwen38-27b | IQ3_XXS | q8_0 | 200 | 80.0 | 80.5 | +0.50 [-1.21, +2.21] | 1 | 2 | 1 |
| qwen38-27b | IQ3_XXS | q4_0 | 200 | 80.0 | 81.0 | +1.00 [-0.97, +2.97] | 1 | 3 | 0.625 |
| qwen38-27b | IQ3_XXS | k8v4 | 97 | 81.4 | 81.4 | +0.00 [-4.11, +4.11] | 2 | 2 | 1 |
| qwen38-27b | Q4_K_XL | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | q4_0 | 190 | 89.5 | 91.6 | +2.11 [-1.17, +5.38] | 3 | 7 | 0.344 |
| qwen38-27b | Q4_K_XL | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q4_0 | 200 | 72.5 | 72.5 | +0.00 [-3.13, +3.13] | 5 | 5 | 1 |
| qwen3-8b | Q2_K_XL | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q4_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q4_0 | 200 | 77.0 | 76.0 | -1.00 [-4.95, +2.95] | 9 | 7 | 0.804 |
| qwen3-8b | Q4_K_XL | k8v4 | 0 | - | - | - | 0 | 0 | 1 |

### K8/V4 vs q8_0 and q4_0 (longctx)

| model | tier | vs | n | diff (points) | k8v4 better | k8v4 worse | exact McNemar p |
|---|---|---|---|---|---|---|---|
| qwen38-27b | Q2_K_XL | q8_0 | 200 | -1.00 [-2.39, +0.39] | 0 | 2 | 0.5 |
| qwen38-27b | Q2_K_XL | q4_0 | 200 | +1.50 [-0.20, +3.20] | 3 | 0 | 0.25 |
| qwen38-27b | IQ3_XXS | q8_0 | 97 | +0.00 [-4.11, +4.11] | 2 | 2 | 1 |
| qwen38-27b | IQ3_XXS | q4_0 | 97 | -2.06 [-4.94, +0.82] | 0 | 2 | 0.5 |
| qwen38-27b | Q4_K_XL | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q4_0 | 0 | - | 0 | 0 | 1 |

## Secondary — each KV type vs f16 (gsm8k)

| model | tier | kv | n | acc f16 | acc kv | diff (points) | lost | gained | exact McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| qwen38-27b | Q2_K_XL | q8_0 | 100 | 95.0 | 96.0 | +1.00 [-0.98, +2.98] | 0 | 1 | 1 |
| qwen38-27b | Q2_K_XL | q4_0 | 100 | 95.0 | 96.0 | +1.00 [-2.45, +4.45] | 1 | 2 | 1 |
| qwen38-27b | Q2_K_XL | k8v4 | 100 | 95.0 | 95.0 | +0.00 [-2.82, +2.82] | 1 | 1 | 1 |
| qwen38-27b | IQ3_XXS | q8_0 | 100 | 97.0 | 96.0 | -1.00 [-2.98, +0.98] | 1 | 0 | 1 |
| qwen38-27b | IQ3_XXS | q4_0 | 100 | 97.0 | 97.0 | +0.00 [+0.00, +0.00] | 0 | 0 | 1 |
| qwen38-27b | IQ3_XXS | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | q4_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q4_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q4_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | k8v4 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q8_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q4_0 | 0 | - | - | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | k8v4 | 0 | - | - | - | 0 | 0 | 1 |

### K8/V4 vs q8_0 and q4_0 (gsm8k)

| model | tier | vs | n | diff (points) | k8v4 better | k8v4 worse | exact McNemar p |
|---|---|---|---|---|---|---|---|
| qwen38-27b | Q2_K_XL | q8_0 | 100 | -1.00 [-2.98, +0.98] | 0 | 1 | 1 |
| qwen38-27b | Q2_K_XL | q4_0 | 100 | -1.00 [-4.45, +2.45] | 1 | 2 | 1 |
| qwen38-27b | IQ3_XXS | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen38-27b | IQ3_XXS | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen38-27b | Q4_K_XL | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q2_K_XL | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | IQ3_XXS | q4_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q8_0 | 0 | - | 0 | 0 | 1 |
| qwen3-8b | Q4_K_XL | q4_0 | 0 | - | 0 | 0 | 1 |

## Secondary — trend of D over weight tiers

| model | contrast | points |
|---|---|---|
| qwen38-27b | D(Q2_K_XL) - D(IQ3_XXS) (long context) | +2.50 [-0.44, +5.44] |
| qwen38-27b | D(IQ3_XXS) - D(Q4_K_XL) (long context) | +1.05 [-2.84, +4.94] |
| qwen3-8b | D(Q2_K_XL) - D(IQ3_XXS) (long context) | - |
| qwen3-8b | D(IQ3_XXS) - D(Q4_K_XL) (long context) | - |

## Secondary — long-context accuracy by family

| cell | single | multikey | multivalue | vartrack |
|---|---|---|---|---|
| qwen38-27b__Q2_K_XL__f16 | 100.0 | 100.0 | 100.0 | 14.0 |
| qwen38-27b__Q2_K_XL__q8_0 | 100.0 | 100.0 | 100.0 | 18.0 |
| qwen38-27b__Q2_K_XL__q4_0 | 100.0 | 100.0 | 100.0 | 8.0 |
| qwen38-27b__Q2_K_XL__k8v4 | 100.0 | 100.0 | 100.0 | 14.0 |
| qwen38-27b__IQ3_XXS__f16 | 100.0 | 100.0 | 100.0 | 20.0 |
| qwen38-27b__IQ3_XXS__q8_0 | 100.0 | 100.0 | 100.0 | 22.0 |
| qwen38-27b__IQ3_XXS__q4_0 | 100.0 | 100.0 | 100.0 | 24.0 |
| qwen38-27b__IQ3_XXS__k8v4 | 100.0 | 100.0 | 100.0 | 25.0 |
| qwen38-27b__Q4_K_XL__f16 | 100.0 | 100.0 | 100.0 | 58.0 |
| qwen38-27b__Q4_K_XL__q8_0 | - | - | - | - |
| qwen38-27b__Q4_K_XL__q4_0 | 100.0 | 100.0 | 100.0 | 66.0 |
| qwen38-27b__Q4_K_XL__k8v4 | - | - | - | - |
| qwen3-8b__Q2_K_XL__f16 | 100.0 | 94.0 | 92.0 | 4.0 |
| qwen3-8b__Q2_K_XL__q8_0 | - | - | - | - |
| qwen3-8b__Q2_K_XL__q4_0 | 100.0 | 98.0 | 88.0 | 4.0 |
| qwen3-8b__Q2_K_XL__k8v4 | - | - | - | - |
| qwen3-8b__IQ3_XXS__f16 | - | - | - | - |
| qwen3-8b__IQ3_XXS__q8_0 | - | - | - | - |
| qwen3-8b__IQ3_XXS__q4_0 | - | - | - | - |
| qwen3-8b__IQ3_XXS__k8v4 | - | - | - | - |
| qwen3-8b__Q4_K_XL__f16 | 100.0 | 100.0 | 98.0 | 10.0 |
| qwen3-8b__Q4_K_XL__q8_0 | - | - | - | - |
| qwen3-8b__Q4_K_XL__q4_0 | 100.0 | 98.0 | 92.0 | 14.0 |
| qwen3-8b__Q4_K_XL__k8v4 | - | - | - | - |

## Secondary — long-context accuracy by needle depth (answer recall per bin)

| cell | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|
| qwen38-27b__Q2_K_XL__f16 | 98.2 (110) | 61.6 (112) | 60.6 (109) | 60.6 (109) | 60.9 (110) |
| qwen38-27b__Q2_K_XL__q8_0 | 97.3 (110) | 64.3 (112) | 62.4 (109) | 62.4 (109) | 62.7 (110) |
| qwen38-27b__Q2_K_XL__q4_0 | 97.3 (110) | 60.7 (112) | 58.7 (109) | 57.8 (109) | 58.2 (110) |
| qwen38-27b__Q2_K_XL__k8v4 | 99.1 (110) | 63.4 (112) | 61.5 (109) | 60.6 (109) | 60.9 (110) |
| qwen38-27b__IQ3_XXS__f16 | 100.0 (110) | 89.3 (112) | 78.9 (109) | 66.1 (109) | 63.6 (110) |
| qwen38-27b__IQ3_XXS__q8_0 | 100.0 (110) | 91.1 (112) | 80.7 (109) | 67.0 (109) | 64.5 (110) |
| qwen38-27b__IQ3_XXS__q4_0 | 100.0 (110) | 87.5 (112) | 79.8 (109) | 67.0 (109) | 65.5 (110) |
| qwen38-27b__IQ3_XXS__k8v4 | 98.4 (64) | 92.2 (64) | 81.5 (54) | 63.4 (41) | 59.5 (42) |
| qwen38-27b__Q4_K_XL__f16 | 83.6 (110) | 81.2 (112) | 80.7 (109) | 80.7 (109) | 80.9 (110) |
| qwen38-27b__Q4_K_XL__q8_0 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen38-27b__Q4_K_XL__q4_0 | 90.7 (107) | 84.8 (105) | 84.3 (102) | 84.5 (103) | 84.3 (102) |
| qwen38-27b__Q4_K_XL__k8v4 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__Q2_K_XL__f16 | 97.3 (110) | 68.8 (112) | 57.8 (109) | 54.1 (109) | 56.4 (110) |
| qwen3-8b__Q2_K_XL__q8_0 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__Q2_K_XL__q4_0 | 99.1 (110) | 70.5 (112) | 58.7 (109) | 56.0 (109) | 56.4 (110) |
| qwen3-8b__Q2_K_XL__k8v4 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__IQ3_XXS__f16 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__IQ3_XXS__q8_0 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__IQ3_XXS__q4_0 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__IQ3_XXS__k8v4 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__Q4_K_XL__f16 | 91.8 (110) | 67.0 (112) | 60.6 (109) | 62.4 (109) | 60.0 (110) |
| qwen3-8b__Q4_K_XL__q8_0 | - (0) | - (0) | - (0) | - (0) | - (0) |
| qwen3-8b__Q4_K_XL__q4_0 | 88.2 (110) | 67.0 (112) | 60.6 (109) | 61.5 (109) | 60.9 (110) |
| qwen3-8b__Q4_K_XL__k8v4 | - (0) | - (0) | - (0) | - (0) | - (0) |

## Equal-memory ranking (fully resident cells only; measured buffers)

### qwen38-27b

| cell | layers | model MiB | KV MiB | KV@16k MiB | KV@32k MiB (computed) | recurrent MiB | compute MiB | device total MiB | nvidia-smi MiB | accuracy | 95% Wilson |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-27b__Q2_K_XL__q8_0 | 66/66 | 8630.57 | 680.0 | 544.0 | 1088.0 | 149.62 | 180.28 | 9640.47 | 9870 | 79.5 | 73.4-84.5 |
| qwen38-27b__Q2_K_XL__k8v4 | 66/66 | 8630.57 | 520.0 | 416.0 | 832.0 | 149.62 | 180.28 | 9480.47 | 9710 | 78.5 | 72.3-83.6 |
| qwen38-27b__Q2_K_XL__q4_0 | 66/66 | 8630.57 | 360.0 | 288.0 | 576.0 | 149.62 | 180.28 | 9320.47 | 9550 | 77.0 | 70.7-82.3 |

Excluded (partly offloaded): qwen38-27b__Q2_K_XL__f16 (63/66), qwen38-27b__IQ3_XXS__f16 (55/66), qwen38-27b__IQ3_XXS__q8_0 (60/66), qwen38-27b__IQ3_XXS__q4_0 (62/66), qwen38-27b__IQ3_XXS__k8v4 (61/66), qwen38-27b__Q4_K_XL__f16 (34/66), qwen38-27b__Q4_K_XL__q4_0 (36/66)


Exchange rate (KV from the resident cells' measured buffers; weight steps from device+host model buffers, which do not depend on placement): Q2_K_XL -> IQ3_XXS: weights +1055 MiB; IQ3_XXS -> Q4_K_XL: weights +6317 MiB


### qwen3-8b

| cell | layers | model MiB | KV MiB | KV@16k MiB | KV@32k MiB (computed) | recurrent MiB | compute MiB | device total MiB | nvidia-smi MiB | accuracy | 95% Wilson |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b__Q4_K_XL__f16 | 37/37 | 4558.28 | 2880.0 | 2304.0 | 4608.0 | None | 116.01 | 7554.29 | 7774 | 77.0 | 70.7-82.3 |
| qwen3-8b__Q4_K_XL__q4_0 | 37/37 | 4558.28 | 810.0 | 648.0 | 1296.0 | None | 132.09 | 5500.37 | 5722 | 76.0 | 69.6-81.4 |
| qwen3-8b__Q2_K_XL__q4_0 | 37/37 | 3000.22 | 810.0 | 648.0 | 1296.0 | None | 132.09 | 3942.31 | 4164 | 72.5 | 65.9-78.2 |
| qwen3-8b__Q2_K_XL__f16 | 37/37 | 3000.22 | 2880.0 | 2304.0 | 4608.0 | None | 116.01 | 5996.23 | 6216 | 72.5 | 65.9-78.2 |

Exchange rate (KV from the resident cells' measured buffers; weight steps from device+host model buffers, which do not depend on placement): Q2_K_XL: KV f16->q4_0 at 16k frees 1656 MiB; Q4_K_XL: KV f16->q4_0 at 16k frees 1656 MiB


## Token counts

- Qwen3.8 tokenizer (frozen items): haystack 16340-16384 (mean 16369.9), full message mean 16431.1
- served tokenizer (qwen38-27b): haystack 16340-16384 (mean 16369.9, n=200)
- served tokenizer (qwen3-8b): haystack 15712-16346 (mean 15977.3, n=200)
- server prompt_tokens (qwen38-27b, long context): median 16445, range 16410-16466
- server prompt_tokens (qwen3-8b, long context): median 16043.0, range 15792-16415
- harness-vs-server prompt token delta (qwen38-27b): 0/1887 nonzero, range 0..0
