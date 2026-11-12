# Paper 8 output quality: 2080 Ti lanes vs the laptop (generated 2026-11-01T15:26:42)

Turing rows: `testsuite/results/turing/quality`. Laptop rows: `/opt/p8/repository/testsuite/evals/results`. Datasets: `/opt/p8/repository/testsuite/evals/datasets`.
Statistics as in `testsuite/results/efficiency/tools/hardening_analyze.py` (Wilson 95% CIs, exact two-sided McNemar on discordant pairs, sha256 identity of reasoning + answer); error rows excluded pairwise; latest row per id. A is the reference side, delta = B - A, wall ratio = A / B (above 1: B is faster). Where the two sides returned the text in different shapes (inline vs split reasoning), byte identity is computed on E47's inline text; where they counted tokens differently (encode vs usage), token identity and mean tokens are shown on the matched encode basis and the raw values are marked 'mixed basis'.

## Turing lanes on gsm8k_500

| label | rows (err) | accuracy [95% CI] | acc. on common items (n=500) | mean tokens | truncated | wall/item s | acc./min | decode tok/s (median) | draft acceptance | thinking | ctk dropped | shape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TURING-exl3-mtp | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 327.2 | 0 | 6.45 | 9.083 | 53.4 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-fp32 | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 325.5 | 0 | 5.95 | 9.834 | 57.5 | 0.623 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w11j2 | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 323.8 | 0 | 5.44 | 10.769 | 63.2 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w11j2-mp05 | 500/500 (0) | 97.0% [95.1, 98.2] | 97.0% | 321.6 | 0 | 5.23 | 11.126 | 65.0 | 0.623 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w13 | 500/500 (0) | 97.2% [95.4, 98.3] | 97.2% | 330.5 | 0 | 4.47 | 13.033 | 79.0 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w13s2 | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 325.9 | 0 | 4.07 | 14.393 | 85.9 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w9g1 | 500/500 (0) | 97.4% [95.6, 98.5] | 97.4% | 324.7 | 0 | 5.62 | 10.392 | 61.1 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-nodraft | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 325.8 | 0 | 14.22 | 4.117 | 23.6 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-fp32 | 500/500 (0) | 97.4% [95.6, 98.5] | 97.4% | 327.3 | 0 | 13.36 | 4.375 | 25.0 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-w11 | 500/500 (0) | 96.8% [94.9, 98.0] | 96.8% | 328.5 | 0 | 7.95 | 7.306 | 42.6 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-w11-mp05 | 500/500 (0) | 96.8% [94.9, 98.0] | 96.8% | 321.2 | 0 | 7.58 | 7.660 | 43.8 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-w9g1 | 500/500 (0) | 97.0% [95.1, 98.2] | 97.0% | 328.5 | 0 | 8.32 | 6.998 | 40.9 | — | 1.000 | 0 | inline |
| TURING-lcpp-iq2xxs | 500/500 (0) | 96.4% [94.4, 97.7] | 96.4% | 376.4 | 0 | 15.69 | 3.687 | 24.7 | — | 1.000 | 0 | inline |
| TURING-lcpp-iq2s | 500/500 (0) | 96.6% [94.6, 97.9] | 96.6% | 364.2 | 0 | 16.01 | 3.620 | 23.4 | — | 1.000 | 0 | inline |
| TURING-lcpp-iq2s-mtp | 500/500 (0) | 97.4% [95.6, 98.5] | 97.4% | 367.6 | 0 | 8.17 | 7.154 | 48.7 | 0.851 | 1.000 | 0 | inline |
| TURING-lcpp-iq2xxs-dflash2 | 500/500 (0) | 96.2% [94.1, 97.6] | 96.2% | 376.5 | 1 | 7.69 | 7.509 | 54.2 | 0.705 | 0.998 | 0 | inline |

Common items: those every stock lane (`TURING-<lane>`) completed without error; reruns and variants are scored on their overlap with that set (n shown when smaller) and do not shrink it.

## Turing lanes on mmlu_500

| label | rows (err) | accuracy [95% CI] | acc. on common items (n=500) | mean tokens | truncated | wall/item s | acc./min | decode tok/s (median) | draft acceptance | thinking | ctk dropped | shape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TURING-exl3-mtp | 500/500 (0) | 86.8% [83.6, 89.5] | 86.8% | 709.6 | 3 | 16.64 | 3.130 | 43.9 | 0.551 | 0.994 | 0 | inline |
| TURING-exl3-nodraft | 500/500 (0) | 86.6% [83.3, 89.3] | 86.6% | 703.5 | 1 | 31.65 | 1.642 | 22.9 | — | 0.996 | 0 | inline |
| TURING-lcpp-iq2xxs | 500/500 (0) | 82.8% [79.2, 85.9] | 82.8% | 710.7 | 2 | 29.42 | 1.688 | 24.6 | — | 0.996 | 0 | inline |
| TURING-lcpp-iq2s | 500/500 (0) | 86.2% [82.9, 88.9] | 86.2% | 698.9 | 6 | 30.45 | 1.698 | 23.4 | — | 0.988 | 0 | inline |
| TURING-lcpp-iq2s-mtp | 500/500 (0) | 86.0% [82.7, 88.8] | 86.0% | 709.7 | 1 | 17.45 | 2.956 | 42.0 | 0.706 | 0.998 | 0 | inline |
| TURING-lcpp-iq2xxs-dflash2 | 500/500 (0) | 83.2% [79.7, 86.2] | 83.2% | 738.4 | 5 | 18.96 | 2.633 | 39.1 | 0.507 | 0.992 | 0 | inline |

Common items: those every stock lane (`TURING-<lane>`) completed without error; reruns and variants are scored on their overlap with that set (n shown when smaller) and do not shrink it.

## Replication of E47: 2080 Ti vs laptop, same config and suite

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.872 [0.84, 0.898] | 499/500 | 0.878 (encode, n=500; raw 0.874 mixed basis) | 331.9 / 327.2 (encode, matched; raw 331.9 / 327.2 mixed basis) | 6.49 / 6.45 (1.006) | 0 / 0 |
| gsm8k_500 | EFF-h38a_nodraft | TURING-exl3-nodraft | 500 (0, 0) | 97.2% [95.4, 98.3] | 97.6% [95.9, 98.6] | +0.40 | 0 / 2 | 0.5000 | 0.628 [0.585, 0.669] | 498/500 | 0.674 (encode, n=500; raw 0.670 mixed basis) | 327.1 / 325.8 (encode, matched; raw 327.1 / 325.8 mixed basis) | 11.61 / 14.22 (0.816) | 0 / 0 |
| mmlu_500 | EFF-h38a_mtp | TURING-exl3-mtp | 500 (0, 0) | 86.8% [83.6, 89.5] | 86.8% [83.6, 89.5] | +0.00 | 4 / 4 | 1.0000 | 0.518 [0.474, 0.561] | 488/494 | 0.536 (encode, n=500; raw 0.530 mixed basis) | 705.3 / 709.6 (encode, matched; raw 705.3 / 709.6 mixed basis) | 16.02 / 16.64 (0.963) | 3 / 3 |
| mmlu_500 | EFF-h38a_nodraft | TURING-exl3-nodraft | 500 (0, 0) | 86.2% [82.9, 88.9] | 86.6% [83.3, 89.3] | +0.40 | 7 / 9 | 0.8036 | 0.198 [0.165, 0.235] | 479/494 | 0.238 (encode, n=500; raw 0.234 mixed basis) | 725.0 / 703.5 (encode, matched; raw 725.0 / 703.5 mixed basis) | 23.78 / 31.65 (0.751) | 5 / 1 |
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp-fp32 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.864 [0.831, 0.891] | 500/500 | 0.870 (encode, n=500; raw 0.864 mixed basis) | 331.9 / 325.5 (encode, matched; raw 331.9 / 325.5 mixed basis) | 6.49 / 5.95 (1.089) | 0 / 0 |
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp-w11j2 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.870 [0.838, 0.897] | 500/500 | 0.878 (encode, n=500; raw 0.874 mixed basis) | 331.9 / 323.8 (encode, matched; raw 331.9 / 323.8 mixed basis) | 6.49 / 5.44 (1.193) | 0 / 0 |
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp-w11j2-mp05 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.0% [95.1, 98.2] | -0.60 | 3 / 0 | 0.2500 | 0.712 [0.671, 0.75] | 497/500 | 0.718 (encode, n=500; raw 0.714 mixed basis) | 331.9 / 321.6 (encode, matched; raw 331.9 / 321.6 mixed basis) | 6.49 / 5.23 (1.240) | 0 / 0 |
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp-w13 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.2% [95.4, 98.3] | -0.40 | 2 / 0 | 0.5000 | 0.860 [0.827, 0.888] | 498/500 | 0.866 (encode, n=500; raw 0.860 mixed basis) | 331.9 / 330.5 (encode, matched; raw 331.9 / 330.5 mixed basis) | 6.49 / 4.47 (1.450) | 0 / 0 |
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp-w13s2 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.868 [0.836, 0.895] | 499/500 | 0.878 (encode, n=500; raw 0.872 mixed basis) | 331.9 / 325.9 (encode, matched; raw 331.9 / 325.9 mixed basis) | 6.49 / 4.07 (1.594) | 0 / 0 |
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp-w9g1 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.4% [95.6, 98.5] | -0.20 | 1 / 0 | 1.0000 | 0.858 [0.825, 0.886] | 499/500 | 0.870 (encode, n=500; raw 0.866 mixed basis) | 331.9 / 324.7 (encode, matched; raw 331.9 / 324.7 mixed basis) | 6.49 / 5.62 (1.154) | 0 / 0 |
| gsm8k_500 | EFF-h38a_nodraft | TURING-exl3-nodraft-fp32 | 500 (0, 0) | 97.2% [95.4, 98.3] | 97.4% [95.6, 98.5] | +0.20 | 0 / 1 | 1.0000 | 0.644 [0.601, 0.685] | 499/500 | 0.690 (encode, n=500; raw 0.684 mixed basis) | 327.1 / 327.3 (encode, matched; raw 327.1 / 327.3 mixed basis) | 11.61 / 13.36 (0.869) | 0 / 0 |
| gsm8k_500 | EFF-h38a_nodraft | TURING-exl3-nodraft-w11 | 500 (0, 0) | 97.2% [95.4, 98.3] | 96.8% [94.9, 98.0] | -0.40 | 3 / 1 | 0.6250 | 0.604 [0.56, 0.646] | 495/500 | 0.654 (encode, n=500; raw 0.652 mixed basis) | 327.1 / 328.5 (encode, matched; raw 327.1 / 328.5 mixed basis) | 11.61 / 7.95 (1.461) | 0 / 0 |
| gsm8k_500 | EFF-h38a_nodraft | TURING-exl3-nodraft-w11-mp05 | 500 (0, 0) | 97.2% [95.4, 98.3] | 96.8% [94.9, 98.0] | -0.40 | 3 / 1 | 0.6250 | 0.536 [0.492, 0.579] | 495/500 | 0.576 (encode, n=500; raw 0.574 mixed basis) | 327.1 / 321.2 (encode, matched; raw 327.1 / 321.2 mixed basis) | 11.61 / 7.58 (1.531) | 0 / 0 |
| gsm8k_500 | EFF-h38a_nodraft | TURING-exl3-nodraft-w9g1 | 500 (0, 0) | 97.2% [95.4, 98.3] | 97.0% [95.1, 98.2] | -0.20 | 1 / 0 | 1.0000 | 0.580 [0.536, 0.622] | 499/500 | 0.626 (encode, n=500; raw 0.624 mixed basis) | 327.1 / 328.5 (encode, matched; raw 327.1 / 328.5 mixed basis) | 11.61 / 8.32 (1.396) | 0 / 0 |

- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp: E47 drafted daily config: laptop 4080 (exllamav3 1.4.7-lab) vs 2080 Ti (1.6.5)
- gsm8k_500 EFF-h38a_nodraft vs TURING-exl3-nodraft: E47 undrafted control: laptop vs 2080 Ti
- mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp: E47 drafted daily config: laptop 4080 (exllamav3 1.4.7-lab) vs 2080 Ti (1.6.5)
- mmlu_500 EFF-h38a_nodraft vs TURING-exl3-nodraft: E47 undrafted control: laptop vs 2080 Ti
- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp-fp32: E47 replication with variant 'fp32' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp-w11j2: E47 replication with variant 'w11j2' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp-w11j2-mp05: E47 replication with variant 'w11j2-mp05' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp-w13: E47 replication with variant 'w13' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp-w13s2: E47 replication with variant 'w13s2' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp-w9g1: E47 replication with variant 'w9g1' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-fp32: E47 replication with variant 'fp32' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-w11: E47 replication with variant 'w11' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-w11-mp05: E47 replication with variant 'w11-mp05' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)
- gsm8k_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-w9g1: E47 replication with variant 'w9g1' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)

Not yet computable: mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp-fp32 (no rows for TURING-exl3-mtp-fp32); mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp-w11j2 (no rows for TURING-exl3-mtp-w11j2); mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp-w11j2-mp05 (no rows for TURING-exl3-mtp-w11j2-mp05); mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp-w13 (no rows for TURING-exl3-mtp-w13); mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp-w13s2 (no rows for TURING-exl3-mtp-w13s2); mmlu_500 EFF-h38a_mtp vs TURING-exl3-mtp-w9g1 (no rows for TURING-exl3-mtp-w9g1); mmlu_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-fp32 (no rows for TURING-exl3-nodraft-fp32); mmlu_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-w11 (no rows for TURING-exl3-nodraft-w11); mmlu_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-w11-mp05 (no rows for TURING-exl3-nodraft-w11-mp05); mmlu_500 EFF-h38a_nodraft vs TURING-exl3-nodraft-w9g1 (no rows for TURING-exl3-nodraft-w9g1).

## Engine reference: laptop exllamav3 1.5.0 (reproducible greedy) vs 2080 Ti

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 first 100 | EFF-h38f_nodraft_rel1 | TURING-exl3-nodraft | 100 (0, 0) | 98.0% [93.0, 99.4] | 99.0% [94.6, 99.8] | +1.00 | 0 / 1 | 1.0000 | 0.760 [0.668, 0.833] | 99/100 | 0.780 (encode, n=100; raw 0.780 mixed basis) | 323.2 / 323.3 (encode, matched; raw 323.2 / 323.4 mixed basis) | 12.09 / 13.84 (0.874) | 0 / 0 |
| gsm8k_500 first 100 | EFF-h38f_nodraft_rel1 | TURING-exl3-mtp | 100 (0, 0) | 98.0% [93.0, 99.4] | 99.0% [94.6, 99.8] | +1.00 | 0 / 1 | 1.0000 | 0.750 [0.657, 0.825] | 99/100 | 0.780 (encode, n=100; raw 0.780 mixed basis) | 323.2 / 325.3 (encode, matched; raw 323.2 / 325.4 mixed basis) | 12.09 / 6.5 (1.861) | 0 / 0 |

- gsm8k_500 EFF-h38f_nodraft_rel1 vs TURING-exl3-nodraft: laptop exllamav3 1.5.0 (reproducible greedy, E52) vs 2080 Ti 1.6.5; first 100 items
- gsm8k_500 EFF-h38f_nodraft_rel1 vs TURING-exl3-mtp: laptop 1.5.0 undrafted vs 2080 Ti 1.6.5 drafted; first 100 items

## Pairs on the 2080 Ti itself

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 | TURING-exl3-nodraft | TURING-exl3-mtp | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.894 [0.864, 0.918] | 500/500 | 0.898 (usage) | 325.8 / 327.2 | 14.22 / 6.45 (2.206) | 0 / 0 |
| gsm8k_500 | TURING-lcpp-iq2s | TURING-lcpp-iq2s-mtp | 500 (0, 0) | 96.6% [94.6, 97.9] | 97.4% [95.6, 98.5] | +0.80 | 0 / 4 | 0.1250 | 0.472 [0.429, 0.516] | 494/500 | 0.506 (usage) | 364.2 / 367.6 | 16.01 / 8.17 (1.960) | 0 / 0 |
| gsm8k_500 | TURING-lcpp-iq2xxs | TURING-lcpp-iq2xxs-dflash2 | 500 (0, 0) | 96.4% [94.4, 97.7] | 96.2% [94.1, 97.6] | -0.20 | 3 / 2 | 1.0000 | 0.374 [0.333, 0.417] | 491/499 | 0.412 (usage) | 376.4 / 376.5 | 15.69 / 7.69 (2.041) | 0 / 1 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-lcpp-iq2s | 500 (0, 0) | 97.6% [95.9, 98.6] | 96.6% [94.6, 97.9] | -1.00 | 8 / 3 | 0.2266 | 0.000 [0.0, 0.008] | 485/500 | 0.006 (usage) | 325.8 / 364.2 | 14.22 / 16.01 (0.888) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-lcpp-iq2xxs | 500 (0, 0) | 97.6% [95.9, 98.6] | 96.4% [94.4, 97.7] | -1.20 | 9 / 3 | 0.1460 | 0.000 [0.0, 0.008] | 482/500 | 0.012 (usage) | 325.8 / 376.4 | 14.22 / 15.69 (0.907) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-lcpp-iq2s-mtp | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.4% [95.6, 98.5] | -0.20 | 4 / 3 | 1.0000 | 0.000 [0.0, 0.008] | 489/500 | 0.004 (usage) | 327.2 / 367.6 | 6.45 / 8.17 (0.789) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-lcpp-iq2xxs-dflash2 | 500 (0, 0) | 97.6% [95.9, 98.6] | 96.2% [94.1, 97.6] | -1.40 | 10 / 3 | 0.0923 | 0.000 [0.0, 0.008] | 481/499 | 0.014 (usage) | 327.2 / 376.5 | 6.45 / 7.69 (0.839) | 0 / 1 |
| mmlu_500 | TURING-exl3-nodraft | TURING-exl3-mtp | 500 (0, 0) | 86.6% [83.3, 89.3] | 86.8% [83.6, 89.5] | +0.20 | 2 / 3 | 1.0000 | 0.588 [0.544, 0.63] | 492/496 | 0.610 (usage) | 703.5 / 709.6 | 31.65 / 16.64 (1.902) | 1 / 3 |
| mmlu_500 | TURING-lcpp-iq2s | TURING-lcpp-iq2s-mtp | 500 (0, 0) | 86.2% [82.9, 88.9] | 86.0% [82.7, 88.8] | -0.20 | 11 / 10 | 1.0000 | 0.128 [0.102, 0.16] | 473/494 | 0.164 (usage) | 698.9 / 709.7 | 30.45 / 17.45 (1.745) | 6 / 1 |
| mmlu_500 | TURING-lcpp-iq2xxs | TURING-lcpp-iq2xxs-dflash2 | 500 (0, 0) | 82.8% [79.2, 85.9] | 83.2% [79.7, 86.2] | +0.40 | 17 / 19 | 0.8679 | 0.070 [0.051, 0.096] | 455/493 | 0.098 (usage) | 710.7 / 738.4 | 29.42 / 18.96 (1.552) | 2 / 5 |
| mmlu_500 | TURING-exl3-nodraft | TURING-lcpp-iq2s | 500 (0, 0) | 86.6% [83.3, 89.3] | 86.2% [82.9, 88.9] | -0.40 | 18 / 16 | 0.8642 | 0.000 [0.0, 0.008] | 452/493 | 0.010 (usage) | 703.5 / 698.9 | 31.65 / 30.45 (1.039) | 1 / 6 |
| mmlu_500 | TURING-exl3-nodraft | TURING-lcpp-iq2xxs | 500 (0, 0) | 86.6% [83.3, 89.3] | 82.8% [79.2, 85.9] | -3.80 | 38 / 19 | 0.0163 | 0.000 [0.0, 0.008] | 432/495 | 0.002 (usage) | 703.5 / 710.7 | 31.65 / 29.42 (1.076) | 1 / 2 |
| mmlu_500 | TURING-exl3-mtp | TURING-lcpp-iq2s-mtp | 500 (0, 0) | 86.8% [83.6, 89.5] | 86.0% [82.7, 88.8] | -0.80 | 21 / 17 | 0.6271 | 0.000 [0.0, 0.008] | 450/496 | 0.008 (usage) | 709.6 / 709.7 | 16.64 / 17.45 (0.953) | 3 / 1 |
| mmlu_500 | TURING-exl3-mtp | TURING-lcpp-iq2xxs-dflash2 | 500 (0, 0) | 86.8% [83.6, 89.5] | 83.2% [79.7, 86.2] | -3.60 | 35 / 17 | 0.0175 | 0.000 [0.0, 0.008] | 436/492 | 0.006 (usage) | 709.6 / 738.4 | 16.64 / 18.96 (0.878) | 3 / 5 |

- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-mtp: H38a's design on the 2080 Ti: is the MTP drafter lossless (and sample-identical) on 1.6.5?
- gsm8k_500 TURING-lcpp-iq2s vs TURING-lcpp-iq2s-mtp: llama.cpp shared MTP sidecar: lossless?
- gsm8k_500 TURING-lcpp-iq2xxs vs TURING-lcpp-iq2xxs-dflash2: DFlash2 drafter: lossless?
- gsm8k_500 TURING-exl3-nodraft vs TURING-lcpp-iq2s: EXL3 2.0 bpw vs UD-IQ2_S, undrafted, same card
- gsm8k_500 TURING-exl3-nodraft vs TURING-lcpp-iq2xxs: EXL3 2.0 bpw vs UD-IQ2_XXS, undrafted, same card
- gsm8k_500 TURING-exl3-mtp vs TURING-lcpp-iq2s-mtp: best drafted lanes: EXL3+MTP vs IQ2_S+MTP
- gsm8k_500 TURING-exl3-mtp vs TURING-lcpp-iq2xxs-dflash2: EXL3+MTP vs IQ2_XXS+DFlash2
- mmlu_500 TURING-exl3-nodraft vs TURING-exl3-mtp: H38a's design on the 2080 Ti: is the MTP drafter lossless (and sample-identical) on 1.6.5?
- mmlu_500 TURING-lcpp-iq2s vs TURING-lcpp-iq2s-mtp: llama.cpp shared MTP sidecar: lossless?
- mmlu_500 TURING-lcpp-iq2xxs vs TURING-lcpp-iq2xxs-dflash2: DFlash2 drafter: lossless?
- mmlu_500 TURING-exl3-nodraft vs TURING-lcpp-iq2s: EXL3 2.0 bpw vs UD-IQ2_S, undrafted, same card
- mmlu_500 TURING-exl3-nodraft vs TURING-lcpp-iq2xxs: EXL3 2.0 bpw vs UD-IQ2_XXS, undrafted, same card
- mmlu_500 TURING-exl3-mtp vs TURING-lcpp-iq2s-mtp: best drafted lanes: EXL3+MTP vs IQ2_S+MTP
- mmlu_500 TURING-exl3-mtp vs TURING-lcpp-iq2xxs-dflash2: EXL3+MTP vs IQ2_XXS+DFlash2

## Reference: laptop 4-bit GGUF (E48) vs 2080 Ti lanes

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 first 200 | EFF-h38b_q4kxl_mtp | TURING-lcpp-iq2xxs | 200 (0, 0) | 98.0% [95.0, 99.2] | 97.0% [93.6, 98.6] | -1.00 | 5 / 3 | 0.7266 | 0.000 [0.0, 0.019] (shapes differ: not comparable) | 190/200 | 0.015 (usage) | 321.9 / 365.9 | 30.24 / 15.23 (1.986) | 0 / 0 |
| gsm8k_500 first 200 | EFF-h38b_q4kxl_mtp | TURING-lcpp-iq2s | 200 (0, 0) | 98.0% [95.0, 99.2] | 97.5% [94.3, 98.9] | -0.50 | 3 / 2 | 1.0000 | 0.000 [0.0, 0.019] (shapes differ: not comparable) | 193/200 | 0.005 (usage) | 321.9 / 358.7 | 30.24 / 15.76 (1.919) | 0 / 0 |
| gsm8k_500 first 200 | EFF-h38b_q4kxl_mtp | TURING-lcpp-iq2s-mtp | 200 (0, 0) | 98.0% [95.0, 99.2] | 99.0% [96.4, 99.7] | +1.00 | 1 / 3 | 0.6250 | 0.000 [0.0, 0.019] (shapes differ: not comparable) | 195/200 | 0.005 (usage) | 321.9 / 359.2 | 30.24 / 7.95 (3.804) | 0 / 0 |
| gsm8k_500 first 200 | EFF-h38b_q4kxl_mtp | TURING-lcpp-iq2xxs-dflash2 | 200 (0, 0) | 98.0% [95.0, 99.2] | 96.5% [93.0, 98.3] | -1.50 | 5 / 2 | 0.4531 | 0.000 [0.0, 0.019] (shapes differ: not comparable) | 191/199 | 0.025 (usage) | 321.9 / 374.3 | 30.24 / 7.6 (3.978) | 0 / 1 |

- gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2xxs: laptop UD-Q4_K_XL + shared MTP (E48), first 200 items: a 4-bit reference, not a replication. CHECK: response shapes differ (A split, B inline): the split side's server moved the reasoning to reasoning_content and dropped '</think>', so its raw content_sha256 cannot match; the split side's sidecar has no inline fingerprints, so byte identity is not comparable
- gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2s: laptop UD-Q4_K_XL + shared MTP (E48), first 200 items: a 4-bit reference, not a replication. CHECK: response shapes differ (A split, B inline): the split side's server moved the reasoning to reasoning_content and dropped '</think>', so its raw content_sha256 cannot match; the split side's sidecar has no inline fingerprints, so byte identity is not comparable
- gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2s-mtp: laptop UD-Q4_K_XL + shared MTP (E48), first 200 items: a 4-bit reference, not a replication. CHECK: response shapes differ (A split, B inline): the split side's server moved the reasoning to reasoning_content and dropped '</think>', so its raw content_sha256 cannot match; the split side's sidecar has no inline fingerprints, so byte identity is not comparable
- gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2xxs-dflash2: laptop UD-Q4_K_XL + shared MTP (E48), first 200 items: a 4-bit reference, not a replication. CHECK: response shapes differ (A split, B inline): the split side's server moved the reasoning to reasoning_content and dropped '</think>', so its raw content_sha256 cannot match; the split side's sidecar has no inline fingerprints, so byte identity is not comparable

## Lane variants on the 2080 Ti (one setting changed against the stock lane)

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-fp32 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.882 [0.851, 0.907] | 499/500 | 0.890 (usage) | 327.2 / 325.5 | 6.45 / 5.95 (1.083) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-w11j2 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.886 [0.855, 0.911] | 499/500 | 0.890 (usage) | 327.2 / 323.8 | 6.45 / 5.44 (1.186) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-w11j2-mp05 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.0% [95.1, 98.2] | -0.60 | 3 / 0 | 0.2500 | 0.704 [0.663, 0.742] | 496/500 | 0.710 (usage) | 327.2 / 321.6 | 6.45 / 5.23 (1.233) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-w13 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.2% [95.4, 98.3] | -0.40 | 2 / 0 | 0.5000 | 0.882 [0.851, 0.907] | 497/500 | 0.892 (usage) | 327.2 / 330.5 | 6.45 / 4.47 (1.441) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-w13s2 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.874 [0.842, 0.9] | 500/500 | 0.878 (usage) | 327.2 / 325.9 | 6.45 / 4.07 (1.585) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-w9g1 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.4% [95.6, 98.5] | -0.20 | 1 / 0 | 1.0000 | 0.886 [0.855, 0.911] | 498/500 | 0.896 (usage) | 327.2 / 324.7 | 6.45 / 5.62 (1.146) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-exl3-nodraft-fp32 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.4% [95.6, 98.5] | -0.20 | 1 / 0 | 1.0000 | 0.896 [0.866, 0.92] | 499/500 | 0.902 (usage) | 325.8 / 327.3 | 14.22 / 13.36 (1.065) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-exl3-nodraft-w11 | 500 (0, 0) | 97.6% [95.9, 98.6] | 96.8% [94.9, 98.0] | -0.80 | 4 / 0 | 0.1250 | 0.666 [0.624, 0.706] | 495/500 | 0.694 (usage) | 325.8 / 328.5 | 14.22 / 7.95 (1.789) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-exl3-nodraft-w11-mp05 | 500 (0, 0) | 97.6% [95.9, 98.6] | 96.8% [94.9, 98.0] | -0.80 | 4 / 0 | 0.1250 | 0.572 [0.528, 0.615] | 495/500 | 0.594 (usage) | 325.8 / 321.2 | 14.22 / 7.58 (1.876) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-exl3-nodraft-w9g1 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.0% [95.1, 98.2] | -0.60 | 3 / 0 | 0.2500 | 0.692 [0.65, 0.731] | 497/500 | 0.716 (usage) | 325.8 / 328.5 | 14.22 / 8.32 (1.710) | 0 / 0 |

- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-fp32: variant 'fp32' against the stock lane, same card
- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-w11j2: variant 'w11j2' against the stock lane, same card
- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-w11j2-mp05: variant 'w11j2-mp05' against the stock lane, same card
- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-w13: variant 'w13' against the stock lane, same card
- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-w13s2: variant 'w13s2' against the stock lane, same card
- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-w9g1: variant 'w9g1' against the stock lane, same card
- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-fp32: variant 'fp32' against the stock lane, same card
- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-w11: variant 'w11' against the stock lane, same card
- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-w11-mp05: variant 'w11-mp05' against the stock lane, same card
- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-w9g1: variant 'w9g1' against the stock lane, same card

Not yet computable: mmlu_500 TURING-exl3-mtp vs TURING-exl3-mtp-fp32 (no rows for TURING-exl3-mtp-fp32); mmlu_500 TURING-exl3-mtp vs TURING-exl3-mtp-w11j2 (no rows for TURING-exl3-mtp-w11j2); mmlu_500 TURING-exl3-mtp vs TURING-exl3-mtp-w11j2-mp05 (no rows for TURING-exl3-mtp-w11j2-mp05); mmlu_500 TURING-exl3-mtp vs TURING-exl3-mtp-w13 (no rows for TURING-exl3-mtp-w13); mmlu_500 TURING-exl3-mtp vs TURING-exl3-mtp-w13s2 (no rows for TURING-exl3-mtp-w13s2); mmlu_500 TURING-exl3-mtp vs TURING-exl3-mtp-w9g1 (no rows for TURING-exl3-mtp-w9g1); mmlu_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-fp32 (no rows for TURING-exl3-nodraft-fp32); mmlu_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-w11 (no rows for TURING-exl3-nodraft-w11); mmlu_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-w11-mp05 (no rows for TURING-exl3-nodraft-w11-mp05); mmlu_500 TURING-exl3-nodraft vs TURING-exl3-nodraft-w9g1 (no rows for TURING-exl3-nodraft-w9g1).

## Notes on the laptop side

- Laptop E47 rows (EFF-h38a_*): RTX 4080 Laptop 12 GB at 80 W, exllamav3 1.4.7 with the lab patches, lab TabbyAPI, daily config (32K cache, Q8, chunk 1024, max_batch_size 1, MTP draft 6 dynamic), mixed CUDA library set (E50), 2026-09-09.
- exllamav3 1.4.7's greedy decoding is not bit-reproducible on this model: the same undrafted config agreed with itself on 50-64 of 100 GSM8K items (E49/E50). Byte identity against E47 is therefore bounded by the laptop's own run-to-run identity and measures numerics, not correctness; the replication test is the paired accuracy (McNemar) and the same-final-answer rate.
- exllamav3 1.5.0 made greedy decoding reproducible on the laptop (E52: 100/100). EFF-h38f_nodraft_rel1 is that engine (first 100 GSM8K-500 items), the closest laptop reference for byte identity with 1.6.5.
- Token counts: the laptop's TabbyAPI rows were re-encoded with /v1/token/encode (the text plus BOS); upstream TabbyAPI and llama-server report usage (generated tokens). Where the two sides differ, token identity, mean tokens and the anatomy's token deltas use the Turing sidecar's encode counts, so both sides share one basis; the raw hardening_analyze values are shown as 'mixed basis'.
- Response shape: the laptop's TabbyAPI returned the reasoning inline in content, with the '</think>' tag. run_quality.sh serves the same shape on every Turing lane (TabbyAPI reasoning parser and tool parsing off, llama-server --reasoning-format none). If a lane split the reasoning anyway, byte identity uses the sidecar's inline fingerprint and the pair gets a CHECK line; without that, a split side shows identity near 0 whatever it generated. The byte-identity bound by the laptop's run-to-run identity applies only when the shapes match.
- EFF-h38b_q4kxl_mtp: UD-Q4_K_XL GGUF, llama.cpp v0.4.0-mtpshared, 30 of 64 layers on the laptop GPU, shared MTP sidecar, 8K context; first 200 GSM8K-500 items (E48). A 4-bit reference, not a replication.

