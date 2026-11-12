# Paper 8 output quality: 2080 Ti lanes vs the laptop (generated 2026-11-01T15:25:16)

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

## Requested pairs

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 | TURING-exl3-nodraft-fp32 | TURING-exl3-mtp-fp32 | 500 (0, 0) | 97.4% [95.6, 98.5] | 97.6% [95.9, 98.6] | +0.20 | 0 / 1 | 1.0000 | 0.918 [0.891, 0.939] | 498/500 | 0.924 (usage) | 327.3 / 325.5 | 13.36 / 5.95 (2.243) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft-fp32 | TURING-exl3-mtp | 500 (0, 0) | 97.4% [95.6, 98.5] | 97.6% [95.9, 98.6] | +0.20 | 0 / 1 | 1.0000 | 0.888 [0.857, 0.913] | 499/500 | 0.894 (usage) | 327.3 / 327.2 | 13.36 / 6.45 (2.072) | 0 / 0 |
| gsm8k_500 | TURING-exl3-nodraft | TURING-exl3-mtp-fp32 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.882 [0.851, 0.907] | 499/500 | 0.890 (usage) | 325.8 / 325.5 | 14.22 / 5.95 (2.389) | 0 / 0 |

- gsm8k_500 TURING-exl3-nodraft-fp32 vs TURING-exl3-mtp-fp32: requested pair
- gsm8k_500 TURING-exl3-nodraft-fp32 vs TURING-exl3-mtp: requested pair
- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-mtp-fp32: requested pair

## Notes on the laptop side

- Laptop E47 rows (EFF-h38a_*): RTX 4080 Laptop 12 GB at 80 W, exllamav3 1.4.7 with the lab patches, lab TabbyAPI, daily config (32K cache, Q8, chunk 1024, max_batch_size 1, MTP draft 6 dynamic), mixed CUDA library set (E50), 2026-09-09.
- exllamav3 1.4.7's greedy decoding is not bit-reproducible on this model: the same undrafted config agreed with itself on 50-64 of 100 GSM8K items (E49/E50). Byte identity against E47 is therefore bounded by the laptop's own run-to-run identity and measures numerics, not correctness; the replication test is the paired accuracy (McNemar) and the same-final-answer rate.
- exllamav3 1.5.0 made greedy decoding reproducible on the laptop (E52: 100/100). EFF-h38f_nodraft_rel1 is that engine (first 100 GSM8K-500 items), the closest laptop reference for byte identity with 1.6.5.
- Token counts: the laptop's TabbyAPI rows were re-encoded with /v1/token/encode (the text plus BOS); upstream TabbyAPI and llama-server report usage (generated tokens). Where the two sides differ, token identity, mean tokens and the anatomy's token deltas use the Turing sidecar's encode counts, so both sides share one basis; the raw hardening_analyze values are shown as 'mixed basis'.
- Response shape: the laptop's TabbyAPI returned the reasoning inline in content, with the '</think>' tag. run_quality.sh serves the same shape on every Turing lane (TabbyAPI reasoning parser and tool parsing off, llama-server --reasoning-format none). If a lane split the reasoning anyway, byte identity uses the sidecar's inline fingerprint and the pair gets a CHECK line; without that, a split side shows identity near 0 whatever it generated. The byte-identity bound by the laptop's run-to-run identity applies only when the shapes match.
- EFF-h38b_q4kxl_mtp: UD-Q4_K_XL GGUF, llama.cpp v0.4.0-mtpshared, 30 of 64 layers on the laptop GPU, shared MTP sidecar, 8K context; first 200 GSM8K-500 items (E48). A 4-bit reference, not a replication.

