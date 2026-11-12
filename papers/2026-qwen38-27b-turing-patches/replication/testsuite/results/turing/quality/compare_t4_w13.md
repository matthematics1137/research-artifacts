# Paper 8 output quality: 2080 Ti lanes vs the laptop (generated 2026-10-31T10:35:44)

Turing rows: `testsuite/results/turing/quality`. Laptop rows: `/opt/p8/repository/testsuite/evals/results`. Datasets: `/opt/p8/repository/testsuite/evals/datasets`.
Statistics as in `testsuite/results/efficiency/tools/hardening_analyze.py` (Wilson 95% CIs, exact two-sided McNemar on discordant pairs, sha256 identity of reasoning + answer); error rows excluded pairwise; latest row per id. A is the reference side, delta = B - A, wall ratio = A / B (above 1: B is faster). Where the two sides returned the text in different shapes (inline vs split reasoning), byte identity is computed on E47's inline text; where they counted tokens differently (encode vs usage), token identity and mean tokens are shown on the matched encode basis and the raw values are marked 'mixed basis'.

## Turing lanes on gsm8k_500

| label | rows (err) | accuracy [95% CI] | acc. on common items (n=500) | mean tokens | truncated | wall/item s | acc./min | decode tok/s (median) | draft acceptance | thinking | ctk dropped | shape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TURING-exl3-mtp | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 327.2 | 0 | 6.45 | 9.083 | 53.4 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w11j2 | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 323.8 | 0 | 5.44 | 10.769 | 63.2 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w11j2-mp05 | 500/500 (0) | 97.0% [95.1, 98.2] | 97.0% | 321.6 | 0 | 5.23 | 11.126 | 65.0 | 0.623 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w13 | 500/500 (0) | 97.2% [95.4, 98.3] | 97.2% | 330.5 | 0 | 4.47 | 13.033 | 79.0 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-mtp-w9g1 | 500/500 (0) | 97.4% [95.6, 98.5] | 97.4% | 324.7 | 0 | 5.62 | 10.392 | 61.1 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-nodraft | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 325.8 | 0 | 14.22 | 4.117 | 23.6 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-w11 | 500/500 (0) | 96.8% [94.9, 98.0] | 96.8% | 328.5 | 0 | 7.95 | 7.306 | 42.6 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-w11-mp05 | 500/500 (0) | 96.8% [94.9, 98.0] | 96.8% | 321.2 | 0 | 7.58 | 7.660 | 43.8 | — | 1.000 | 0 | inline |
| TURING-exl3-nodraft-w9g1 | 500/500 (0) | 97.0% [95.1, 98.2] | 97.0% | 328.5 | 0 | 8.32 | 6.998 | 40.9 | — | 1.000 | 0 | inline |

Common items: those every stock lane (`TURING-<lane>`) completed without error; reruns and variants are scored on their overlap with that set (n shown when smaller) and do not shrink it.

## Requested pairs

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 | TURING-exl3-mtp-w11j2 | TURING-exl3-mtp-w13 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.2% [95.4, 98.3] | -0.40 | 2 / 0 | 0.5000 | 0.882 [0.851, 0.907] | 498/500 | 0.890 (usage) | 323.8 / 330.5 | 5.44 / 4.47 (1.215) | 0 / 0 |
| gsm8k_500 | TURING-exl3-mtp | TURING-exl3-mtp-w13 | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.2% [95.4, 98.3] | -0.40 | 2 / 0 | 0.5000 | 0.882 [0.851, 0.907] | 497/500 | 0.892 (usage) | 327.2 / 330.5 | 6.45 / 4.47 (1.441) | 0 / 0 |

- gsm8k_500 TURING-exl3-mtp-w11j2 vs TURING-exl3-mtp-w13: requested pair
- gsm8k_500 TURING-exl3-mtp vs TURING-exl3-mtp-w13: requested pair

## Notes on the laptop side

- Laptop E47 rows (EFF-h38a_*): RTX 4080 Laptop 12 GB at 80 W, exllamav3 1.4.7 with the lab patches, lab TabbyAPI, daily config (32K cache, Q8, chunk 1024, max_batch_size 1, MTP draft 6 dynamic), mixed CUDA library set (E50), 2026-09-09.
- exllamav3 1.4.7's greedy decoding is not bit-reproducible on this model: the same undrafted config agreed with itself on 50-64 of 100 GSM8K items (E49/E50). Byte identity against E47 is therefore bounded by the laptop's own run-to-run identity and measures numerics, not correctness; the replication test is the paired accuracy (McNemar) and the same-final-answer rate.
- exllamav3 1.5.0 made greedy decoding reproducible on the laptop (E52: 100/100). EFF-h38f_nodraft_rel1 is that engine (first 100 GSM8K-500 items), the closest laptop reference for byte identity with 1.6.5.
- Token counts: the laptop's TabbyAPI rows were re-encoded with /v1/token/encode (the text plus BOS); upstream TabbyAPI and llama-server report usage (generated tokens). Where the two sides differ, token identity, mean tokens and the anatomy's token deltas use the Turing sidecar's encode counts, so both sides share one basis; the raw hardening_analyze values are shown as 'mixed basis'.
- Response shape: the laptop's TabbyAPI returned the reasoning inline in content, with the '</think>' tag. run_quality.sh serves the same shape on every Turing lane (TabbyAPI reasoning parser and tool parsing off, llama-server --reasoning-format none). If a lane split the reasoning anyway, byte identity uses the sidecar's inline fingerprint and the pair gets a CHECK line; without that, a split side shows identity near 0 whatever it generated. The byte-identity bound by the laptop's run-to-run identity applies only when the shapes match.
- EFF-h38b_q4kxl_mtp: UD-Q4_K_XL GGUF, llama.cpp v0.4.0-mtpshared, 30 of 64 layers on the laptop GPU, shared MTP sidecar, 8K context; first 200 GSM8K-500 items (E48). A 4-bit reference, not a replication.

## Turing run records

| label | suite | status | start | duration h | engine | VRAM after load | GPU (max temp, mean power, mean SM clock) |
|---|---|---|---|---|---|---|---|
| VALIDATE-exl3-mtp | gsm8k_500 | done | 2026-10-30T09:20:42-04:00 | 0.03 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8541 MiB | 69.0 C, 239.2 W, 1800.0 MHz |
| VALIDATE-lcpp-iq2xxs | mmlu_500 | done | 2026-10-30T09:22:57-04:00 | 0.11 | llama.cpp d63fbfe794b6 / mtpshared 5266f24da75d | 8394 MiB | 79.0 C, 252.0 W, 1785.0 MHz |
| TURING-exl3-mtp | gsm8k_500 | done | 2026-10-30T09:51:48-04:00 | 0.89 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8350 MiB | 80.0 C, 252.8 W, 1748.9 MHz |
| TURING-exl3-nodraft | gsm8k_500 | done | 2026-10-30T10:45:29-04:00 | 1.99 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8381 MiB | 81.0 C, 254.6 W, 1741.4 MHz |
| TURING-exl3-nodraft-w9g1 | gsm8k_500 | done | 2026-10-30T18:20:20-04:00 | 0.24 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8182 MiB | 80.0 C, 251.4 W, 1674.6 MHz |
| TURING-exl3-mtp-w9g1 | gsm8k_500 | done | 2026-10-30T19:09:42-04:00 | 0.79 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 7995 MiB | 80.0 C, 252.2 W, 1719.0 MHz |
| TURING-exl3-nodraft-w11 | gsm8k_500 | done | 2026-10-30T21:33:28-04:00 | 1.11 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8018 MiB | 80.0 C, 252.7 W, 1656.1 MHz |
| TURING-exl3-mtp-w11j2 | gsm8k_500 | done | 2026-10-30T22:41:10-04:00 | 0.76 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8143 MiB | 80.0 C, 252.3 W, 1706.6 MHz |
| TURING-exl3-mtp | mmlu_500 | done | 2026-10-30T23:30:12-04:00 | 2.05 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8499 MiB | 80.0 C, 252.9 W, 1748.2 MHz |
| TURING-exl3-nodraft | mmlu_500 | done | 2026-10-31T02:12:07-04:00 | 4.4 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 8478 MiB | 81.0 C, 255.2 W, 1750.4 MHz |
| TURING-exl3-nodraft-w11-mp05 | gsm8k_500 | done | 2026-10-31T09:28:30-04:00 | 1.06 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 7823 MiB | 80.0 C, 251.6 W, 1652.7 MHz |
| TURING-exl3-mtp-w11j2-mp05 | gsm8k_500 | done | 2026-10-31T11:11:19-04:00 | 0.74 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 7923 MiB | 80.0 C, 251.3 W, 1698.7 MHz |
| TURING-exl3-mtp-w13 | gsm8k_500 | done | 2026-10-31T12:46:44-04:00 | 0.63 | exllamav3 1.6.5+cu128.torch2.10.0, TabbyAPI b1c225394d0f | 7933 MiB | 80.0 C, 248.9 W, 1637.8 MHz |

