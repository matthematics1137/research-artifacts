# Paper 8 output quality: 2080 Ti lanes vs the laptop (generated 2026-10-30T09:46:56)

Turing rows: `testsuite/results/turing/quality`. Laptop rows: `/opt/p8/repository/testsuite/evals/results`. Datasets: `/opt/p8/repository/testsuite/evals/datasets`.
Statistics as in `testsuite/results/efficiency/tools/hardening_analyze.py` (Wilson 95% CIs, exact two-sided McNemar on discordant pairs, sha256 identity of reasoning + answer); error rows excluded pairwise; latest row per id. A is the reference side, delta = B - A, wall ratio = A / B (above 1: B is faster). Where the two sides returned the text in different shapes (inline vs split reasoning), byte identity is computed on E47's inline text; where they counted tokens differently (encode vs usage), token identity and mean tokens are shown on the matched encode basis and the raw values are marked 'mixed basis'.

## Turing lanes on gsm8k_500

| label | rows (err) | accuracy [95% CI] | acc. on common items (n=500) | mean tokens | truncated | wall/item s | acc./min | decode tok/s (median) | draft acceptance | thinking | ctk dropped | shape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TURING-exl3-mtp | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 327.2 | 0 | 6.45 | 9.083 | 53.4 | 0.622 | 1.000 | 0 | inline |
| TURING-exl3-nodraft | 500/500 (0) | 97.6% [95.9, 98.6] | 97.6% | 325.8 | 0 | 14.22 | 4.117 | 23.6 | — | 1.000 | 0 | inline |

Common items: those every stock lane (`TURING-<lane>`) completed without error; reruns and variants are scored on their overlap with that set (n shown when smaller) and do not shrink it.

## Replication of E47: 2080 Ti vs laptop, same config and suite

| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | wall/item A / B s (A/B) | truncated A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_500 | EFF-h38a_mtp | TURING-exl3-mtp | 500 (0, 0) | 97.6% [95.9, 98.6] | 97.6% [95.9, 98.6] | +0.00 | 0 / 0 | 1.0000 | 0.872 [0.84, 0.898] | 499/500 | 0.878 (encode, n=500; raw 0.874 mixed basis) | 331.9 / 327.2 (encode, matched; raw 331.9 / 327.2 mixed basis) | 6.49 / 6.45 (1.006) | 0 / 0 |
| gsm8k_500 | EFF-h38a_nodraft | TURING-exl3-nodraft | 500 (0, 0) | 97.2% [95.4, 98.3] | 97.6% [95.9, 98.6] | +0.40 | 0 / 2 | 0.5000 | 0.628 [0.585, 0.669] | 498/500 | 0.674 (encode, n=500; raw 0.670 mixed basis) | 327.1 / 325.8 (encode, matched; raw 327.1 / 325.8 mixed basis) | 11.61 / 14.22 (0.816) | 0 / 0 |

- gsm8k_500 EFF-h38a_mtp vs TURING-exl3-mtp: E47 drafted daily config: laptop 4080 (exllamav3 1.4.7-lab) vs 2080 Ti (1.6.5)
- gsm8k_500 EFF-h38a_nodraft vs TURING-exl3-nodraft: E47 undrafted control: laptop vs 2080 Ti

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

- gsm8k_500 TURING-exl3-nodraft vs TURING-exl3-mtp: H38a's design on the 2080 Ti: is the MTP drafter lossless (and sample-identical) on 1.6.5?

Not yet computable: gsm8k_500 TURING-lcpp-iq2s vs TURING-lcpp-iq2s-mtp (no rows for TURING-lcpp-iq2s and TURING-lcpp-iq2s-mtp); gsm8k_500 TURING-lcpp-iq2xxs vs TURING-lcpp-iq2xxs-dflash2 (no rows for TURING-lcpp-iq2xxs and TURING-lcpp-iq2xxs-dflash2); gsm8k_500 TURING-exl3-nodraft vs TURING-lcpp-iq2s (no rows for TURING-lcpp-iq2s); gsm8k_500 TURING-exl3-nodraft vs TURING-lcpp-iq2xxs (no rows for TURING-lcpp-iq2xxs); gsm8k_500 TURING-exl3-mtp vs TURING-lcpp-iq2s-mtp (no rows for TURING-lcpp-iq2s-mtp); gsm8k_500 TURING-exl3-mtp vs TURING-lcpp-iq2xxs-dflash2 (no rows for TURING-lcpp-iq2xxs-dflash2).

## Reference: laptop 4-bit GGUF (E48) vs 2080 Ti lanes

Not yet computable: gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2xxs (no rows for TURING-lcpp-iq2xxs); gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2s (no rows for TURING-lcpp-iq2s); gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2s-mtp (no rows for TURING-lcpp-iq2s-mtp); gsm8k_500 EFF-h38b_q4kxl_mtp vs TURING-lcpp-iq2xxs-dflash2 (no rows for TURING-lcpp-iq2xxs-dflash2).

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

