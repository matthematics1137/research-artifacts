# Claim–evidence map: self-drafted 2-bit Qwen3.8-27B on an 80 W, 12 GB laptop GPU

Run every public check from the artifact root:

```bash
python3 check_claims.py
python3 replication/evals/run_eval.py --selftest
```

`check_claims.py` first verifies every file under `data/` (and `replication/`)
against `data/EXPORT_MANIFEST.json`, whose own SHA-256 is fixed in the checker.
It then recomputes each number below from the row-level data and compares it
with the value printed in the paper at the paper's rounding; a mismatch fails
the run. The paper's single numerical source is the laboratory's campaign
document (experiments E1–E52); the rows here are the raw evidence that document
cites, exported without model output text, prompts, benchmark text, or local
paths. The export manifest records the SHA-256 of every private input; each
equals the file at laboratory commit `9a62b2da50ef8198a3a12dc3837358444cb4d710`.

## Headline and supporting claims

The September 28 revision preserves numbers but narrows interpretation:
non-rejection is not equivalence; one repeat is not a general noise floor;
mechanisms apply to the tested implementation. E7 EXL3 rates include request
wall time, unlike llama.cpp server-timed decode.

| Paper claim | Public evidence | Check |
|---|---|---|
| Drafting raises EXL3 completion throughput from about 29 to 48.7–50.4 tok/s on code/math and 35.7 on prose (E7) | `data/rows/e7_rows.jsonl`, labels `exl3_147_nodraft`, `exl3_147_mtpdyn` (means over two sessions) | end-to-end request means; not kernel-only decode |
| Supporting near-cap speed/energy ratios (not the title) | same; at the cap energy per token is power over rate | prose 1.22×, math 1.74× |
| 0.56–0.58 tokens per joule sustained at the cap on both engines (E25) | `data/rows/tpj_rows.jsonl`, `data/rows/tpj_gpu_*.jsonl` | tokens / integrated energy; token sums from per-request rows |
| No measured accuracy decrease on 1,000 paired greedy items (not equivalence): 97.6 vs 97.2% (GSM8K-500), 86.8 vs 86.2% (MMLU-500), McNemar p ≥ 0.50 (E47) | `data/items/EFF-h38a_mtp/`, `data/items/EFF-h38a_nodraft/`, `data/dataset_item_order.json` | accuracy, Wilson intervals, exact McNemar, byte identity by `content_sha256` |
| Observed two-bit vs higher-bit scores: 99.0 vs 98.0% on 200 paired GSM8K items, p = 0.63, nearly equal mean length (E48), not equivalence | `data/items/EFF-h38b_q4kxl_mtp/`, first 200 of `EFF-h38a_mtp` in dataset order | as above; mean completion tokens 321.9 / 322.0 |
| Edited-prompt latency 10.1 → 5.0 s at a 1,024-token checkpoint grid (E43) | `data/rows/edit_rows.jsonl` | named edit-position comparison; host-memory and cold-prefill costs stated |
| The drafter's 2,432 MiB; 10,226 → 8,050 MiB with one config line (E27, E28) | `data/logs/vram_log_extracts.json` (TabbyAPI load lines) | 10,226 − 7,794; default vs `max_batch_size: 1` |
| Four recurrent-state history slots: 4 × 5 × 48 × 3 MiB = 2,880 MiB drafted vs 576 undrafted (E28) | allocation shape from the engine reading | 48 × 128 × 128 fp32 = 3 MiB, arithmetic only |
| Tested stock-head proposal comparison: a sampled-draft verifier would gain +0.018 over 9,821 positions (E35) | `data/rows/spec_meas.jsonl`; prompt ranges from the B6 log lines in `data/logs/vram_log_extracts.json` | per-set means of acc, p(x\*), q(x\*), Σmin(p,q) |
| Prefill is 74% fp16 GEMM (E37) | `data/rows/prefill_profile.jsonl` (`profile` step) | GPU-time buckets |
| The 2-bit model reasons 1.28× longer than the 4-bit GGUF on 25 MATH items at temperature 0 (E40) | `data/traces/e40_trace_stats.jsonl` | ratio of mean completion tokens |
| On-policy recalibration (E32) and bit reallocation (E41) miss the predicted token reduction; off-policy calibration worsens measured lengths (E42); marker suppression trims about 5% (E45) | `data/items/EFF-exl3_20cot_147_nomtp8k/`, `EFF-exl3_20varA_147_nomtp8k/`, `EFF-exl3_20off_147_nomtp8k/`, `EFF-b10_*`, comparators `EFF-exl3_147_mtp4/`, `EFF-exl3_147_mtpdyn{,-r2}/` | paired token changes, discordant counts, exact tests |
| Temperature-0 decoding not bit-reproducible: undrafted re-runs identical on 50/100, 64/100 under a clean library set, drafted re-runs on 91/100 (E49, E50) | `data/items/EFF-h38c_*`, `EFF-h38d_*` | byte identity on the first 100 GSM8K-500 items |
| On exllamav3's development branch (E51) and in the released 1.5.0 (E52) two undrafted greedy runs are identical on 100/100 | `data/items/EFF-h38e_nodraft_dev1/2/`, `EFF-h38f_nodraft_rel1/2/` | same |
| One same-config GSM8K-100 re-roll moved 3 items (98 → 95, p = 0.25; E26) | `data/items/EFF-exl3_147_mtpdyn/`, `EFF-exl3_147_mtpdyn-r2/` | paired counts |

## Table 1 (artifacts)

| Cell | Evidence | Check |
|---|---|---|
| Byte counts (EXL3 as the sum of two shards) | `data/hash_audit_2026-09-06.json`; `environment/model_artifacts.json` | exact |
| GGUF historical whole-file/base-parameter ratios 2.49 / 2.16 / 5.22 | same | approximate base-model denominator ≈26.9 × 10⁹, not stored-tensor count; caption notes Q4 = 5.14 with 27.3207 × 10⁹ stored parameters |
| EXL3 2.00 / 2.20 bpw | branch names and the quantization recipe (nominal per-tensor precision) | not re-derivable from bytes; stated in the caption |
| Resident 7,794 / 8,050 (EXL3), 8,242 (2.20), 8,356 / 7,248 (GGUF), 1,124 (sidecar), 2,462 (DFlash2) MiB | `data/logs/vram_log_extracts.json`; `data/rows/footprint_rows.jsonl` | exact; sidecar and drafter as differences of E3 rows |

## Table 2 (tokens per joule)

| Cell | Evidence | Check |
|---|---|---|
| E25: minutes, tokens, tok/s, mean and idle W, tok/J gross and net | `data/rows/tpj_rows.jsonl` summary rows and per-request rows | exact / rounded; net recomputed from energy, idle, and minutes |
| E25 capped share | `data/rows/tpj_gpu_*.jsonl` throttle reasons | power-cap-only share of all samples (see discrepancies) |
| E25 per-prompt tok/J | per-request rows | request means |
| E29 aggregate, per-stream, mean W, tok/J, ×1.59 / ×2.12 / ×2.61 | `data/rows/e29_rows.jsonl` summary rows | exact / rounded |

## Hardening block (H38; Tables A6–A9, E51, E52)

| Result | Evidence | Check |
|---|---|---|
| E47 accuracy, Wilson intervals, discordant 0/2 and 6/9, p = 0.50 / 0.61, pooled 91.7 / 92.2%, 6/11, p = 0.33 | `data/items/EFF-h38a_*` | recomputed |
| E47 byte identity 318/500 [59.3, 67.7], 105/500 [17.7, 24.8], 423/1,000; token-count identity 68.4 / 24.8% | `content_sha256`, `completion_tokens` | recomputed |
| E47 mean tokens, walls, speed-ups ×1.79 / ×1.48, truncations | per-item fields | recomputed |
| E47 identity by length tercile, tercile edges, hazards ε ≈ 1.6 / 2.6 × 10⁻³ | per-item fields; median lengths 280 / ≈ 600 | recomputed (hazard with the stated median lengths) |
| E47 differing pairs: 179/182 same final answer, 180 same correctness; MMLU 374 of 387 parsed, 380; median length difference 1 / −1 (mean −25) | `final_answer_sha256`, `correct`, `completion_tokens` | recomputed |
| E48 accuracy, intervals, 1/3 discordant, p = 0.63, 321.9 / 322.0 tokens, 30.24 / 6.30 s, ×4.8, 195/200 same final answer, 34 / 16 items > 20% longer / shorter | `data/items/EFF-h38b_q4kxl_mtp/`, `EFF-h38a_mtp/` | recomputed |
| E49 (i), E47 pairs, (iii): 91 / 63 / 50 of 100 with intervals and hazards; 50/50 same answer; walls 6.39 / 5.87 and 11.74 / 9.86 s | `data/items/EFF-h38a_*`, `EFF-h38c_*` | recomputed |
| E49 (ii): 20 identical / 5 divergent traces, gaps 0.000–0.031 nat, runner-up 5/5, first divergence 39–357 (median 92, 4–61%), 23/25 same ending, near-ties 0.35% / 0.13% of 8,475 positions | `data/traces/h38c_pairs.jsonl`, `data/traces/h38c_trace_gaps.jsonl` | recomputed |
| E50: 64/100 [54.2, 72.7] clean vs clean; 58 / 61 against the mixed runs; drafted 91/100; walls 10.81 / 10.53, 6.39 / 5.92 s; loaded libraries | `data/items/EFF-h38d_*`; `data/rows/EFF-h38d_*_cuda_libs.txt` | recomputed; six bundled libraries per run, none from the system toolkit |
| E51: 100/100 [96.3, 100], accuracy 97 / 97; 1.4.7 vs dev 67/100 [57.3, 75.4] | `data/items/EFF-h38e_*`, `data/rows/EFF-h38e_*_cuda_libs.txt` | recomputed |
| E52: released 1.5.0, 100/100 [96.3, 100], accuracy 98 / 98, 0 discordant; dev (E51) vs release 72/100 [62.5, 79.9] | `data/items/EFF-h38f_*`, `data/rows/EFF-h38f_*_cuda_libs.txt` | recomputed; six bundled libraries per run |
| Committed per-comparison summaries | `data/rows/h38*.json` | agree with the recomputation |

## Other checked values

E7 J/token for the fixed-3 row; E25 cold-start step 46.6 → 44.2 tok/s; E35
per-position decay 0.77 / 0.66 / 0.53 / 0.37; E36 acceptance 0.592 on 19,074
positions against Σmin = 0.597; E37 bucket shares and chunk rates; the E40 table
(transitions 2.0 vs 4.7, corrections 1 vs 4, low-confidence 2.3 vs 1.4%,
transition-token log-probabilities, reasoning-token estimates, HumanEval+ rows,
first divergence at the third token on 24 of 25 math items); the E41, E42, and
E45 rows of Table A12.

## Figures that rest on logs or campaign text (not asserted by the checker)

- Per-slot recurrent cost 728 / 438 / 148 MiB (E28): the campaign's derivation
  from the allocation shape; the checker verifies only the 3 MiB per layer and
  step and the 2,880 / 576 MiB totals.
- E12 fit checks (stripped sidecar at 8K and 32K, DFlash2 at 6K, old build).
- E38 decode-timer figures (50.1 → 39.5 tok/s, TabbyAPI `Generate:` lines); the
  row file holds the end-to-end figures.
- E48 llama.cpp draft acceptance 0.84–0.85 and mean draft length 3.5 (server
  timing lines).
- E27 host RSS (3,744 vs 3,745 MiB) and E43 host RSS and replay rate.
- E36 distribution tests (first-token z = −0.7, unigram χ² = 53 on 100 dof).
- GPU hours of the hardening cells; the host CPU and RAM (verified on the host);
  the library-path environment facts (`/proc/<pid>/maps` reading); and the
  upstream pull-request and issue states, checked against GitHub on 2026-09-19
  but not re-derivable from the bundle.

## Corrected figures (2026-09-19)

The checker first found seven figures that the rows reproduce only at a
neighbouring rounding or definition. The paper and the campaign document were
corrected to the values below, and the checker now asserts each one exactly.
No verdict changed.

| Figure | Was | Now | Reason |
|---|---|---|---|
| E7 code/math speed-up | 1.64–1.73× | 1.67–1.74× | same-version undrafted baseline (29.2 / 29.0 tok/s) |
| Table 2 "Capped", IQ2_S lane | 99% | 96% | all-samples definition, as for the EXL3 row |
| MTP sidecar size | 1.27 GiB | 1.28 GiB | rounded, not truncated (1.2755 GiB) |
| Table A4 bare API, no draft, 1 slot | 7,256 MiB | 7,260 MiB | after-generate, like the other rows |
| Table A10 HumanEval+ token ratio | 0.97 | 0.98 | ratio of means, like the math row |
| E42 GSM8K token change | +15.5% | +15.6% | per-item means (+15.57%) |
| E45 GSM8K token change at −1.0 | −5.3% | −5.2% | per-item means (−5.249%) |

## Interpretation boundary

The evidence supports the measured configurations on this laptop. It does not
support claims about other models, cards, power limits, or engine versions, a
global optimum, a speed ranking of the engines, the offloaded 4-bit wall-clock
ratio (recorded, not claimed), or 2-bit-versus-4-bit parity beyond GSM8K at
n = 200.
