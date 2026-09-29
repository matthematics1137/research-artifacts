# Paper 3 quantitative claims report

This report is generated from `paper3/derived/claims.json`. It is an analysis-only audit of existing evidence; no inference or simulation is run.

## Release-blocking corrections

- Flash routed-expert storage is **37.109 GiB**, not 58 GiB. The 26.822 GiB `per_layer_token_embd.weight` tensor is not a routed expert tensor.
- Flash's mean routed expert-layer slice is **1.546 MiB**; the no-reuse active-expert counterfactual is **185.547 GiB per 256 tokens**, not about 290 GiB.
- gpt-oss routed-expert storage is **56.879 GiB** and its mean expert-layer slice is **12.640 MiB**. Its no-reuse counterfactual is **455.032 GiB per 256 tokens**.
- The mean routed expert slice is **8.17x** larger for gpt-oss, not about 5x.
- Do not report page-cache hit rates, decode-only bytes/token, or effective NVMe bandwidth from these diskstats deltas. The counters bracket the whole CLI invocation and include unrelated reads on the physical device.
- Do not plot the logged cache values on one numeric x-axis: the initial Flash rows use post-run `Cached` and pre-run `MemAvailable`, while quiet Flash and gpt-oss use post-run `MemAvailable`.

## Artifact inventory

| artifact | exact file size | effective file bpw | routed experts | mean routed slice |
|---|---:|---:|---:|---:|
| Flash | 67.564 GiB | 3.280 | 37.109 GiB | 1.546 MiB/expert-layer |
| gpt-oss | 59.034 GiB | 4.341 | 56.879 GiB | 12.640 MiB/expert-layer |

The effective file bpw values divide exact deployed bytes by logical GGUF tensor elements; they are not the nominal quant labels.

## Generation observations

| run group | generation tok/s (individual) | mean | whole-invocation device-read delta (GiB) |
|---|---:|---:|---:|
| flash_initial_loaded_label | 9.4 / 10.4 / 10.0 | 9.93 | 104.2 / 102.1 / 103.6 |
| flash_quiet_gate | 14.7 / 14.4 / 13.9 | 14.33 | 84.3 / 82.1 / 72.0 |
| gptoss_initial_loaded_label | 9.3 / 8.7 / 9.2 | 9.07 | 126.2 / 124.1 / 120.8 |

These are observational sequences, not randomized condition estimates. In particular, the loaded/quiet contrast changes system state as well as the reported memory metric.

## Prefill observations

| model | approximate prompt tokens | observed prompt tok/s | requested decode tokens |
|---|---:|---:|---:|
| flash | 160 / 640 / 2560 | 6.6 / 13.1 / 20.1 | 16 / 16 / 16 |
| gptoss | 160 / 640 / 2560 | 4.2 / 9.1 / 23.5 | 64 / 64 / 64 |

The observed throughput increases are descriptive. With one ascending run per approximate length, different prompts, no fixed seed, and no Flash I/O trace, these data do not establish expert-read de-duplication.

## Quality evidence

| model | MATH raw | MATH lenient | HumanEval+ saved rows | HumanEval+ aggregate | GSM8K |
|---|---:|---:|---:|---:|---:|
| flash | 72.0% (18/25) | 92.0% (23/25) | 94.1% (16/17) | 85.0% (17/20) | 96.0% (48/50) |
| gptoss | 48.0% (12/25) | 80.0% (20/25) | 90.0% (18/20) | 90.0% (18/20) | 92.0% (46/50) |

**Flash HumanEval+ provenance blocker:** The saved file contains 17/20 rows (16 correct), while summary.jsonl reports 17/20 correct. The 85% result is aggregate-only unless the three missing item rows are recovered; the row-backed snapshot is 16/17.

## Defensible claim boundary

The logs support feasibility and descriptive throughput on one laptop: both artifacts completed generation while their files were roughly 59--68 GiB, and the recorded prompt-throughput values rise across each single ascending prompt-scale sweep. They do not currently support causal cache-hit, cache-size, NVMe-bandwidth, expert-read-de-duplication, or asymptotic-performance claims.

The machine-readable file preserves all run-level values, source hashes, interval semantics, protocol facts, Wilson intervals, and GGUF tensor-group arithmetic.

The manuscript must load `paper3/derived/claims.tex` and cite its generated macros rather than retyping quantitative values. `python3 paper3/scripts/build_claims.py --check` rebuilds all derived outputs in memory and fails on any difference.
