# Claim–evidence map: hardware-specific llama.cpp mmap benchmark

Run every public check from the artifact root:

```bash
python3 check_claims.py
python3 evaluation/run_eval.py --selftest
```

`data/claims.json` is the frozen audit record generated in the private research
tree. `data/timing_measurements.jsonl` and `data/quality_outcomes.jsonl` are
sanitized, row-level public evidence. Their hashes are hard-coded in
`check_claims.py`; changed or missing input fails before any value is reported.

## Deployment and timing claims

| Paper claim | Public evidence | Deterministic check |
|---|---|---|
| Flash artifact is 72,546,461,344 bytes (67.564 GiB), 3.280 whole-file bpw | `data/claims.json`; exact upstream files in `environment/model_artifacts.json` | byte count and bpw assertions in `check_claims.py`; private audit parsed frozen GGUF headers |
| gpt-oss artifact is 63,387,346,208 bytes (59.034 GiB), 4.341 whole-file bpw | same | same |
| Routed-expert storage is 37.109 GiB for Flash and 56.879 GiB for gpt-oss | `data/claims.json` tensor inventory | fixed audit-record assertions; private audit grouped GGUF tensor records by standardized `*_exps` names |
| Flash initial decode rates are 9.4, 10.4, 10.0 tok/s | `data/timing_measurements.jsonl`, group `flash_initial_loaded_label` | exact row comparison and mean recomputation |
| Flash later quiet-gated rates are 14.7, 14.4, 13.9 tok/s | same, group `flash_quiet_gate` | same |
| gpt-oss rates are 9.3, 8.7, 9.2 tok/s | same, group `gptoss_initial_loaded_label` | same |
| Every principal timing invocation requested 256 output tokens | the same nine rows | assertion over every principal row |
| Device values are whole-invocation physical-device deltas, not decode traffic | `data/claims.json` → `measurement_contract`; each timing row's `device_read_interval` | contract field and exact evidence record; this is a scope statement, not a derived estimate |

The public checker validates the frozen tensor inventory but cannot reconstruct
it without the 126.60 GiB upstream weights. `FULL_RERUN.md` pins those files for
a complete independent header audit.

## Row-backed task claims

| Artifact/task | Original score | 95% Wilson interval | Public rows |
|---|---:|---:|---:|
| Flash MATH | 18/25 (72%) | 52.4–85.7% | 25 |
| Flash MATH, post-hoc lenient | 23/25 (92%) | 75.0–97.8% | same 25 |
| Flash GSM8K | 48/50 (96%) | 86.5–98.9% | 50 |
| gpt-oss MATH | 12/25 (48%) | 30.0–66.5% | 25 |
| gpt-oss MATH, post-hoc lenient | 20/25 (80%) | 60.9–91.1% | same 25 |
| gpt-oss HumanEval+ | 18/20 (90%) | 69.9–97.2% | 20 |
| gpt-oss GSM8K | 46/50 (92%) | 81.2–96.8% | 50 |

`check_claims.py` counts the original row-level `correct` fields. For MATH it
also reruns the disclosed formatting/numeric-equivalence rule on the retained
short expected and predicted answer fields. Prompt text and response prose are
not needed for this analysis-only rescore and are absent from the sanitized
outcome file.

Flash HumanEval+ is deliberately not a paper result. The saved evidence has 17
rows and 16 correct responses, while the aggregate ledger says 17/20. The
three missing rows cannot be audited. The public checker preserves the 16/17
saved-row fact, and the manuscript omits the aggregate percentage.

## Protocol and provenance claims

| Claim | Frozen record |
|---|---|
| 62 GiB RAM, RTX 4080 Laptop GPU, 12,282 MiB VRAM, 80 W, kernel/driver/runtime inventory | `environment/hardware.json`; source fingerprint in `data/claims.json` |
| llama.cpp build `b1-035e227` associated with source commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3` | `environment/software.json` and manuscript bibliography |
| Default mmap, `-ngl 99 --n-cpu-moe 999`, context 4096, request 256 | `data/claims.json` → `protocol.generation`; `FULL_RERUN.md` |
| Quality context 8192, temperature 1.0, top-p 0.95, top-k 20, concurrency 1, seed-42 item inventory | `data/claims.json` → `protocol.quality`; frozen files in `evaluation/datasets/` |
| Exact four model files, revisions, sizes, hashes, and licenses | `environment/model_artifacts.json` |

## Interpretation boundary

The following are expressly **not** supported by the evidence: page-cache hit
rate, decode-only bytes per token, an NVMe bandwidth law, causal benefit from a
quiet host, mmap quality preservation, or a universal speed ranking. Those
claims would require new controlled measurements; no public checker can create
them from the original observational logs.
