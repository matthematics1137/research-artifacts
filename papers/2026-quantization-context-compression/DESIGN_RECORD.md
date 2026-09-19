# Design record: P2R1 (Paper 2)

This record gives the dated chronology that the report's corrective-status
statement relies on. The frozen machine-readable protocol is
`replication/paper2/R1_PROTOCOL.json`, and its human-readable companion is
`replication/paper2/RERUN_PLAN.md`. Both are copied byte for byte from the
measurement bundle that every question and run contract hash-binds.

## Why a corrective rerun exists

The first full-grid campaign (P2G) was quarantined during release audit. Its
launcher accepted any healthy HTTP endpoint on the expected port. A stale
server from an earlier session answered requests attributed to several
deployments while the intended servers failed to start. No P2G number is used
anywhere in the report.

P2R1 was designed after the P2G outputs, the hypotheses, and the corpus
frames had been seen. It is therefore labelled a **corrective exploratory
rerun, not an independent confirmation**, in the protocol, the report, and
this artifact.

## What was frozen before any inference (2026-08-31)

The protocol and the recovery harness were committed together before the
first model load. They fix these elements:

- **Deployments.** The three artifact–engine–placement deployments, their
  pinned artifacts, and the engine revisions.
- **Corpora.** The frame windows (160 WildChat and 126 private) and the exact
  stratified question quotas: 30/30/30/30 WildChat and 37/37/6/37 private.
- **Questions.** The seeded question-generation schedule, with retry seed
  offsets 0, 1000000, and 2000000, plus the verifier and the selection rule.
- **Requests.** The prompts, requested decoding (seed 42, temperature 0.2, top-p 0.95),
  scoring, and the 14,000-character window cap.
- **Analysis.** The marginal paired difference-in-differences estimand, the
  CR2/Satterthwaite source-conversation-cluster-robust inference, the rule
  that every degree of freedom is at least 5, and Holm across 12 tests as the
  sole rule setting the result branch.
- **Stage order and gates.** The fixed stage order, the cool-state gates, and
  the append-only journal rules.

**Post-run seed disclosure, 2026-09-20.** The pinned TabbyAPI revision
`4a4f9f4` did not accept a per-request seed, so the EXL3 evaluation seed was
recorded but not applied. The llama.cpp endpoints accept the seed field;
Q4 question generation and seeded source selection/item ordering are separate.
The frozen protocol and harness describe what was requested and are retained
unchanged. Exact analysis reproduction uses the saved outputs; EXL3 responses
cannot be regenerated from the recorded seed alone. No response or score was
changed by this disclosure.

Three pre-inference amendments followed on 2026-08-31, each before any
request:

- **Loader verbosity.** Exact llama.cpp verbosity `-lv 4`, so that the
  loader's placement readback (`33/66`, `65/65`) is captured and required.
- **CUDA runtime mappings.** A classifier correction for deleted anonymous
  mappings in the llama.cpp CUDA-runtime check.
- **Desktop admission.** A graphics-aware desktop admission gate that records
  utilization but does not cap it.

## Attempt chronology

| Attempt | Date | How it stopped | Evaluation requests | Amendment registering the next attempt |
|---|---|---|---|---|
| v1 | 2026-09-01 | Question generation. The frozen 512-token completion cap made one stratum's quota infeasible. | none | Completion cap 512 → 2048 for generation and evaluation |
| v2 | 2026-09-01 | Inside the unmeasured warmup. Its 64-token cap was exhausted before the answer token. | none | Warmup cap 64 → 2048 |
| v3 | 2026-09-11 | An operator pause tool stopped the unit containing the campaign mid-request. This left one request intent without a terminal. | Q4_K_XL rows only | Operator tooling fixed; desktop-admission thresholds relaxed (VRAM ≤ 1536 MiB, load ≤ 8) |
| v4 | 2026-09-13 | At the first real run of the EXL3 placement check. It rejected the driver's anonymous `/dev/zero (deleted)` mappings, a Triton runtime helper, and an unmeasured 8,000 MiB allocation bound. | Q4_K_XL rows only | EXL3 placement check repaired and rehearsed; the bound set to 7,000 MiB against a measured 7,666–7,706 MiB |
| **v5** | 2026-09-13 to 2026-09-16 | **Completed and validated** | all 2,844 | — |

These rules applied to every restart:

- Each restart was a committed protocol amendment made before the next
  launch.
- The questions were regenerated under the same seeds, and all six grids were
  redone.
- No amendment changed the inputs, question quotas, prompts, decoding,
  scoring, or statistical analysis.
- Attempts v1 and v2 stopped before any evaluation request. The v3 and v4
  higher-footprint rows are retained only as forensic evidence and enter no
  analysis. They reproduce the v5 responses byte for byte: 948 of 948 for v4,
  and 871 of 872 for v3, where the one difference is an empty v3 response.
- Before v5 was registered, every stage that no attempt had reached was
  rehearsed with the launcher's own functions against a scratch directory.
  Those stages were the EXL3 and IQ2_S launches and the post-campaign
  verifications.

During v5 there was one clean operator pause, in question generation, made
exactly at a journal boundary. The harness resumed from the next window,
repeating no request.

## Post-run analysis record

The campaign's own validator promoted the evidence on 2026-09-16. The first
real run of the post-promotion pipeline then exposed three defects, all in
analysis files:

- a release-gate false positive on a scalar hash key;
- a case-sensitive test assertion;
- a figure legend overlapping data.

The fixes touch no measurement file. The promotion record binds the
analysis-file hashes, so the validator was run again after a rehearsal in
which its only write was redirected. The rehearsed record differed from the
original only in those analysis hashes. The superseded record is retained
privately.

## Planned but not claimed

No equivalence margin, smallest effect of interest, or power target was
pre-specified. Any non-detection is therefore reported as bounded by its
pointwise intervals, never as equivalence or independence.
