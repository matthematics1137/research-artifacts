# Public design and corrective-validation record

This chronology separates the original exploratory records, the later evidence
audit, and the P4R1 replacement campaign. The historical campaign is forensic
context only. Current scientific claims use P4R1 exclusively.

## 2026-08-27 and 2026-08-29: historical prospective records

Before the historical Mixtral arm ran, the working record proposed expert
granularity as a possible moderator of low-bit robustness. A later Qwen
addendum selected a second public three-tier GGUF ladder. Sanitized snapshots
are retained under `design/` with their original Git commits and timestamps.

Those records are useful provenance, but they do not make the current result
confirmatory. The item IDs, historical outcomes, and motivating hypothesis
were known before P4R1.

## Historical audit

The initial analysis had both statistical and provenance problems:

- same-item tier comparisons required paired inference rather than the planned
  independent-sample tests;
- an excluded Flash artifact's whole-file precision differed from its tier
  label and it lacked a same-model Q4 anchor;
- historical launchers could accept a pre-existing server and shared log paths
  were overwritten;
- five of six cells lacked enough retained evidence to bind saved responses to
  the intended endpoint, model, and placement under the later standard.

The historical scores therefore remain forensic only and are not exported as
release data.

## 2026-08-31: P4R1 replacement campaign

P4R1 prospectively froze a corrective six-cell protocol before replacement
inference. It used the same 100 GSM8K item IDs for all files, context 4,096,
one request at a time, a shared deterministic per-item inference seed, fixed
generation/scoring, separate immutable cell directories, exact process and
endpoint identity checks, placement readback, shutdown receipts, and full
post-campaign artifact verification.

The retained analysis is:

- Mixtral IQ1_M, IQ2_XXS, and Q4_K_M on the same 100 items;
- Qwen UD-IQ1_M, UD-IQ2_XXS, and Q4_K_M on those items;
- one stochastic completion per artifact and item;
- paired Q4-to-low differences within each model;
- exact McNemar tests, Newcombe paired intervals, and Holm adjustment across
  exactly four comparisons.

MATH and HumanEval+ are outside P4R1.

## P4R1 result boundary

Mixtral scored 68/46/26 at Q4/IQ2/IQ1. Its Q4-minus-low differences were +22
and +42 percentage points and both survived Holm correction. Qwen scored
91/97/96; its signed differences were -6 and -5 points, and neither comparison
was rejected after Holm correction. The Qwen results establish neither
improvement nor equivalence.

These observations answer an artifact-specific deployment question. They do
not establish a universal tier ordering, introduce a quantization optimization,
or identify expert granularity as the cause of the cross-model contrast.

## Current release state

The P4R1 evidence and privacy-sanitized data export are locally verified. A
private controller exercise proved reproducible unmarked rendering from a
physically sanitized source projection, and the planned package includes the
complete plan-bound replication-source layout. Canonical rendering remains
blocked until author narrative review. The paper and package remain
unpublished pending that review, rendered-PDF inspection,
rights/bibliography/disclosure review, exact package approval, and separate
authorization for remote publication.
