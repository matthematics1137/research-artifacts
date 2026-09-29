# Contrasting GSM8K Accuracy Across Low-Bit Mixtral and Qwen3 GGUFs

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This artifact accompanies the version-1.0.0 technical report and preserves
the checked P4R1 measurements, statistical analysis, and replication sources.

- Exact artifact version: [10.5281/zenodo.22346482](https://doi.org/10.5281/zenodo.22346482).
- Versioned source and report: [GitHub tag `moe-quantization-granularity-v1.0.0`](https://github.com/matthematics1137/research-artifacts/tree/moe-quantization-granularity-v1.0.0/papers/2026-moe-quantization-granularity).
- The artifact's version-1.0.0 citation date is 2026-09-28. The DOI identifies
  the frozen mixed-license compendium; no separate manuscript DOI is minted.

## Question and result

For two public MoE model families, how much does GSM8K accuracy change when a
local deployment uses a substantially smaller IQ2 or IQ1 GGUF instead of its
within-model Q4 file?

On the same 100 GSM8K items, the tested Q4/IQ2/IQ1 files scored:

| Model | Q4 | IQ2 | IQ1 | IQ2/IQ1 file-size reduction from Q4 |
|---|---:|---:|---:|---:|
| Mixtral-8x7B | 68 | 46 | 26 | 56% / 62% |
| Qwen3-30B-A3B | 91 | 97 | 96 | 44% / 48% |

For Mixtral, the paired Q4-minus-IQ2 and Q4-minus-IQ1 differences were +22 and
+42 percentage points and both survived Holm correction. For Qwen, the signed
differences were -6 and -5 points; neither comparison was rejected after Holm
correction (both adjusted p=0.0625). Those Qwen non-rejections establish
neither improvement nor equivalence.

The lower tiers are not precision-matched across models: Qwen's IQ1 file uses
2.54 whole-file bits per parameter, above even Mixtral's IQ2 at 2.15. The
contrast therefore does not isolate architectural robustness to quantization.

This is an artifact-specific paired benchmark. It introduces no quantizer,
optimizer, compression format, runtime kernel, expert-count scaling law, or
general benchmark-priority claim.

## Evidence status

P4R1 campaign `campaign-003` replaced the historical six-cell campaign, whose
saved rows could not be bound to the intended endpoint, model identity, and
placement under the later provenance standard. The historical scores are
forensic context only and are not release evidence.

P4R1 is corrective and exploratory, not independent: its item IDs, historical
outcomes, and motivating hypothesis were already known. All six cells used one
pinned llama.cpp build, context 4,096, the same 100 item IDs, a shared
deterministic per-item seed, fixed generation and scoring, and recorded
offloaded-layer counters and model-buffer readbacks. The layer counter can
include partial offload; it is not a complete weight-residency measurement.

## Artifact contents

- `data/primary_gsm8k.jsonl`: 600 privacy-sanitized item-level outcome rows;
- `data/claims.json`: the frozen P4R1 statistical and provenance record;
- `data/EXPORT_MANIFEST.json`: hashes binding those files to the promoted P4R1
  evidence and checker;
- `environment/`: exact artifact, host, engine, protocol, and placement
  metadata, including hash-bound, line-numbered Qwen startup excerpts;
- `check_claims.py`: standard-library recomputation of scores, paired tables,
  intervals, exact McNemar tests, and Holm adjustment;
- `P4R1_PROTOCOL.md` and `FULL_RERUN.md`: the frozen protocol and documented
  full-rerun path;
- `REPLICATION_SOURCE.md` and the nested source-layout snapshot: the exact
  frozen harness, promoter, checker, tests, and plan-bound analysis sources;
- `paper/main.pdf` and `paper/main.tex`: the technical report and its sanitized,
  reproducible source.

Response prose, benchmark questions and answers, model weights, private paths,
and secrets are excluded.

## Five-minute analysis quickstart

```bash
python3 check_claims.py
python3 publication/papers/moe-quantization-granularity/public/replication/run_gsm8k.py --selftest
```

Both commands use Python's standard library and run without weights, network
access, or a GPU. The checker prints `ALL PAPER 4 P4R1 PUBLIC CLAIM CHECKS
PASSED` and the six scores above. It checks exact frozen hashes; a changed
data file is an error, not an input for silently updating this report.

To rebuild the paper from the shipped figures, install Tectonic 0.17.0 and
populate its standard bundle cache once, then run:

```bash
make -B -C paper
```

Dependency installation and initial TeX-bundle retrieval may require network
access. The Makefile then uses the pinned timestamp and cached, untrusted
build mode. `FULL_RERUN.md` and `REPLICATION_SOURCE.md` describe the much larger
inference path, including its source-layout snapshot, hardware requirements,
84.23 GiB of exact model files, and separate prospective experiment approval.

`data/EXPORT_MANIFEST.json` retains the status recorded when the sanitized
projection was created locally. That frozen provenance field is not a live
publication-status indicator; version identifiers and release receipts name
the published object.

## Scope limits

- Two public model families, six exact GGUF files, one 100-item task, one
  machine, and one stochastic completion per file/item.
- Q4 is an operational within-model anchor, not an unquantized ceiling.
- The ladders are not precision-matched, and file bpw does not isolate routed
  expert precision.
- Expert count is confounded with model family, training, architecture,
  quantizer, placement, and other differences; no causal mechanism is tested.
- Mixtral's GPU-offloaded-layer counters were 12/33, 26/33, and 29/33 for Q4,
  IQ2, and IQ1. Qwen reported 49/49 for all three files, but Q4 had 25
  partially overflowing layers and IQ2 had two; IQ1 required no fit adjustment.
  These counters do not establish complete GPU weight residency or matched
  tensor placement. `environment/placement_readback_excerpt.json` preserves
  the source-bound readbacks and separate CPU-mapped/CUDA model-buffer sizes;
  those sizes must not be summed as disjoint physical resident memory.

## Replication-source integrity

The package includes the
complete replication source through an explicit file allowlist. Security
scanners and their tests
necessarily spell a few values they are designed to reject; each such source
exception is limited to an exact public path, exact SHA-256, and exact literal,
while structured-secret detection and every other scanner rule remain active.
`REPLICATION_SOURCE.md` explains how the prospective Git/source-layout guard is
reconstructed without carrying any prior approval into a new run.

## Rights and AI assistance

Authored code is MIT-licensed. Original documentation and measurement records
are offered under CC BY 4.0 to the extent the author holds rights. Upstream
benchmark, model, and engine terms remain controlling.

Claude (Anthropic) assisted with evaluation-harness development and experiment
orchestration. Claude and Codex (OpenAI) assisted with evidence auditing,
analysis and figure code, citation checking, and manuscript drafting and
revision. Matthew Schwartz directed the research and is responsible for its
methods, results, interpretation, claims, citations, rights, and released
artifacts.
