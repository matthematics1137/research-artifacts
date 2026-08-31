---
pretty_name: "GSM8K Accuracy of Low-Bit Mixtral and Qwen3 GGUFs"
license: other
license_name: "Mixed CC-BY-4.0, MIT, and third-party terms"
license_link: "https://huggingface.co/datasets/mv1137/p2026-004-mixtral-qwen3-gsm8k-results/blob/main/LICENSE"
task_categories:
  - question-answering
language:
  - en
tags:
  - reproducibility
  - llm-inference
  - mixture-of-experts
  - quantization
  - gguf
  - gsm8k
configs:
  - config_name: "primary-gsm8k"
    data_files:
      - split: train
        path: "data/primary_gsm8k.jsonl"
---

# GSM8K Accuracy of Low-Bit Mixtral and Qwen3 GGUFs

This result dataset mirrors the version-1.0.0 reproducibility artifact:
[10.5281/zenodo.22346482](https://doi.org/10.5281/zenodo.22346482).
The [versioned report and full replication sources](https://github.com/matthematics1137/research-artifacts/tree/moe-quantization-granularity-v1.0.0/papers/2026-moe-quantization-granularity)
are maintained together in the research-artifacts repository. Cite the exact
Zenodo version for the frozen evidence; this HF copy is a discovery mirror.

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This dataset is the privacy-sanitized evidence package for
“Contrasting GSM8K Accuracy Across Low-Bit Mixtral and Qwen3 GGUFs.” It asks
how accuracy changes when a local user chooses a smaller IQ2 or IQ1 GGUF
instead of the same model's Q4 file.

## Headline result

All six deployments answered the same 100 GSM8K items.

| Model | Q4 | IQ2 | IQ1 | IQ2/IQ1 smaller than Q4 |
|---|---:|---:|---:|---:|
| Mixtral-8x7B | 68% | 46% | 26% | 56% / 62% |
| Qwen3-30B-A3B | 91% | 97% | 96% | 44% / 48% |

Mixtral's Q4-minus-low differences were +22 and +42 percentage points and
both remained significant after Holm correction. Qwen's differences were -6
and -5 points; neither comparison was rejected after Holm correction (both
adjusted p=0.0625). The Qwen results do not establish improvement or
equivalence.

The contribution is a paired, provenance-bound measurement for six exact
artifact-engine deployments. It is not a new quantizer or optimization, a
general ordering of GGUF tiers, or a causal result about expert granularity.

## Files

| Path | Description | Rows |
|---|---|---:|
| `data/primary_gsm8k.jsonl` | Sanitized item ID, correctness, truncation, timing, token, seed, and provenance fields | 600 |
| `data/claims.json` | Frozen P4R1 cell, paired-statistics, artifact, protocol, and provenance record | one JSON object |
| `data/EXPORT_MANIFEST.json` | Hash binding from the local export to the promoted P4R1 evidence | one JSON object |
| `environment/` | Exact model, host, engine, protocol, and observed placement metadata, including hash-bound Qwen startup excerpts | four JSON files |
| `CLAIM_EVIDENCE_MAP.md` | Human-readable claim-to-field map | one document |
| `check_claims.py` | Offline, standard-library verification of this dataset copy | one Python program |

Response prose, prompts, benchmark questions and answers, model weights,
private paths, and secrets are excluded.

The dataset viewer loads only the 600 homogeneous item-level rows in the
`primary-gsm8k` configuration. The claims and export-manifest JSON files remain
downloadable supporting records; they are not additional dataset rows.

## Verify

From the root of a downloaded copy of this dataset repository:

```bash
python3 check_claims.py
```

The checker uses only Python's standard library. It verifies the two exported
data-file hashes, recomputes all six scores and four paired comparisons, and
checks the result against the frozen P4R1 claims and environment records.

## Protocol and provenance

- P4R1 campaign `campaign-003`; corrective and exploratory, not independent.
- llama.cpp commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`.
- Context 4,096, one request at a time, fixed generation/scoring, and the same
  deterministic per-item seed across all six files.
- Mixtral GPU-offloaded-layer counters: Q4 12/33, IQ2 26/33, IQ1 29/33.
- Qwen counters: 49/49 for all tiers, including partial offload in Q4 (25
  overflowing layers) and IQ2 (two); IQ1 required no fit adjustment. Matching
  counters do not establish matched tensor placement or complete GPU weight
  residency. `environment/placement_readback_excerpt.json` supplies exact
  startup lines, source hashes, and separate CPU-mapped/CUDA model-buffer sizes.
  These sizes are not additive measurements of physical resident memory.
- Exact upstream repositories, revisions, basenames, byte counts, and SHA-256
  values are in `environment/model_artifacts.json`.

The historical campaign is excluded from the data files because it did not
meet P4R1's endpoint/model/placement provenance standard.

## Limitations

- One machine, two model families, one 100-item task, and one stochastic
  completion per artifact/item.
- Q4 is a within-model deployment anchor, not an unquantized ceiling.
- The two ladders are not precision-matched: Qwen IQ1 uses 2.54 whole-file
  bits per parameter, above Mixtral IQ2 at 2.15. This is not a controlled
  cross-model test of architectural robustness.
- Non-rejection is not equivalence, and expert count is not a manipulated
  variable.

## Rights

Measurement fields and original documentation are CC BY 4.0; authored code is
MIT. Upstream material retains its own terms. See `LICENSE`, `LICENSES`, and
`THIRD_PARTY_NOTICES.md` in this dataset.

## AI assistance

Claude (Anthropic) assisted with evaluation-harness development and experiment
orchestration. Claude and Codex (OpenAI) assisted with evidence auditing,
analysis and figure code, citation checking, and manuscript drafting and
revision. Matthew Schwartz directed the research and is responsible for its
methods, results, interpretation, claims, citations, rights, and released
artifacts.
