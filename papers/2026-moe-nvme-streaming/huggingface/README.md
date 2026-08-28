---
pretty_name: "llama.cpp Throughput for Qwen3.8-Flash-Next and gpt-oss-120b on a 62 GiB RAM Laptop — frozen measurements"
license: other
license_name: "Mixed CC-BY-4.0, MIT, and third-party terms"
license_link: "https://huggingface.co/datasets/mv1137/p2026-003-qwen38-flash-gpt-oss-throughput-results/blob/main/LICENSE"
task_categories:
  - text-generation
language:
  - en
tags:
  - reproducibility
  - llm-inference
  - mixture-of-experts
  - llama-cpp
  - memory-mapping
  - consumer-hardware
configs:
  - config_name: "quality-outcomes"
    data_files:
      - split: train
        path: "data/quality_outcomes.jsonl"
  - config_name: "timing-measurements"
    data_files:
      - split: train
        path: "data/timing_measurements.jsonl"
---

# llama.cpp Throughput for Qwen3.8-Flash-Next and gpt-oss-120b on a 62 GiB RAM Laptop — result index

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This dataset accompanies “llama.cpp Throughput for Qwen3.8-Flash-Next and
gpt-oss-120b on a 62 GiB RAM Laptop.” The immutable tagged report and full artifact
use the following versioned path:

<https://github.com/matthematics1137/research-artifacts/tree/moe-nvme-streaming-v1.0.0/papers/2026-moe-nvme-streaming>

The version-1.0.0 artifact is identified by
[doi:10.5281/zenodo.22314465](https://doi.org/10.5281/zenodo.22314465), with
release date 2026-09-28. The Hub copy is a discovery mirror; the versioned
Zenodo archive is the artifact of record. A reserved DOI in a prepublication
copy does not itself establish public availability; the released Zenodo record
and matching GitHub tag do.

## Data

| Path | Description | Rights/provenance |
|---|---|---|
| `data/timing_measurements.jsonl` | generation, placement, and prefill measurements retained for the report | original measurement fields, CC BY 4.0 |
| `data/quality_outcomes.jsonl` | item IDs, correctness, timings, token metadata, and short MATH answer fields | original measurements plus limited MIT-licensed benchmark answer fields |
| `data/claims.json` | deterministic protocol, tensor-inventory, statistics, and evidence-fingerprint record | original audit record, CC BY 4.0 |
| `environment/` | machine, engine association, and four exact upstream model-file identities | original documentation; weights are not included |

Prompts, model response prose, private paths, secrets, model weights, and engine
source are excluded from this Hub mirror.

## Question and headline result

The narrow question is what decode throughput unmodified llama.cpp's ordinary
mmap path delivered for these exact deployments. On one laptop with 62 GiB RAM
and an RTX 4080 Laptop GPU, a 67.564 GiB Qwen3.8-Flash-Next GGUF produced
9.4–14.7 decode tokens/s across six observations in two sequential host-state
groups. A 59.034 GiB gpt-oss-120b GGUF produced 8.7–9.3 decode tokens/s across
three observations. Each principal observation requested 256 output tokens.
Decode excludes loading and prompt processing; the recorded prompt rates were
slower. These measurements do not establish interactive response latency.
This is a bounded, hardware-specific throughput baseline, not a new mmap
technique or inference optimization, state-of-the-art claim, cache-hit
estimate, decode-bandwidth law, quality-preservation result, or universal
model ranking.

## Verify

From the full GitHub/Zenodo artifact:

```bash
python3 check_claims.py
python3 evaluation/run_eval.py --selftest
```

The first program verifies the sanitized evidence hashes and recomputes the
headline rates and row-backed scores, including the disclosed MATH rescore.
Full inference instructions are in `FULL_RERUN.md` in the tagged artifact.

## Upstream artifacts

- Flash: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`, Qwen Community License 1.0.
- gpt-oss: `ggml-org/gpt-oss-120b-GGUF`, revision
  `238abdd290bb874b90a5da1b4549881b7d05c091`, Apache-2.0.

Every basename, byte count, and Git LFS SHA-256 is in
`environment/model_artifacts.json`. This dataset claims no ownership of the
weights.

## Limitations

- One machine, one NVMe device, one associated engine revision, two artifacts,
  and three sequential observations per principal group.
- Host state was neither randomized nor held fixed; memory fields differ
  across groups.
- Storage counters span the physical device and whole process invocation.
- No resident same-model quality control was run.
- Flash HumanEval+ is omitted because three item rows are missing.

## Rights and citation

Measurement fields and documentation are CC BY 4.0; authored code is MIT.
Benchmark and model terms differ, so the dataset uses Hugging Face's `other`
label; consult the full artifact's `LICENSE` and `THIRD_PARTY_NOTICES.md`.
Cite the report for scientific claims and the version DOI for frozen evidence.

## AI assistance

Claude (Anthropic) assisted with evaluation-harness development and experiment
orchestration. Claude and Codex (OpenAI) assisted with evidence auditing,
analysis and figure code, citation checking, and manuscript drafting and
revision. Matthew Schwartz directed the research and is responsible for its
methods, results, interpretation, claims, citations, rights, and released
artifacts.
