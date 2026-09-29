# llama.cpp Throughput for Qwen3.8-Flash-Next and gpt-oss-120b on a 62 GiB RAM Laptop — reproducibility artifact

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This is the curated evidence bundle for:

> “llama.cpp Throughput for Qwen3.8-Flash-Next and gpt-oss-120b on a
> 62 GiB RAM Laptop.” 2026.

The benchmark asks what decode throughput unmodified llama.cpp's ordinary mmap
path delivered for two exact GGUF deployments on one 62 GiB RAM laptop. With
the tested artifacts, engine checkout, and placement, a 67.564 GiB
Qwen3.8-Flash-Next artifact produced 9.4–14.7 decode tokens/s across six
observations, each requesting 256 output tokens. A 59.034 GiB gpt-oss-120b
artifact produced 8.7–9.3 decode tokens/s across three such observations.
These decode rates exclude loading and prompt processing and do not establish
interactive response latency. Recorded prompt-processing rates were slower
than decode. These are deployment-specific throughput measurements. The larger file's
successful run is supporting context, not a new mmap-feasibility result, and
the report introduces no inference optimization or causal model of page-cache
behavior or decode I/O.

## Release identity

- Version: `1.0.0`
- GitHub tag: `moe-nvme-streaming-v1.0.0`
- Tagged artifact path: <https://github.com/matthematics1137/research-artifacts/tree/moe-nvme-streaming-v1.0.0/papers/2026-moe-nvme-streaming>
- Hugging Face result dataset: <https://huggingface.co/datasets/mv1137/p2026-003-qwen38-flash-gpt-oss-throughput-results>
- Zenodo version DOI: [doi:10.5281/zenodo.22314465](https://doi.org/10.5281/zenodo.22314465)
- Release date: `2026-09-28`

The version DOI identifies the frozen reproducibility bundle; it is not a
separate manuscript DOI. A prepublication copy can contain a reserved DOI:
public availability is established by the released Zenodo record and matching
GitHub tag, not by the presence of a DOI in this file.

## What the bundle contains

- `paper/`: canonical light PDF and sanitized TeX source.
- `data/timing_measurements.jsonl`: the nine principal generation rows plus
  retained placement and prefill measurements.
- `data/quality_outcomes.jsonl`: item IDs, scored outcomes, timings, token
  metadata, and short MATH answer fields needed for the disclosed rescore.
  Prompts and model response prose are omitted.
- `data/claims.json`: deterministic audit record for the protocol, tensor
  inventory, statistics, evidence fingerprints, and measurement boundaries.
- `evaluation/`: the frozen benchmark inputs and standard-library harness used
  for a full rerun; upstream rights are mapped in `THIRD_PARTY_NOTICES.md`.
- `environment/`: exact machine, engine association, and model-file identities.
- `check_claims.py`: offline verification of evidence hashes, generation rates,
  file inventory values, row-backed task scores, and the lenient MATH rescore.

## Five-minute analysis-only check

Python 3.11 or newer is sufficient. No model, GPU, network access, or package
installation is required.

```bash
python3 check_claims.py
python3 evaluation/run_eval.py --selftest
```

The first command must end with `ALL PAPER 3 CLAIM CHECKS PASSED`; the second
must end with `SELFTEST PASSED (all checks)`. Both normally finish in seconds.
They fail closed if the sanitized evidence or scorer contract changes.

## Full inference replication

See `FULL_RERUN.md`. It pins the associated llama.cpp revision, four GGUF
files, build settings, prompt, placement, quiet gate, task inputs, and scoring
commands. The weights total 126.60 GiB, so allow at least 150 GiB for downloads
and more for build products and new results. Exact historical timings and item
outcomes are not expected because the runs were sequential, sampling was
stochastic, and no inference seed was fixed.

## Result boundary

The `/proc/diskstats` values cover the physical NVMe device over each complete
`llama-cli` invocation. They include loading, prefill, decode, teardown, and
unrelated reads by other processes. They are not decode-only model bytes,
expert-cache misses, or bytes per generated token. Initial and later run groups
also retained different memory fields and were not randomized.

Small task samples confirm that both deployed stacks returned correct answers.
They do not show that mmap preserved model quality because no resident
same-model control was run. The Flash HumanEval+ aggregate is omitted from the
paper because three underlying item rows are missing.

## Rights, citation, and security

Authored code is MIT-licensed. The paper, original documentation, figures, and
measurement records are offered under CC BY 4.0 to the extent the author holds
the relevant rights. MATH and GSM8K are MIT; HumanEval+ is Apache-2.0 and also
retains HumanEval's MIT terms. The Qwen artifact is subject to Qwen Community
License 1.0; gpt-oss and llama.cpp are Apache-2.0 and MIT respectively. Model
weights and engine source are linked, not redistributed.

Use `CITATION.cff` for the report and versioned artifact. Treat generated code
as untrusted: execute it only in an isolated container or VM with no secrets
and no network access.

## AI assistance

Claude (Anthropic) assisted with evaluation-harness development and experiment
orchestration. Claude and Codex (OpenAI) assisted with evidence auditing,
analysis and figure code, citation checking, and manuscript drafting and
revision. Matthew Schwartz directed the research and is responsible for its
methods, results, interpretation, claims, citations, rights, and released
artifacts.
