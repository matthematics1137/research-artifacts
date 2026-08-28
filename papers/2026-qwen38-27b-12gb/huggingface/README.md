---
pretty_name: "Qwen3.8-27B on 12 GB — frozen deployment measurements"
license: other
tags:
  - reproducibility
  - llm-inference
  - quantization
  - consumer-gpu
---

# Qwen3.8-27B on 12 GB — result dataset

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This dataset mirrors the redistribution-cleared measurements for “Deploying
Qwen3.8-27B in 12 GB of VRAM: Accuracy and Throughput Across Quantized
Inference Stacks.”

The immutable version-of-record artifact is
[doi:10.5281/zenodo.22166977](https://doi.org/10.5281/zenodo.22166977). The
matching tagged GitHub source and analysis instructions are at
<https://github.com/matthematics1137/research-artifacts/tree/qwen38-27b-12gb-v1.0.0/papers/2026-qwen38-27b-12gb>.
The matching GitHub release is
<https://github.com/matthematics1137/research-artifacts/releases/tag/qwen38-27b-12gb-v1.0.0>.
This discovery dataset is
<https://huggingface.co/datasets/mv1137/qwen38-27b-12gb-results>. The DOI is
assigned to the reviewed Zenodo deposition. These links identify version 1.0.0,
dated 2026-08-29.

This repository contains measurement records under mixed terms rather than a
single blanket license. Original measurements and documentation are CC BY 4.0;
benchmark-derived fields retain their upstream terms. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

The study measured eight complete artifact–engine deployments on one 80 W RTX
4080 Laptop GPU. It is a single-machine characterization, not a universal model
or quantization ranking. Full model outputs were not retained; the per-item
files contain scoring fields and short answer excerpts. No model weights are
included. Exact upstream weight revisions and hashes are in
`environment/model_artifacts.json`.

The full analysis-only verifier and fixed benchmark subsets live in the
canonical GitHub/Zenodo artifact rather than this discovery mirror. From that
artifact root, run:

```bash
python3 check_claims.py
```

## AI assistance

Claude (Anthropic) and Codex (OpenAI) assisted with evaluation-harness
development, experiment execution, evidence auditing, analysis and figure code,
and manuscript drafting and revision. Matthew Schwartz directed the work and
reviewed and verified the retained outputs against the archived evidence and
cited sources, edited the manuscript, and takes full responsibility for the
methods, results, interpretation, citations, and released artifacts.
