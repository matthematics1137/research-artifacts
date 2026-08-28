# Qwen3.8-27B on a 12 GB laptop GPU — reproducibility artifact

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This is the curated evidence bundle for:

> “Deploying Qwen3.8-27B in 12 GB of VRAM: Accuracy and
> Throughput Across Quantized Inference Stacks.” Technical report, 2026.

The paper characterizes complete artifact–engine deployments on one 80 W RTX
4080 Laptop GPU. It does not introduce a quantizer, claim a universal optimum,
or establish parity with bf16/Q8.

## Release identifiers

- Version: 1.0.0, dated 2026-08-29
- Version-of-record artifact: [doi:10.5281/zenodo.22166977](https://doi.org/10.5281/zenodo.22166977)
- GitHub release: <https://github.com/matthematics1137/research-artifacts/releases/tag/qwen38-27b-12gb-v1.0.0>
- Tagged source and results: <https://github.com/matthematics1137/research-artifacts/tree/qwen38-27b-12gb-v1.0.0/papers/2026-qwen38-27b-12gb>
- Hugging Face result dataset: <https://huggingface.co/datasets/mv1137/qwen38-27b-12gb-results>

The DOI is assigned to the reviewed Zenodo deposition. The immutable DOI and
GitHub tag identify version 1.0.0.

## Contents

- `paper/main.pdf`: canonical light paper.
- `paper/main.tex`, `paper/references.bib`, and `paper/figures/`: sanitized
  publication source; the private explainer/defense text is excluded.
- `testsuite/results/phase1.jsonl`: sanitized fit and throughput records.
- `testsuite/evals/results/summary.jsonl`: Qwen3.8-only aggregate rows.
- `testsuite/evals/results/<configuration>/`: retained per-item scoring fields
  and final-answer excerpts.
- `testsuite/evals/rescore_math25.py`: disclosed alternate MATH normalizer.
- `paper/scripts/make_figures.py`: figure/statistics generator.
- `check_claims.py`: fail-closed verification of the paper's key reported
  values and statistical tests.
- `environment/`: hardware, software, and exact model-artifact provenance.

Full response transcripts were not retained. Per-item records contain scoring
fields and, where applicable, only the final 200 characters of a response. The
Q4 easy-suite GSM8K and HumanEval runs survive only as aggregate summary rows;
their per-item files were not retained. These gaps are stated in the paper and
must not be reconstructed after the fact.

## Five-minute analysis-only check

No model, GPU, network access, or inference server is required.

```bash
python3 check_claims.py
python3 -m pip install -r requirements-analysis.txt
python3 paper/scripts/make_figures.py
```

`check_claims.py` must finish with `ALL CLAIM CHECKS PASSED`. The figure command
recomputes effective bpw, alternate MATH scores, Wilson intervals, matched-workload
rates, and the four paper figures from the frozen files. It writes only inside
the checked-out artifact.

To rebuild the paper after regenerating the figures:

```bash
make -B -C paper
```

The released `paper/Makefile` fixes `SOURCE_DATE_EPOCH`, forces source-date PDF
metadata, uses the cached Tectonic bundle in untrusted mode, and compiles the
already sanitized public `main.tex`. Its `main.pdf` must match the version in
the release checksum inventory byte for byte.

## Full inference replication

See `FULL_RERUN.md`. Reproduction requires a CUDA-capable system, the listed
engine revisions, and approximately 100 GiB for all tested checkpoints. The
exact 12 GB residency boundary depends on background VRAM, power state, driver,
engine buffers, and context settings. A full campaign is an overnight-class
run, not part of the analysis-only check.

The weight files are not redistributed. `environment/model_artifacts.json`
records each upstream repository, immutable revision, filename, bytes, SHA-256,
and license.

## Security

HumanEval and HumanEval+ execute model-generated Python. The original harness
used a restricted environment and timeout, but it is not a hardened jail. Run
untrusted generated code inside an isolated container or VM with no secrets and
no network access.

## Rights and citation

Authored code is MIT-licensed. The paper, original documentation, figures, and
released measurement records are offered under CC BY 4.0 to the extent the
author holds rights. Benchmark subsets and derived fields retain their upstream
terms; see `THIRD_PARTY_NOTICES.md` and `LICENSE`.

Use `CITATION.cff` for citation metadata. Cite the
[immutable tagged report PDF](https://github.com/matthematics1137/research-artifacts/blob/qwen38-27b-12gb-v1.0.0/papers/2026-qwen38-27b-12gb/paper/main.pdf)
for scientific claims and
[doi:10.5281/zenodo.22166977](https://doi.org/10.5281/zenodo.22166977) for the
frozen evidence bundle.

## AI assistance

Claude (Anthropic) and Codex (OpenAI) assisted with evaluation-harness
development, experiment execution, evidence auditing, analysis and figure code,
and manuscript drafting and revision. Matthew Schwartz directed the work and
reviewed and verified the retained outputs against the archived evidence and
cited sources, edited the manuscript, and takes full responsibility for the
methods, results, interpretation, citations, and released artifacts.
