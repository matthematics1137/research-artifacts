---
pretty_name: "Deployed Quantization Tier and Lossy Context Compression in Extractive QA"
license: other
license_name: "Mixed CC-BY-4.0, MIT, and third-party terms"
license_link: "https://huggingface.co/datasets/mv1137/p2026-002-quantization-context-compression-results/blob/main/LICENSE"
task_categories:
  - question-answering
language:
  - en
tags:
  - reproducibility
  - llm-inference
  - prompt-compression
  - quantization
  - gguf
  - exl3
  - llmlingua
configs:
  - config_name: "cells"
    data_files:
      - split: train
        path: "data/cells.csv"
  - config_name: "interaction"
    data_files:
      - split: train
        path: "data/interaction.csv"
---

# Deployed Quantization Tier and Lossy Context Compression in Extractive QA

This result dataset mirrors the version-1.0.0 reproducibility artifact:
[10.5281/zenodo.22847291](https://doi.org/10.5281/zenodo.22847291).
The [versioned report and full replication sources](https://github.com/matthematics1137/research-artifacts/tree/quantization-context-compression-v1.0.0/papers/2026-quantization-context-compression)
are maintained together in the research-artifacts repository. Cite the exact
Zenodo version for the frozen evidence; this Hugging Face copy is a discovery
mirror.

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This dataset is the aggregate-only evidence for "No Detected Interaction
Between Deployed Quantization Tier and Lossy Context Compression in Extractive
QA." The study asks whether lossy prompt compression costs more accuracy when
a local deployment of one 27B model moves to a lower-footprint quantized
artifact.

## Headline result

Three deployments of Qwen3.8-27B on a 12 GiB laptop GPU answered the same
extractive questions (120 WildChat, 117 private) under four conditions:

- **Deployments:** Q4_K_XL partly offloaded, EXL3 2.0 bpw, and IQ2_S.
- **Conditions:** the original text, LLMLingua-2, and two rungs of a
  stopword/code-stripping pipeline.

Findings:

- **No interaction detected.** None of the 12 paired difference-in-differences
  tests survives Holm correction. The estimates run from −5.0 to 6.0
  percentage points, with a pointwise interval envelope of −11.6 to 12.9.
  This is a bounded non-detection, not an equivalence result.
- **Compressor choice is the larger effect.** LLMLingua-2 costs 15–21 points,
  and the naive pipeline costs 47–62 points.

## Files

| Path | Description | Rows |
|---|---|---:|
| `data/cells.csv` | Per corpus, deployment, and condition: n, exact-correct count, accuracy with a Wilson interval, paired cost versus original with a Newcombe interval, exact McNemar p, mean request time, and correct answers per minute | 24 |
| `data/interaction.csv` | Per corpus, condition, and low deployment: the difference-in-differences estimate, CR2 interval and SE, Satterthwaite df, CR2 p, Holm p and decision, sign-flip p, and the number of source clusters | 12 |
| `data/claims.json` | The complete aggregate statistical and provenance record the tables are derived from | — |

The tables contain aggregates only. No conversation text, question, answer,
response, per-item row, or identifier from either corpus is included. The
private corpus can never be released.

## Reproduce the paper's numbers

```bash
python3 check_claims.py
```

The standard-library checker in the artifact recomputes every interval, test,
and Holm decision from the per-cluster histograms in `claims.json`. It also
re-derives both tables and every numeric macro printed in the paper.

## Scope

- One dense 27B checkpoint, three artifact–engine–placement deployments, one
  laptop, extractive QA, and one sampled decode per cell.
- The harness requested seed 42, but the pinned TabbyAPI server did not apply
  it for EXL3. Q4 question generation through llama.cpp and item ordering were
  separately seeded. Exact analysis reproduction uses the saved outputs.
- A corrective exploratory rerun, not an independent confirmation.
- Engine, format, footprint, and GPU residency change together.

## License and AI assistance

Original measurement records are offered under CC BY 4.0 to the extent the
author holds rights; code is MIT. WildChat-1M (ODC-By) text is not
redistributed.

Claude (Anthropic) and Codex (OpenAI) assisted with evaluation-harness
development, experiment orchestration, evidence auditing, analysis and figure
code, citation checking, and manuscript drafting and revision. Matthew
Schwartz directed the research and is responsible for its methods, results,
interpretation, claims, citations, rights, and released artifacts.
