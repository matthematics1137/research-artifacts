# No Detected Interaction Between Deployed Quantization Tier and Lossy Context Compression in Extractive QA

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This artifact accompanies the version-1.0.0 technical report and preserves the
aggregate P2R1 measurements, the complete statistical record, a standalone
verifier, and the frozen measurement and analysis sources.

- Exact artifact version: [10.5281/zenodo.22847291](https://doi.org/10.5281/zenodo.22847291).
- Versioned source and report: [GitHub tag `quantization-context-compression-v1.0.0`](https://github.com/matthematics1137/research-artifacts/tree/quantization-context-compression-v1.0.0/papers/2026-quantization-context-compression).
- The artifact's version-1.0.0 citation date is 2026-09-28. The DOI identifies
  the frozen mixed-license compendium; no separate manuscript DOI is minted.

## Question and result

When a local deployment of one 27B model moves to a lower-footprint quantized
artifact, does lossy prompt compression cost more accuracy than it did at the
higher-footprint deployment?

Three real artifact–engine–placement deployments of Qwen3.8-27B on one
12 GiB laptop GPU answered the same extractive questions under four context
conditions: the original text (O), LLMLingua-2 (L2), and two rungs of a
deterministic stopword/code-stripping pipeline (A, E). Exact accuracy, in
percent:

| Corpus (items) | Deployment | O | L2 | A | E |
|---|---|---:|---:|---:|---:|
| WildChat (120) | Q4_K_XL, 16.4 GiB, partly offloaded | 88 | 70 | 25 | 25 |
| WildChat (120) | EXL3 2.0 bpw, 9.5 GiB, full GPU | 84 | 63 | 24 | 23 |
| WildChat (120) | IQ2_S, 7.8 GiB, full GPU | 86 | 68 | 28 | 25 |
| private (117) | Q4_K_XL | 88 | 71 | 41 | 38 |
| private (117) | EXL3 2.0 bpw | 91 | 75 | 38 | 37 |
| private (117) | IQ2_S | 89 | 72 | 40 | 37 |

- **No cross-tier interaction was detected.** The 12 paired
  difference-in-differences estimates range from −5.0 to 6.0 percentage
  points. Holm's step-down procedure over 12 CR2/Satterthwaite
  source-conversation-cluster-robust tests rejects none of them. The
  unadjusted p-values run from 0.06 to 1.00, and every pointwise 95% interval
  contains zero.
- **This is a bounded non-detection.** The interval envelope is −11.6 to
  12.9 points, and no equivalence margin was pre-specified. The result
  establishes neither independence nor equivalence.
- **Compressor choice is the larger effect.** LLMLingua-2 costs 15–21 points.
  The naive rungs cost 47–62 points at comparable character retention.
- **Original context is fastest per correct answer.** Within every
  deployment and corpus, the original context gives the most correct answers
  per recorded serving minute.

This is a corrective exploratory rerun (P2R1) of a quarantined earlier grid.
It was designed after those contaminated outputs and the hypotheses had been
seen, so it is not an independent confirmation.

## Comparison with prior work

[Ahmed et al. (2025)](https://arxiv.org/html/2508.00305) already combine
GPTQ-4bit with LLMLingua-2 on LongBench. This report's narrower contribution is
a paired estimate of how compression cost changes across three deployed
low-bit stacks, with source-conversation-clustered uncertainty. Its generated
exact-match QA scores are not directly comparable to LongBench F1.
[Ayadi et al. (2026)](https://arxiv.org/html/2607.10855) include filler-word
deletion in their quantized-model robustness tests; that overlaps with our
stopword condition, but not the full learned-versus-aggressive-compression
grid. [Shahrabi-Farahani and Rahmati (2026)](https://arxiv.org/html/2608.18578v2)
study an input--precision interaction from conflicting history, rather than
context removal. The primary-source search refreshed on 2026-09-28 located
no exact replacement for this deployment comparison. It is not a claim of
first combining quantization and compression or of a general independence law.

## Evidence status

The official attempt is `p2r1-primary-v5`. It ran from 2026-09-13 to
2026-09-16 under a fail-closed harness with these checks:

- identity-verified servers;
- exact loader placement readbacks (`33/66` and `65/65`);
- hash-chained append-only journals;
- pinned CUDA runtimes;
- full artifact hashes before and after timing.

All six runs and 24 cells completed: 2,844 scored requests, with 0 evaluation
and 0 question-generation request errors. Four earlier attempts stopped
fail-closed. Each restart was a committed protocol amendment with regenerated
questions (see `DESIGN_RECORD.md`), and none of the amendments changed inputs,
prompts, decoding, scoring, or analysis.

The promotion record that binds all private evidence has SHA-256
`454b9bd0a19003957efbda9b3d6a47918142fd98103cdcd7580bc1e4cf57b711`.

## Artifact contents

- `data/claims.json`: the complete aggregate statistical and provenance record.
  It holds per-cell counts, paired discordance counts, interval and test
  results, a per-contrast histogram of source-conversation clusters, answer
  survival tallies, and path-free hardware, software, and artifact
  provenance.
- `data/cells.csv` and `data/interaction.csv`: flat tables derived from
  `claims.json` for browsing.
- `data/EXPORT_MANIFEST.json`: the hashes that bind these files to the promoted
  evidence and to the replication sources.
- `check_claims.py`: a standard-library recomputation of every accuracy,
  Wilson interval, Newcombe method-10 paired interval, exact McNemar test,
  CR2 variance, Satterthwaite degree of freedom, cluster-robust t test,
  sign-flip sensitivity p-value, Holm decision, headline range, and all 414
  numeric macros printed in the paper.
- `environment/`: path-free hardware, software, model-artifact, and protocol
  records.
- `DESIGN_RECORD.md`: the pre-registration and amendment chronology.
- `FULL_RERUN.md`: the documented inference path.
- `REPLICATION_SOURCE.md` and `replication/`: the exact frozen measurement
  harness (22 files) and post-run analysis sources (8 files), each
  hash-verified at export.
- `paper/main.pdf` and `paper/main.tex`: the technical report and its
  sanitized, reproducible source.

Conversation text, windows, questions, gold answers, model responses, per-item
rows, item or conversation identifiers, model weights, private paths, and
secrets are excluded.

## Five-minute analysis quickstart

```bash
python3 check_claims.py
```

It needs Python 3.11 or newer and its standard library only. It runs in
seconds with no weights, network access, or GPU, and it prints `ALL PAPER 2
P2R1 PUBLIC CLAIM CHECKS PASSED`. It checks exact frozen hashes, so a changed
data file is reported as an error, not silently accepted.

To rebuild the paper from the shipped figures, install Tectonic 0.17.0 and
populate its standard bundle cache once. Then run:

```bash
make -B -C paper
```

## Why only aggregates

The private arm is the author's own assistant-chat history and can never be
released. WildChat is distributed under ODC-By, but that database license does
not necessarily clear the independent rights or privacy of each conversation.
Windows, questions, answers, and responses therefore stay private in both
arms.

The per-cluster histograms in `claims.json` are enough to recompute every
cluster-robust test exactly, without any item order or identifier.
`FULL_RERUN.md` explains how an independent rerun regenerates its own
questions from the pinned WildChat revision.

## Scope limits

- **Scope.** One dense 27B checkpoint, three deployed configurations, one
  laptop, extractive short-answer QA, and one sampled decode per cell.
- **Seed handling.** The harness requested seed 42, but the pinned TabbyAPI
  server did not apply it for EXL3. Seeded Q4 question generation through
  llama.cpp and seeded item ordering are separate. Exact analysis reproduction
  uses saved outputs; the recorded EXL3 seed cannot reproduce new responses.
- **Confounding.** Engine, format, footprint, and GPU residency change
  together, so nothing is attributed to weight precision alone.
- **Character retention.** Retention is measured in characters. It is not
  token retention or KV-cache memory, and all inputs already fit.
- **Timing.** Timing is end-to-end request time, excluding the offline
  compression pass. No prefill/decode mechanism is claimed.
- **Question generation.** The higher-footprint deployment generated the
  questions. The private long/no-code stratum has only 6 windows.

## Rights and AI assistance

Authored code is MIT-licensed. Original documentation and measurement records
are offered under CC BY 4.0 to the extent the author holds rights. Upstream
dataset, model, and engine terms remain controlling; see
`THIRD_PARTY_NOTICES.md`.

Claude (Anthropic) and Codex (OpenAI) assisted with evaluation-harness
development, experiment orchestration, evidence auditing, analysis and figure
code, citation checking, and manuscript drafting and revision. Matthew
Schwartz directed the research and is responsible for its methods, results,
interpretation, claims, citations, rights, and released artifacts.
