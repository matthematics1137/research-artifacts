# Retrieval Yields 6.2 Times as Many Correct Answers as Recency Truncation When a Local 27B Model's Chat History Overflows

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->
Version 0.1.0, private candidate. The ambassador selected the main title; redistribution licenses, final exact-file review and release authorization remain outstanding. No public artifact identifier has been issued for this candidate.

This is a benchmark report, not a new memory method. On 500 LongMemEval-S questions at a 4,096-token full-prompt cap, retrieval preserved 355 judged-correct answers (71.0%) and recency truncation 57 (11.4%): 6.2 times as many. Summary and compression were intermediate. Oracle evidence achieved 153/157 on the fitting subset; this excludes non-fitting questions and removes distractors, so it does not establish a universal selection-only explanation of forgetting.

## Verify scored-data analysis

With Python 3.12 or later:

```sh
python3 -B verify_inputs.py --staged
python3 -B check_claims.py
```

For figures, install the versions in `requirements-analysis.txt` in an environment you control, then run:

```sh
python3 -B verify_inputs.py --staged --render-check
```

The public checker reconstructs primary aggregates, uncertainty calculations and generated claim text from sanitized scored records. It does not rescore withheld answers, rerun model inference, establish that original requests executed, or reproduce LoCoMo's conversation-bootstrap token-F1 intervals. LoCoMo is aggregate-only. Read `replication/analysis/README.md` for the exact fields and limits.

`INPUTS.json` binds scientific components. Controller-generated paper bytes are checked in staged mode; `SHA256SUMS` and `release-files.json` bind the final archive. Checksums detect changes, not coordinated replacement or provenance without an approved release identifier.

## Contents and boundaries

`paper/` contains only the canonical light manuscript, references, numeric claims and three figures. `replication/analysis/` contains sanitized outcomes and portable numerical/rendering code. `replication/experiment/` contains source, asset pins, historical runtime observations and inference-replication instructions. No models, original benchmark corpora, conversations, question/answer/judge text, credentials or private reading companions are included.

The experiment runbook is frozen historical documentation, not evidence of a new campaign. The fresh-experiment adapter does not yet provide a complete prospective execution gate; see `FULL_RERUN.md` before planning any new inference run. A public analysis replay is distinct from full experimental replication.

See `CLAIM_EVIDENCE_MAP.md`, `THIRD_PARTY_NOTICES.md` and `LICENSE`. External datasets, engines and models retain their own licenses; none is relicensed by this candidate.

## AI assistance and responsibility

Claude Code (Anthropic) assisted amendment drafts, harness development, orchestration, analysis and manuscript preparation. Codex (OpenAI) assisted request pacing, isolated follow-up preparation, evidence checks, scorer diagnostics, analysis-to-manuscript generation, figures and editorial revision. Matthew Schwartz is responsible for methods, claims, citations, rights and released files. His review of the exact revised release and final approval remain required; automated checks do not confer approval.
