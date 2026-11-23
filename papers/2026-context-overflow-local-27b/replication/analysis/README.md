# P5 analysis-only data candidate

This is a private release component, not a publication or author approval.
It requires a completed, privately audited primary study and BOTH follow-ups.
No real candidate is created while any of those measurements is missing.

With Python 3.12, run: python3 -B check_results.py

No model, network, GPU, repository checkout, or third-party Python package is
required for this numerical/text check. It reconstructs every primary aggregate,
all cross-model binary results, and their valid-verdict diagnostics from the
sanitized scored records. This includes paired Wilson/Newcombe/McNemar/Holm
calculations, type/abstention subsets, the oracle-fit subset, timings, evidence
retention diagnostics, truncations, and all three scorer-agreement pairs.
The four statistical functions and the primary aggregation block are copied
verbatim from the study's audited sources. Local export additionally requires
independent SciPy statistics, raw-request audits, tokenizer checks, execution
receipts, and replay of both frozen follow-up analyzers.

The checker also recreates claims.tex: all numeric macros, abstract, results
tables, discussion, conclusion and follow-up summary. Its text functions are
allowlisted, unchanged copies of the local paper renderers. They consume
checked scores/aggregates, not additional observations or edited statistics.

To rebuild the results TeX plus all three figures in BOTH light and dark
styles, install Matplotlib and run:

    python3 -B rebuild_results.py --output ../p5-rendered

The render receipt records Python, NumPy and Matplotlib versions and every
output hash. Output is separate from the immutable input candidate; an exact
rerun is allowed, but changed or unrelated existing output is never replaced.
Font/plot-library environments can change PDF bytes. These are results
components, not a full manuscript compilation or final author reading set.

## What is—and is not—independently checked

Public replay starts from archived scores, not withheld answer or judge text.
It cannot rescore language-model correctness or prove that the recorded
requests ran. Valid verdict means a successful nonempty reply, not validated
correctness. The registered yes-substring rule remains unchanged, including
noncanonical replies and any credited empty visible answers.

LoCoMo remains aggregate-only. Its public contingency counts reproduce binary
accuracy, intervals, and paired comparisons. Its diagnostic counts and token-F1
means/percentile intervals are archived aggregates whose shape, denominator,
selection rule, and internal consistency are checked. They are NOT a public
reconstruction of its conversation bootstrap. That bootstrap is verified
privately from all 200 question scores before export. This limitation is
intentional: no LoCoMo item or conversation identifiers, per-question scores,
questions, answers, or transcript excerpts are exported. Follow-up intervals
remain descriptive and outside the primary Holm family; shared-conversation
dependence limits the question-independent binary intervals.

## Fields and exclusions

LongMemEval uses opaque lme-NNN IDs in the frozen 500-item order, shared across
the primary and cross-model experiments. Source-ID hashes support mapping
against a separately obtained benchmark. Observation order follows the frozen
analyzer's prompt iteration order, not arrival time. Oracle rows are only the
frozen fitting subset, not all 500 questions.

Only allowlisted scalar fields are included: outcomes, valid/noncanonical
judge flags, containment eligibility/score, empty/unfinished thinking flags,
token counts, HTTP durations, prompt/response hashes and policy diagnostics.
Server-usage completion counts may be absent; harness-tokenizer counts are
separate and are not silently substituted into the original analysis.
HTTP durations are not throughput or total end-to-end policy costs.
Evidence turns retained, sessions summarized, and evidence-word recall are
different diagnostics; none is silently relabeled as answer preservation.

Raw prompts, conversation/question/answer/reasoning/judge text, source IDs,
weights, local paths, host/process identifiers, and raw logs are excluded.
Source bindings are SHA256(path-string) -> SHA256(file-bytes), not readable
paths. Response hashes bind canonical private terminal records but cannot
reveal or independently validate those withheld records.

Checksums detect changes against this manifest, not coordinated replacement
of data, code and manifest. The eventual author-approved release identifier
must establish provenance. This component does not compile the full paper,
establish redistribution rights, grant author approval, or publish anything.
