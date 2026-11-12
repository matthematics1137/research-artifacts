# A 27-Billion-Parameter Model at Up to 72 Tokens per Second on a 2018 Gaming GPU: Turing Kernels Nearly Double exllamav3’s 2-Bit Decode Speed

Private release-package candidate, version 1.0.0. Independent final review and
Matthew's approval remain pending. No public identifier has been created.
The eventual archive is the paper's reproducibility artifact, not a second
manuscript DOI. The patches remain private until the paper is published.

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

Independent researcher · matthew.schwartz.research@gmail.com

## Five-minute analysis check

With Python 3.12, run from a copy of the extracted artifact root:

```bash
python3 -B check_claims.py
```

The checker validates the export inventory, replays the original claim builder
and paired long-context scoring, and compares all 1027 macros and every generated
TeX fragment byte for byte. A separate named-arm check verifies mode-2
empty-context per-prompt rates. It checks the original 500-item quality lanes
and fourteen paired discordance counts, then the registered Bonsai replay's
nine numeric lanes, five paired/Holm tests and fixed 5,000-replicate intervals.
The fleet check validates two rotated 36-request speed fixtures and eight
500-item quality projections, including completeness and paired IDs. Pooled
fleet speed is recomputed from tokens/time and checked against the summaries.
The macro/fragment count is discovered from the selected scientific freeze.

These checks use no model, GPU, tokenizer, benchmark downloads or network.
Correctness projections count saved verdicts; they cannot independently rescore
missing benchmark response text or establish identity between engines.
The earlier P6 macros remain frozen inputs rather than a fresh P6 analysis.

To redraw the light figures, install `requirements-analysis.txt` in a separate
environment, then run `python3 -B check_claims.py --figures` on an artifact copy.
Figure PDFs must match `paper/figures/` byte for byte. Rebuild the light paper
with `make -C paper` using an already cached Tectonic installation. The checker
writes regenerated files; preserve an immutable extracted stage separately.

## Contents and provenance

`paper/` contains the sanitized light manuscript, bibliography, all generated
TeX fragments and only the light figures used by the manuscript. The private
dark, brief and defense copies are excluded. `replication/` contains the
curated analysis inputs, scorers, registrations, harness sources and patches.
`EXPORT_MANIFEST.json` names source freeze
`def4d933f6b591927a8a8ac605221a2d35deb4f1`, with original/export hashes and
explicit transformations. No source evidence, result or registration was edited.

The eight long-context item files are compact projections of complete original
files verified against their frozen manifests. They omit LongMemEval haystacks,
retain synthetic questions/gold and metadata needed by the scorer, and bind
both full-file and projection hashes. A full-file hash is not a projection hash.
Rebuilding complete prompts needs the pinned data and tokenizer in
`FULL_RERUN.md`; scoring projections are unsuitable as model prompts.

Fleet projections keep numeric timings/counts, task/condition labels, saved
correctness, paired item IDs and response hashes. They omit prompts, benchmark
questions/gold, response text, deployment host names and machine queue wrappers.
Logical labels `card-4070` and `card-4080-super` identify fixtures, not machines
to contact. The rest of the fleet repository and unrelated history are excluded.

## Scope and reuse

The primary study measures one stock 11 GiB RTX 2080 Ti, one model, separate
single-stream sessions and three sampled speed prompts. The two added sm_89
fleet slices use the same 2-bit EXL3 weights/w13 chain over a different HTTP
path; they are deployment observations, not a controlled cross-card ranking.
Nominal seed labels were not forwarded by the pinned TabbyAPI request path.
Registered configured 200/320 W limits lack per-run actual-limit/device/driver
snapshots in the pulled evidence. Model state, context, workload, timing boundary
and resource coordinate must stay attached to each result.

The November 12 refresh uses the old GGUF files, not their November 10–11
replacements. Pooled speed leads do not imply every prompt wins. No adjusted
significant quality difference does not establish equivalence. The Bonsai
counterpoint compares a ternary derivative and normalization, with caps and
request-only timing; it is not an isolated quantizer or engine intervention.

Original code is proposed under MIT; original prose/measurements under CC BY 4.0.
Inherited notices and upstream terms remain controlling. See `CREDITS.md`,
`LICENSE`, `THIRD_PARTY_NOTICES.md` and `environment/license-reads.json`.
Final rights, platform-rule and author review remain pending. Full inference
replication needs separately acquired weights/data, a new reviewed run plan and
machine placement; generated model code needs an isolated execution sandbox.

<!-- canonical-abstract:start -->
## Abstract

Fitting a large model in an older GPU's memory does not ensure fast decoding. We adapted exllamav3's kernels to run the original Qwen3.8-27B weights at 2 bits per weight on a stock 11 GiB RTX 2080 Ti, a 2018 consumer card. Pre-registered rotated rounds compared the patched and stock engines on three prompts. The patched engine decoded 47.53 tokens/s undrafted, 1.88 times stock, and 71.61 with the model's multi-token-prediction (MTP) head, 1.54 times stock; a second session matched within 1%. Most of the undrafted gain comes from an int8 path fitted to Turing's shared memory with unfused projections. Split-K verification and tensor-core attention support drafting and filled contexts. The patches changed no GSM8K outcome in the registered MTP check. On 500 GSM8K and 500 MMLU items, the stock engine's EXL3 weights scored 97.6% and 86.6%; differences from llama.cpp's 2-bit GGUFs were unresolved after adjustment. An 8-bit key-value cache held 72K tokens, and the tensor-core kernel retained the reference kernel's retrieval accuracy at 64K. These measurements show how kernels adapted to an older card recover decode speed after low-bit weights make the model fit. They cover our fastest measured configuration of the original weights on one card and three prompts, not the fastest model on the card; dated comparisons and detailed limits are in the appendix.
<!-- canonical-abstract:end -->

## AI assistance

Claude Code (Anthropic) wrote the Turing patches W7 to W12p, including the fd75d kernel, and ported rafatxf's split-K kernel (W13); wrote the registrations, harnesses and analysis scripts; operated the runs under the author's standing rules and registrations; and drafted the manuscript, figures, claim checks and bibliography. fd75d was checked against an fp64 reference in the run that first timed it, W12p for bit identity, and the builds in registered accuracy checks , with the test outputs in the artifact; every number is recomputed from the archived outputs by a script that refuses to build on any disagreement; and every citation was checked against arXiv, Crossref, GitHub or the source page. Codex (OpenAI) performed the registered Bonsai analysis and formatted its results; in a separate review, independently recalculated the counts, rates and paired tests and checked the timing boundaries and provenance. Codex also implemented the replacement speed harness, revised the manuscript from the independently reviewed comparison, and prepared candidate reading copies and the release package. Matthew Schwartz directed the research and is responsible for its methods, results, claims, citations, rights and released files.
