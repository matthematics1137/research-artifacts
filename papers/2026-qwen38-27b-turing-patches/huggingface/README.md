---
pretty_name: "Qwen3.8-27B on an 11 GiB RTX 2080 Ti — frozen measurements"
license: cc-by-4.0
tags:
  - reproducibility
  - llm-inference
  - quantization
configs:
  - config_name: quality_summary
    data_files:
      - split: train
        path: quality.csv
  - config_name: fleet_quality_summary
    data_files:
      - split: train
        path: fleet_quality.csv
---

# A 27-Billion-Parameter Model at Up to 72 Tokens per Second on a 2018 Gaming GPU: Turing Kernels Nearly Double exllamav3’s 2-Bit Decode Speed

Private discovery-card candidate for version 1.0.0. Final independent review,
rights clearance and author approval remain pending. No Hub repository,
preprint or DOI has been created by this preparation. The eventual Hub card
indexes the paper and its versioned artifact; it is not another manuscript DOI.

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

Independent researcher · matthew.schwartz.research@gmail.com

## Measurement summaries and provenance

`quality.csv` summarizes the original 500-item GSM8K/MMLU quality lanes on the
stock 11 GiB RTX 2080 Ti. `fleet_quality.csv` is generated from eight saved
500-item verdict projections for two separately registered RTX 4070/4080 SUPER
slices, shared 2-bit EXL3 weights and w13 target. Its logical card labels are
fixture coordinates, not network hosts. The `train` splits are viewer containers
for measurements, not training examples. Accuracy is percent; original summary
wall time is seconds/answer and decode rate tokens/second.

The archival artifact contains the definitions, numeric projections, scorers,
registrations, code and original/export hashes. Its default `check_claims.py`
replays the original claims, Bonsai paired/Holm/bootstrap analysis and fleet
pooled speed/counts without model execution. `--figures` redraws the light
figures with the pinned Matplotlib 3.10.6. See `CLAIM_EVIDENCE_MAP.md` and
`FULL_RERUN.md`. Saved correctness flags cannot independently rescore absent
responses. This summary card cannot replace the evidence archive.

## Upstream artifacts and limits

No weights, benchmark questions/gold, model responses or LongMemEval haystacks
are included. The measured old Unsloth GGUF revision is
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, not its November 10–11 replacements;
EXL3 is pinned to `d7e02d63dddd4e047a8d397acb22aeb2b0bedcec`.
Bonsai uses the pinned WaveCut conversion and ProCreations MTP adaptation.
The artifact's complete model inventory distinguishes upstream/local derivatives.

The main three-prompt Turing workload and the HTTP fleet slices have separate
sessions and timing boundaries; these are not a controlled cross-card or universal
engine ranking. The fleet harness's nominal seeds were not forwarded by the
pinned TabbyAPI path. Actual per-run device/driver/power-limit snapshots are
absent from the pulled fleet evidence. Failure to detect an adjusted quality
difference is not equivalence. Keep every number with its task, model state,
resource coordinate and stated limits.

## Rights and citation

Original numeric summaries are proposed under CC BY 4.0. Code, linked weights,
upstream material and adaptations retain their own notices. The artifact's
`CREDITS.md`, `THIRD_PARTY_NOTICES.md` and exact license texts identify W13's
rafatxf MIT port, buun's MIT related work and NInfer's Apache-2.0 engine separately
from NInferno. No weights or engine runtime is redistributed here. Matthew's
rights and release approvals remain pending. Identifiers stay blank until
separately authorized provider operations return verified versioned records.

<!-- canonical-abstract:start -->
## Abstract

Fitting a large model in an older GPU's memory does not ensure fast decoding. We adapted exllamav3's kernels to run the original Qwen3.8-27B weights at 2 bits per weight on a stock 11 GiB RTX 2080 Ti, a 2018 consumer card. Pre-registered rotated rounds compared the patched and stock engines on three prompts. The patched engine decoded 47.53 tokens/s undrafted, 1.88 times stock, and 71.61 with the model's multi-token-prediction (MTP) head, 1.54 times stock; a second session matched within 1%. Most of the undrafted gain comes from an int8 path fitted to Turing's shared memory with unfused projections. Split-K verification and tensor-core attention support drafting and filled contexts. The patches changed no GSM8K outcome in the registered MTP check. On 500 GSM8K and 500 MMLU items, the stock engine's EXL3 weights scored 97.6% and 86.6%; differences from llama.cpp's 2-bit GGUFs were unresolved after adjustment. An 8-bit key-value cache held 72K tokens, and the tensor-core kernel retained the reference kernel's retrieval accuracy at 64K. These measurements show how kernels adapted to an older card recover decode speed after low-bit weights make the model fit. They cover our fastest measured configuration of the original weights on one card and three prompts, not the fastest model on the card; dated comparisons and detailed limits are in the appendix.
<!-- canonical-abstract:end -->

## AI assistance

Claude Code (Anthropic) wrote the Turing patches W7 to W12p, including the fd75d kernel, and ported rafatxf's split-K kernel (W13); wrote the registrations, harnesses and analysis scripts; operated the runs under the author's standing rules and registrations; and drafted the manuscript, figures, claim checks and bibliography. fd75d was checked against an fp64 reference in the run that first timed it, W12p for bit identity, and the builds in registered accuracy checks , with the test outputs in the artifact; every number is recomputed from the archived outputs by a script that refuses to build on any disagreement; and every citation was checked against arXiv, Crossref, GitHub or the source page. Codex (OpenAI) performed the registered Bonsai analysis and formatted its results; in a separate review, independently recalculated the counts, rates and paired tests and checked the timing boundaries and provenance. Codex also implemented the replacement speed harness, revised the manuscript from the independently reviewed comparison, and prepared candidate reading copies and the release package. Matthew Schwartz directed the research and is responsible for its methods, results, claims, citations, rights and released files.
