# Design pre-registration — Paper 6: Where to Spend the Bytes

Status: frozen 2026-09-01, before any harness code or measurement. Prior
art: `prior-art-kv-allocation.md` (same date). Runs after Paper 5 (scoop
decay ordering) and never beside the P2R1 campaign.

## Question

On one 12 GiB consumer GPU, (1) do aggressive weight quantizations amplify
KV-cache quantization damage on downstream tasks (interaction), and (2) at
equal total resident bytes, which allocation of the budget among weight
precision, KV precision, and kept context yields the most accuracy
(iso-VRAM policy)? Both are unmeasured; single-axis effects are established
and are engaged, not re-asked.

## Arms

Factorial grid per model:
- Weight tier (3): ~IQ3-class (~3.3 bpw), Q4_K_M-class, Q5-class — drawn
  from the existing paper-1 Qwen3.8-27B ladder artifacts already on disk
  where possible; exact artifacts pinned by repo+revision+sha256 in an
  amendment before the campaign.
- KV type (4): f16, q8_0, q4_0, asymmetric K=q8_0/V=q4_0 (the community
  folk remedy, never formally evaluated).
- Context regime (2): short (4k) and long (24-32k, model-dependent maximum
  that fits the weight tier on 12 GiB).

= 24 cells/model. Primary model: the Qwen3.8-27B ladder (continuity with
papers 1-2). Optional second model (one GQA-architecture contrast, ~8-9B,
shared with Paper 5's artifact set) at the long-context regime only, to
engage the known architecture-dependence finding.

Plus iso-VRAM frontier points: 3-5 configurations tuned so total resident
bytes (weights + KV at the regime's max context) are equal within tolerance,
differing only in allocation; measured on the long-context task.

## Task and scoring

- Short context: the frozen paper-1 suites (MATH-25, HumanEval+, GSM8K
  n=50-100) with the existing scoring harness — direct continuity.
- Long context: a public RULER-style retrieval/aggregation task (needle +
  multi-fact synthesis), n=50 per cell, built once and frozen; this is where
  KV damage is known to concentrate, so it carries the interaction test.
- Seeded decode (temp 0.2, top-p 0.95, completion cap 2048, 600 s timeout),
  one seed per item (variability noted as limitation, consistent with the
  series).

## Pre-registered outcomes (all publishable)

1. Primary interaction: accuracy drop f16->q4_0 KV at long context compared
   across weight tiers (super-additivity), mixed logistic regression
   correct ~ weight_tier * kv_type + context with item random effect, Wald
   test on the interaction, Holm across tasks. Null (additivity) is fully
   publishable: "KV damage does not compound with weight damage" is the
   operator-relevant answer.
2. Co-primary: iso-VRAM ranking at equal bytes (accuracy ordering of the
   allocation strategies, Wilson CIs, pointwise).
3. The asymmetric K8/V4 folk remedy vs q8_0 and q4_0 (first formal test).
4. Descriptive Pareto frontier (accuracy vs total resident bytes) in the
   paper-1 house style.

## Plan

- ~240-340 scored items/cell x 24-32 cells; one P2-scale armed overnight for
  the short grid (~9 h), one to two for long-context cells (slower decode),
  plus iso-VRAM points: ~20-30 GPU-hours total.
- Harness: paper-1 eval suite + P2R1 identity/journal discipline; KV type
  set via `-ctk/-ctv`, placement readbacks required as in P2R1.
- Statistics via the audited paper-4 stats module plus a small
  mixed-model addition (implemented and self-tested before the campaign;
  analysis-bundle style separation as in P2R1).

## Constraints and notes

- Framing is the scientific content: interaction + allocation policy on
  fixed hardware. Never lead with "does KV quantization hurt?" — engage
  the five established findings from the sweep up front.
- Must-cite-and-differentiate: exllamav2 qcache_eval, longctx_bench,
  SCBench, KVQuant(+SqueezeLLM), MiniKV, arXiv:2601.14277.
- Long-context cells at 24-32k on a 27B partial offload may be slow; the
  regime maximum per tier is fixed at harness freeze from measured VRAM,
  not tuned after results.
- Extends papers 1 (weight axis) and 2 (context axis) with their own models,
  suites, and statistics — "extend them further" literally.

## Amendments

(none yet)

### 2026-09-19 — narrowed-question amendment (before any harness code, download use, or measurement)

Authority: Matt, 2026-09-19: "proceed with full finished drafts for all in
2-7 'bring it home' then i can review and edit". This takes the go decision
in `research/DECISIONS-PAPERS-5-6-2026-09-19.md`: pursue the narrower
question rather than retire Paper 6. It also approves the downloads listed
there, except the Q5 file, which is dropped below. The prior-art refresh
`research/prior-art-kv-allocation-2026-09-19.md` §D sets every change here.
Nothing had been measured when this was written. The model downloads were
in progress, and the FA_ALL_QUANTS build had started.

**1. Question and novelty (replaces "Both are unmeasured").** Single-axis
effects and descriptive weight × KV grids at ≥4-bit weights already exist
(Li et al. 2024; exllamav2; KV Pareto; Kim et al. 2026; MoE-XBench; Kozyrev
& Maiboroda 2026; the churn harness). This study is a pre-registered test of
whether architecture (a Gated-DeltaNet hybrid vs standard GQA) moderates the
weight × KV-cache interaction. It uses the sub-4-bit tiers that a 12 GiB
card forces, runs long-context retrieval and a generative task, and adds a
measured equal-memory ranking. The K8/V4 "first formal test" claim is
dropped; that arm is kept only as a deployed configuration. The working
title changes away from "Where to Spend the Bytes", which echoes Kim et al.

**2. Models (both required, co-primary).**

- **Hybrid:** Qwen3.8-27B. KV is cached in 16 of 64 layers (4 KV heads ×
  256 dims), and the other 48 layers are recurrent.
- **GQA contrast:** Qwen3-8B (`unsloth/Qwen3-8B-GGUF` @ `a6adef1…`): dense,
  full attention in all 36 layers, 8 KV heads × 128 dims. It was chosen
  before any screening by a written rule:
  - the same developer and tokenizer lineage as the hybrid;
  - standard full attention everywhere;
  - a thinking-capable chat template;
  - every tier resident on 12 GiB.

  No screening run was made. The prior-art note's floor/ceiling risk is
  accepted and reported. A ceiling or floor lowers power; it does not change
  the analysis.

**3. Weight tiers (matched Unsloth recipes).** UD-Q2_K_XL, UD-IQ3_XXS, and
UD-Q4_K_XL for both models, all pinned by revision and SHA-256
(`testsuite/artifacts/downloads.jsonl`; the on-disk 27B UD-Q4_K_XL is
`3f227079…`).

- The Q5 tier and its 21 GB download are dropped, because on the 27B it
  would not be resident.
- IQ2_S is dropped so that the recipes match across the two models.
- The 27B UD-Q4_K_XL, and any other cell that `--fit` cannot place fully on
  the GPU, runs partly offloaded. Its accuracy counts in the interaction;
  its timing and bytes do not enter the equal-memory analysis.

**4. KV types.** f16, q8_0, q4_0, and K=q8_0/V=q4_0 (K8/V4). The recurrent
state stays F32 whatever `-ctk`/`-ctv` say; this is reported, so "KV" is
not read as the whole sequence state.

**5. Engine.**

- llama.cpp `035e22731a7f`, the pinned commit, rebuilt in
  `engines/llama.cpp-p6fa` with `GGML_CUDA_FA_ALL_QUANTS=ON`. Its other
  options equal the pinned build's (CUDA 12.6, sm_89, Release, shared
  libraries). This build serves every cell, f16 included, so the engine is
  constant. The pinned build is untouched.
- Serving settings: flash attention on; `--fit on` decides GPU layers, and
  they are recorded per cell; one slot; `cache_prompt: false` on every
  request (llama.cpp issue #28527: a quantized K cache makes cached repeats
  diverge).

**6. Tasks.**

- **Primary: long-context retrieval at 16,384 haystack tokens.**
  - 200 synthetic items, built once by a seeded, frozen generator before
    any measurement: 4 RULER-style task families × 50 items.
  - Families: single needle; multi-key needle (1 of 4 keys asked);
    multi-value (4 values for one key, all required); variable tracking (a
    5-hop assignment chain, all names required).
  - Haystack: LongMemEval-S sessions rendered as plain chat text (MIT).
    Needles are placed at depths spread evenly through the haystack.
  - Length is counted with the Qwen3.8 tokenizer, and the Qwen3-8B count is
    reported too.
  - Answering: thinking off, greedy, at most 128 tokens. Scoring is a
    deterministic string match in which every required value must appear.
- **Secondary: GSM8K-100.** The lab's frozen `gsm8k_100.jsonl` with the
  Paper 1 scorer. Thinking on: `reasoning_effort` medium for Qwen3.8, and
  the template default for Qwen3-8B. Greedy, completion cap 8,192. These
  long decoded outputs are where KV error can accumulate.
- **Decoding.** Temperature 0 on every request, a deviation from the frozen
  0.2: per-item flips are interpretable only under greedy decoding.
- **Dropped.** The frozen MATH-25 and HumanEval+ short-context suites.
  Their n is too small to estimate an interaction.

**7. Estimands and tests.**

- **Definitions.** KV damage for model m at weight tier w is D_m(w) =
  acc(w, f16) − acc(w, q4_0). The interaction is I_m = D_m(Q2) − D_m(Q4);
  a positive value is super-additive, meaning more KV damage at lower
  weight precision. The moderation estimand is M = I_hybrid − I_GQA.
- **Unit of analysis.** Every estimand is a paired item-level contrast over
  the same 200 items. Each item contributes a sum of ±1 indicators over 4
  (I) or 8 (M) cells.
- **Co-primary 1 (long context):** M ≠ 0, two-sided. Tested by a sign-flip
  permutation over items (100,000 draws, fixed seed), with a t interval.
- **Co-primary 2 (long context):** I_hybrid is equivalent to 0 within ±10
  points. Tested by TOST at 0.05, that is, whether the 90% interval lies
  inside ±10.
- **Multiplicity.** Holm across the two co-primaries.
- **Pre-registered predictions** (from prior art): I_hybrid is equivalent
  to 0, and I_GQA ≥ I_hybrid.
- **The frozen mixed logistic regression is replaced** by these paired
  item-level contrasts. No GLMM library is on the lab machine, and the
  contrasts answer the same question with fewer assumptions.
- **Secondary (descriptive):**
  - each KV type vs f16 per model and tier (exact McNemar, pointwise);
  - the IQ3 tier's D, and the trend over tiers;
  - K8/V4 vs q8_0 and vs q4_0;
  - the same estimands on GSM8K-100;
  - paired flips vs each f16 cell;
  - accuracy by needle depth.

**8. Equal-memory ranking (co-primary 2 of the frozen design, now
descriptive with a pre-registered prediction).**

- **Scope.** For each model, among the cells fully resident at 16K, record
  the measured device memory: the server's reported model, KV, and compute
  buffers, plus the process's nvidia-smi figure. Rank the cells by
  long-context accuracy, with Wilson intervals.
- **Prediction.** On the hybrid, f16 → q4_0 at 16K frees about 0.72 GiB,
  less than one weight step (UD-Q2_K_XL → UD-IQ3_XXS is +1.1 GB), so KV
  precision buys at most one weight step. On the GQA contrast, the same
  change frees about 1.6 GiB.
- **Input context.** Bytes at 32K are reported as computed, not measured.

**9. Cost and order.** 24 cells × (200 long-context + 100 GSM8K items),
about 30–40 GPU-hours. It runs after Paper 5's GPU work and never beside it.
The operational rules (pause file, thermal pause, server stopped while
paused) are Paper 5's and remain adjustable.

**Clarifications 2026-09-19 14:35 (harness written; nothing measured).** The
amendment left these open, and the harness had to fix them. Each is recorded
before any cell ran; the reasoning is in `testsuite/paper6/HARNESS_NOTES.md`.

1. **"A 5-hop assignment chain, all 5 names required"** is read as five
   assignment statements and five names: one value assignment and four
   variable-to-variable hops. The other reading needs six names and
   contradicts "all 5 names required".
2. **The 16,384-token haystack length counts the inserted needles.** Built
   items land at 16,340–16,384 tokens.
3. **The Qwen3-8B token count** cannot be produced offline, because only its
   GGUF is on disk. Each server counts its own prompts through `/tokenize`
   once per model, and the analysis reports both counts. The frozen items do
   not depend on it.
4. **"Fully resident"** for the equal-memory table means every layer
   offloaded, no KV or recurrent buffer on the host, and no layer split by
   `--fit`, which can report all layers offloaded while still splitting one.
   Buffers are measured at the served 20,480 cells, and KV bytes are scaled to
   16,384 and 32,768, which matches the amendment's "computed at 32K".
5. **Prompt-token parity is recorded, not gated.** Paper 5 gates it because a
   budget is its independent variable. Paper 6 has no budget, so a one-token
   difference between the `tokenizers` count and llama.cpp's must not abort an
   overnight cell. The deltas are reported.
6. **Scoring** is case-insensitive containment of every required string, as in
   RULER's `string_match_all` and the lab's Paper 1 needle scorer.
7. **No sampling field other than `temperature: 0`** is sent. llama.cpp is
   greedy at temperature 0, so top-p, top-k, and a seed cannot change the
   output, and the server's own defaults stay visible in `/props`.
