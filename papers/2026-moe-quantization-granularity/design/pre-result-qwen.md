# Pre-result working record: Qwen arm addendum

This is a sanitized public rendering of the substantive record preserved in
Git commit `44c515896e9a8dbe0355d81ac01f9e265bf97d1b`, committed
2026-08-29T16:06:48-07:00 before the Qwen downloads completed and before its
evaluation began. The internal original is blob
`50f3ef7cc9868b33ef08e535dd5713830c08dcff`. Its final operational-budget
paragraph is omitted because it contained a private approval quote. No
hypothesis, protocol, outcome band, or scientific caveat was altered.

---

# Design addendum — Qwen3-30B-A3B as the fully-measured 128-expert arm (paper 4)

*Pre-registered 2026-08-29, before any download completed or eval ran. Extends the
pre-registered granularity design (`design-granularity-experiment.md`); the prior-art
gate for the granularity question already passed (`prior-art-granularity-sweep.md`)
and covers this arm — same question, new dose point.*

## Why this arm exists (and why gpt-oss can't be it)

The original close-out plan was a low-bpw tier of gpt-oss-120b to upgrade the
128-expert point from healthy-anchor to a full within-model Δ. **That tier does not
exist and cannot be downloaded** — verified 2026-08-29 across all three major GGUF
publishers via the HF API:

| Repo | Smallest "low-bit" file | Size |
|---|---|---|
| unsloth/gpt-oss-120b-GGUF | Q2_K | 58.3 GiB |
| mradermacher/gpt-oss-120b-i1-GGUF | IQ1_S / IQ1_M | 61.5 GiB |
| bartowski/openai_gpt-oss-120b-GGUF | IQ2_M | 58.4 GiB |

Native MXFP4 is 59.0 GiB. Every published "quant" keeps the expert tensors in
MXFP4 (the QAT format) and requantizes only attention/embedding/router — ~5% of
weights — so "IQ1_M" of gpt-oss is the same effective bpw as the original.
**Ecosystem finding worth one paragraph in the paper:** for a QAT'd MXFP4 MoE, the
quantization ecosystem doesn't even attempt true low-bit requantization. A
self-requant from BF16 would be possible (~61 GiB download + hours of CPU quantize)
but produces PTQ-on-top-of-QAT weights with no imatrix lineage comparable to the
other arms — muddier provenance at higher cost than measuring a clean model.

## The model

**Qwen3-30B-A3B-Instruct-2507** (unsloth GGUFs): 128 experts/layer, 8 active,
30.5B total / 3.3B active params. Same expert count as gpt-oss-120b, but with a
standard imatrix PTQ quant ladder — the same *kind* of artifacts as the Mixtral and
dense arms. gpt-oss remains in the paper as the QAT healthy anchor at 128.

## Tiers (mirroring the Mixtral protocol)

| Tier | File | Size | eff. bpw (size×8/30.5e9) |
|---|---|---|---|
| low | UD-IQ1_M | 9.0 GiB | ~2.53 |
| low-mid | UD-IQ2_XXS | 9.6 GiB | ~2.70 |
| healthy | Q4_K_M | 17.3 GiB | ~4.87 |

Healthy tier lands at the same 4.87 eff bpw as Mixtral Q4_K_M. **Pre-registered
caveat:** the low tiers sit *higher* than Mixtral's (2.53/2.70 vs 1.86/2.15)
because unsloth UD keeps attention/first/last layers high-precision. A null at
2.53 is therefore weaker evidence than a null at 1.86 would be; report eff bpw
alongside every number and say this in limitations. (No lower tier exists:
UD-IQ1_S at 8.4 GiB ≈ 2.37 is the floor of the published ladder — grab it too if
IQ1_M shows an interesting drop.)

## Protocol (identical to Mixtral arm unless stated)

- GSM8K n=100, same items, same prompts, same temp defaults, raw + lenient
  scoring, Wilson CIs, Fisher exact vs healthy tier.
- Suite validity anchors at healthy tier first: math25 n=25 + HumanEval+ n=20
  (check neither floors/ceilings for this model before interpreting Δ on them).
- Runtime: IQ1_M and IQ2_XXS fit fully in 12 GB VRAM (`-ngl 99`); Q4_K_M
  (17.3 GiB) partial offload, standard back-off protocol. 3.3B active params →
  evals should be fast even partially offloaded.
- Same armed gates: cool_gate before launch, ram_gate ≥ 8 GiB, per-run guards,
  setsid nohup + log, labels `QWEN3MOE-{IQ1M,IQ2XXS,Q4KM}-n100` in summary.jsonl
  for idempotent resume.

## Pre-registered predictions (3-outcome)

The granularity hypothesis says fine-grained MoE (many small experts) is robust to
low-bit PTQ, like dense and Flash, unlike coarse Mixtral.

1. **Hypothesis-consistent:** Δ(Q4_K_M − IQ1_M) on GSM8K ≤ 10 pp. The dose-response
   interior fills in on the robust side; 128-expert point becomes fully measured.
2. **Hypothesis-challenged:** Δ ≥ 20 pp (Mixtral-class collapse at *higher* eff bpw
   than Mixtral's low tier). This would be serious evidence against the moderator
   claim — paper 4's central figure gets reworked, not quietly footnoted.
3. **Intermediate (10–20 pp):** graded result; report as measured, claim softens to
   "granularity moderates but does not determine."

Known ceiling risk, pre-registered: this model is modern and strong; healthy-tier
GSM8K may land ≥ 95%, compressing Δ resolution the way Flash's did. If healthy
ceilings, the measured Δ is still valid (all tiers same suite) but small Δs lose
power; math25 becomes the informative suite *if* its healthy anchor is in usable
range (unlike Mixtral, where math25 floored at 1/25).

## Omitted private operational paragraph

The original record ended with a storage-budget and runtime paragraph. It is
omitted from this public rendering because it contained a private approval
quote. The omission does not change the design, predictions, protocol, or
analysis plan above.
