# Pre-result working record: initial Mixtral arm

This is the substantive text preserved in Git commit
`a5c77e0583b4cf2cd7b578992e4fed3b7ea28da3`, committed
2026-08-27T00:39:57-07:00 before the Mixtral evaluation. The original blob is
`3a3024206f3907b51f1e91ed112f25e04c593776`. It is reproduced as a historical
plan, including assumptions later corrected by the audit; it is not presented
as a registered or confirmatory protocol.

---

# Design Doc (pre-registration): Expert Granularity as the Moderator of Quantization Robustness

*2026-08-27. Research Engine stage-3 pre-registration — written BEFORE implementation
per policy. Motivating contradiction (prior-art-moe-sweep.md): MoQE (2310.02410)
finds MoE robust at 2-bit; 2402.18158 finds 8-expert Mixtral MORE fragile than dense;
QMoE succeeds on 2048-expert Switch, fails on Mixtral. Nobody has tested granularity
as the moderator. Our Flash datapoint (512 experts, 1.56 bpw bulk ≈ dense 2.44 bpw on
tasks) is one arm already.*

## Hypothesis
Task-level quantization robustness of MoE models increases with expert granularity
(experts-per-layer count / smaller expert size), because damage to any single small
expert removes less unique capability and routing can compensate. Prediction: at
matched effective bpw (~1.6–2.0), a fine-grained MoE (512 experts) retains
substantially more task accuracy than a coarse MoE (8 experts), with dense as the
known-collapsing baseline.

## Arms (matched-suite, matched-effort, our standard battery)
1. Dense: Qwen3.8-27B at 2.13 eff bpw (DONE — collapsed: 64% math-lenient, 70% HE+).
2. Fine MoE: Flash-Next 512-expert at ~1.7–1.8 eff bpw (DONE — 92% math-lenient,
   85% HE+, 96% GSM8K).
3. **Coarse MoE (TO RUN): Mixtral-8x7B-Instruct** (the literature's own contradiction
   subject) at the lowest available quant tier (IQ1_S/IQ2 class, targeting ~1.6–2.1
   eff bpw) AND at a healthy tier (Q4 class) as its own within-model control.
4. Optional 4th arm if available: a mid-granularity MoE (e.g. 60–64-expert class).

## Metrics & analysis
math25 (raw + lenient), HumanEval+, GSM8K×50; Wilson CIs; Fisher exact on the
low-tier gaps; per-model Δ(healthy-tier − low-tier) is the robustness measure —
comparing Δs across arms is the granularity test (controls for base capability).

## Confounds pre-declared
(a) different base models/training — mitigated by comparing within-model degradation
Δ, not absolute scores; (b) different quant recipes per repo — prefer same maker
(unsloth/bartowski) per arm, report effective bpw always; (c) Mixtral is old
(2023) — its instruct quality floor differs; Δ-comparison mitigates; (d) our n
(20–50/suite) — report CIs, claim direction not magnitude.

## Falsifiers (what kills it)
- Mixtral low-tier Δ ≈ Flash Δ → granularity is NOT the moderator → informative
  null: the contradiction needs a different resolution (training era? QAT-vs-PTQ?
  recipe?) — publishable either way per null-results policy.
- Mixtral low-tier ≈ collapse worse than dense → replicates 2402.18158, granularity
  hypothesis survives.

## Cost & schedule
One download (~10–14 GiB Mixtral low-tier + ~26 GiB Q4 tier), ~2 evening runs of
suites at Mixtral partial-offload speeds. Schedule: after paper-1 release day;
before/parallel with capstone grids. Niche decay: HIGH (Flash per-quant accuracy
niche is empty and public threads are active) — this experiment is the priority
queue head.
