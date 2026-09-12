# Public design and pre-registration record

The campaign behind this report was pre-registered hypothesis by hypothesis:
each hypothesis was written with a prediction and a falsifier before the block
of experiments that tested it, and its verdict was recorded clause by clause in
the campaign document afterwards. Experiments are numbered E1–E52 and cited by
that label with their sample size. This record lists when each hypothesis was
written relative to its evidence, what changed during the work, and what the
final evidence can support. The laboratory's hypothesis file and campaign
document are private; their verdicts are summarized in the paper's Table A14
and Section 4.

## Pre-registration chronology

| Hypotheses | Written | Relation to the evidence | Tested by |
|---|---|---|---|
| H1–H16 (drafting speed and VRAM on both lanes, prefill collapse, checkpoints, engine versions, draft knobs, temperature, sustained power, effort, n-gram drafting, EXL3 headroom, the daily-driver synthesis, reasoning inflation, marker penalty, recurrent-state precision) | frozen at 08:05 on 2026-09-06 | before the experiment phase | E1–E17, R0 |
| H17–H18 (client-side early exit; the 2.20-bpw branch) | amendment at 16:52 on 2026-09-06 | before the E23 evaluation results and before E24 ran | E23, E23c, E24, E26 |
| H19–H23 (the drafter's VRAM, batching, prefix reuse, forced answers, on-policy recalibration) | amendment at 05:55 on 2026-09-07 | after a research hour and the E28 instrumented loads (the memory figures that motivated H19), before any of E28's TabbyAPI runs and E29–E32 | E28–E32 |
| H24–H30 (error anatomy, bit reallocation, the inflation mechanism, shared serving, long context, edit cost, clock policy) | amendment at 00:40 on 2026-09-08 | before any test of that block; H24 is a measurement from the E32 conversion logs already in hand; H30 was never run (it needs root) | E33, E38–E41 |
| H31–H34 (off-policy calibration, the verifier cap, the sampled-draft verifier, prefill profile) | added during the same block at 01:30, 02:05, 02:50, and 05:20 on 2026-09-08 | each before its test | E35–E37, E42 |
| H35–H37 (checkpoint interval, penalty strength, draft length) | amendment of 2026-09-08, committed with its drivers and the engine patch at 16:35 | before E43–E45 began at 20:46 | E43–E45 |
| H38a, H38b (sample identity and accuracy of drafting at temperature 0; 2-bit against 4-bit) | 01:40 on 2026-09-09 | before any GPU use for the hardening block | E47, E48 |
| H38c (i), (ii) (same-config drafted re-run; divergence anatomy) | 05:30 on 2026-09-09 | after an early read of H38a (25 of the first 33 items identical), before either control ran | E49 |
| H38c (iii) (undrafted same-config re-run) | 14:25 on 2026-09-09 | after (i) had been read, before (iii) ran | E49 |
| H38d (clean CUDA library set) and its speed addendum | 11:45 and 12:30 on 2026-09-10 | before they ran; after the exllamav3 maintainer asked about the lab's library path | E50 |
| H38e (exllamav3 development branch) | 04:10 on 2026-09-12 | before it ran | E51 |
| H38f (released exllamav3 1.5.0) | 08:43 on 2026-09-19 | before it ran; committed first (`8ce7d5f`, time corrected in `6cb41b5`) | E52 |

E46 is a no-GPU analysis of the server's own request logs made after the
evidence freeze of the main campaign; the paper does not cite it.

## Changes and corrections retained in the paper

- **Audit.** After the main campaign, every verdict in the campaign document was
  checked against the row files; 133 corrections were applied before the paper
  was drafted.
- **H23's premise.** The stock 2.00-bpw quant turned out to be already calibrated
  on the model's own reasoning traces, so H23 tested a narrower question than
  written; the paper says so and adds the off-policy control (H31, E42).
- **E27 → E28.** The first reading of the drafter's 2.4 GiB (a cached fp16 head)
  was wrong; E28 traced it to the recurrent-state history slots.
- **Library environment (E50).** Every EXL3-lane server of E7–E49 and the
  E32/E41/E42 conversions ran with a system CUDA 12.1 library path that mixed
  cuBLASLt and NVRTC 12.1 into torch 2.10.0+cu128. A CUDA Cholesky failure the lab
  had patched around was this fault, not an engine bug: the patch is withdrawn
  (kept as history) and its upstream pull request was closed. The pre-registered
  clean-library controls (H38d) left the temperature-0 non-reproducibility in
  place and found no resolvable throughput shift, so the E49 numbers are labelled
  "mixed library set" and the throughput rows stand as measured with a caveat.
- **Non-determinism upstream (E51, E52).** The temperature-0 non-reproducibility
  was reported upstream on 2026-09-10. About 35 hours later a deterministic
  Gated-DeltaNet reduction on exllamav3's development branch made two undrafted
  greedy runs identical on 100 of 100 items (E51). The fix shipped in the
  released 1.5.0, whose unmodified wheel again gave 100 of 100 (E52,
  pre-registered as H38f), and the issue was closed on 2026-09-19. The paper's
  non-identity findings are stated for the measured version (1.4.7).
- **Rounding corrections (2026-09-19).** Building the public claim checker found
  seven figures printed at a neighbouring rounding or definition. The paper and
  the campaign document were corrected to the values the rows give (campaign
  errata table, commit `9a62b2d`); no verdict changed.
- **Negative results kept.** Every falsified or missed prediction is reported with
  its outcome and sample (Table A14), including the identity clause of H38a and
  the engine-determinism predictions of H38c(iii) and H38d.

## Evidence boundary

The main campaign's evidence was frozen at laboratory commit `691e153` plus the
H36 rows of `0cd5ab9`, and extended by the hardening commits `01665a3` (E47),
`0db24bf` (E48), `2f4bdec` (E49), `e339d27`, `ff9d383`, `569bca6` (E50),
`a07f69c` (E51), `d11f652` (E52, the released exllamav3 1.5.0), and `9a62b2d`
(six rounding corrections, listed in the campaign document's errata). This
artifact's export pins every input file to its content at
`9a62b2da50ef8198a3a12dc3837358444cb4d710`, the last commit that touched the
campaign's evidence.

## Final evidentiary boundary

The evidence supports observed throughput, energy, memory, acceptance, and
paired-accuracy measurements for the named configurations on one laptop, and
the mechanisms only where a source reading and a measurement agree (E28, E30,
E35, E37, E38/E43, E40). It does not support statements about other models,
cards, or engine versions, a speed ranking of the engines, a global optimum,
sample identity of drafted and undrafted greedy decoding, or 2-bit-versus-4-bit
parity beyond GSM8K at n = 200.
