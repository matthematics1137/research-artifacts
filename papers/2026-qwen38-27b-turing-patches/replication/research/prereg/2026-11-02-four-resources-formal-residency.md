# Pre-registration: four resources formalized (C2), part 2: residency and prefill composed from P8 stage 1, predicted for stage 2's calibration arms, stage 2 and stage 3

Registered 2026-11-02 02:56 EST by the C2 CPU researcher of the ideas program. The first commit of this file freezes it. Later
changes are dated amendments appended at the end.

- Gate: `research/gates/2026-11-02-ideas-program.md`, cluster C2. Prior art:
  `research/prior-art-four-resources-formal-2026-11-02.md` (O2 and O6 as narrowed).
- RIFE: research-ledger-reference (stage 1), research-ledger-reference (P8).
- CPU only. Every input is read only. Nothing in `testsuite/p8_longctx/` or `testsuite/results/turing/longctx/` is
  edited. The desktop and its queue are not touched.

## Question

Do residency and prefill time compose from per-component rates? Residency would be the sum of co-resident parts,
with transients combining by maximum over time. Prefill would be a sum over chunks of a per-chunk cost that grows with
depth. If they compose, rates fitted on stage 1 should predict other sessions at the same operating points, new
depths and extrapolated depths.

## Training data (seen in full)

`testsuite/results/turing/longctx/stage1/results.json` and its report: 13 steps, with peak memory, prompt tokens and
prefill time. Also seen: the stage 1 registration's projections and the stage 2-3 registration.

**Not seen:** any value in `testsuite/results/turing/longctx/stage2/`. Its file list and journal keys were read to learn
what it records: two calibration arms at 64K `8,8` with 72 items, `peak_mib` per arm, and `prompt_tokens` and
`prompt_time_s` per item.

## Residency model R1 (composition: sum across co-resident parts, plus one transient)

peak(fmt, L, P) = F(fmt) + s(fmt) x L/1024 + sigma x [P > 65,536], in MiB, where:
- L is `cache_size` = `max_seq_len`, since the cache is allocated at load, and P is the longest prompt the server
  prefilled.
- s(fmt) is the cache arithmetic for 16 attention layers plus the MTP layer, in MiB per 1K tokens: fp16 68.0;
  `8,8` 36.125; `8,4` 27.625; `4,4` 19.125. The per-layer, per-token bytes are 1024 x (bk/8 + 1/16) for keys plus
  1024 x (bv/8 + 1/16) for values, with one fp16 scale per 32 values; fp16 is 4096.
- F(fmt) is fitted on the stage 1 steps with P <= 65,536:
  - `8,8` 7,785.0, `4,4` 7,791.0 and `8,4` 7,787.0 MiB;
  - fp16 7,923.7 (mean of three steps). fp16 carries about 135 MiB more fixed memory than the quantized formats: an
    operating-point dependence of a "fixed" part.
- sigma is a prefill staging step for prompts longer than 65,536 tokens. Fitted as the mean excess of the 7 longer
  steps: 107.6 MiB (SD 9.6; range 91-117). No second step appeared at 131,072: the 136K step's prompt was 138,702
  tokens, with excess 103.

**Nulls:**
- N1, stage 1's census projection: refuted at 64K by -204 to -222 MiB.
- N2, no staging step (sigma = 0).
- N3, one shared intercept for all formats (7,787.7 MiB).

## Prefill model PF1 (composition: sum over 1,024-token chunks of linear work plus attention that grows with depth)

Per-token time t(P) = alpha + beta(fmt) x P/2, and T(P) = P x t(P). Fitted by least squares on stage 1's 13 points:
- alpha = 1.3985 ms per token;
- beta = 9.928e-6 (`8,8`), 1.0437e-5 (`4,4`), 1.0506e-5 (`8,4`) and 1.1257e-5 (fp16), in ms per token squared;
- training RMS 1.06%. The `8,8` 64K point lies 3.2% below the fit.

**Nulls:**
- N4, the flat rate: each format's mean stage 1 per-token time, with no depth term.
- N5, one beta for all formats (alpha 1.4075, beta 1.0202e-5; training RMS 1.27%).
- N6, replication: the stage 1 rate at the same format and length, where one exists.

## Predictions

Peaks are maxima of 1 Hz nvidia-smi samples, as in stage 1. A replicate is the same format and length as a stage 1
step.

| Id | Arm (source) | Predicted peak, MiB | Band | Predicted prefill |
|---|---|---|---|---|
| R-c | stage 2 calibration arms, 64K `8,8`, each arm (on disk) | 10,097 | ±40 | per item P x t(P); 111.8 s at P = 64,980 |
| R-a | stage 2, 64K `8,8` (replicate) | 10,097 | ±40 | as R-c |
| R-b | stage 2, 64K `4,4` (replicate) | 9,015 | ±40 | 112.9 s at P = 64,980 |
| R-d | stage 2, 64K `8,4`, if the rescue arm runs (replicate) | 9,555 | ±40 | 113.1 s at P = 64,980 |
| R-e | stage 2, 32K fp16 (replicate) | 10,100 | ±40 | 633 tok/s at P = 32,256 |
| R-f | stage 2, 32K `8,8` (**new depth**) | 8,941 | ±60 | 642 tok/s at P = 32,256 |
| R-g | stage 2, 32K `4,4` (**new depth**) | 8,403 | ±60 | 638 tok/s at P = 32,256 |
| R-h | stage 3, 96K `4,4` (replicate) | 9,735 | ±40 | 187 s at P = 97,774 |
| R-i | stage 3, 128K `4,4` (replicate) | 10,347 | ±40 | 271 s at P = 130,546 |
| R-j | stage 3 probe, 152K `4,4` (**extrapolation**) | 10,806 | ±40 | about 343 s at P = 155,136 |
| R-k | stage 3 probe, 168K `4,4` (**extrapolation**) | 11,112 | ±40 | about 393 s at P = 171,520 |

**Pass or fail by the 512 MiB headroom rule (peak at most 10,752 MiB):**
- **R-j: 152K fails** by 14-94 MiB across the whole band. Without the staging step (N2) it would pass at 10,698 MiB, so
  this probe discriminates R1 from N2.
- **R-k: 168K fails** with under 200 MiB free, and may end in an out-of-memory error.

**Reach implied by R1** (the largest length at most 10,752 MiB):
- `4,4` about 149K, so a 144K step passes with about 100 MiB to spare and 148K passes with about 25;
- `8,4` about 103K, so 100K passes and 104K fails;
- `8,8` about 79K, consistent with stage 1's 80K failure by 25 MiB (R1 gives 10,783 at 80K against 10,777
  measured);
- fp16 about 41.5K.

These refine stage 1's lower bounds. No registered step tests the 144K, 148K, 100K or 104K steps.

**Judging:**
- **Peaks:**
  - a replicate holds if the peak is within ±40 MiB, and a new depth or extrapolation within ±60 MiB (±40 for R-j and
    R-k as tabled);
  - "partly" if within twice the band;
  - "refuted" beyond that.
- **Prefill:**
  - an arm's pooled prediction (sum of predicted over sum of measured times) holds within ±5% for replicates and
    interpolation, ±7% for the new depths and ±8% for the extrapolations;
  - the item-level RMS should be at most 6%.
  - It is also reported with each arm's first item left out, in case first-use kernel work inflates it.
- **Arms not on disk** at compare time stay pending, with the predictions frozen. The compare script re-runs as data
  lands.

## Kill criteria

- **K-a.** A calibration arm's peak lies more than 100 MiB from 10,097. Then the same operating point does not
  reproduce, or the composition misses a session-dependent part.
- **K-b.** The calibration arms' pooled prefill misses by more than 10%. Then the per-chunk composition does not
  transfer across sessions at its own training point.

## Also reported (descriptive)

- Whether the calibration arms side with PF1 (fit +3.2% above stage 1's 64K `8,8` point) or with the N6 replication
  null. This tells whether that point was a low draw or a real `8,8` advantage.
- The item-to-item spread of prefill time at fixed P: the session's own noise floor.

## Software and outputs

- `engines/analysis-env` (numpy 2.3.3), with CUDA_VISIBLE_DEVICES="", nice 19 and 8 threads.
- Code: `tools/ideas/four-resources-formal/residency_prefill.py` (`predict` reads stage 1 only and writes the frozen
  predictions; `compare` reads stage 2 and 3).
- Results: `testsuite/results/ideas/four-resources-formal/`.
