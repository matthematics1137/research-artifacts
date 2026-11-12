# Paper 8, W7: letting exllamav3's int8 GEMV run on the 2080 Ti (2026-10-30)

## The gate

exllamav3 1.6.5 has an int8 decode path for EXL3 matrices (`exl3_gemv_int8.cu`). It quantizes the activation to
int8 and evaluates codebook(x) times the activation in one `dp4a` instruction, which sm_75 has. Upstream measured
it at 29% less time per call than the fp16 tensor-core kernel at K = 2 on a 3090. The W1 audit found its kernels
compiled for sm_75 (72 per-slice-scale, 44 cooperative instances). At run time, though, one line declines every
call on a device with less than 80 KB of opt-in shared memory:

```cpp
if (DevCtx::instance().get_smem_max(device) < 80 * 1024) return false;   // exl3_gemv_int8.cu:279
```

The 2080 Ti has 64 KB, so the 2080 Ti never takes this path. Upstream's comment gives the reason: the row caps
are sized for 80 KB, and the kernels recompute their work split from the same constants on the device, so the
host cannot ask for less.

## What the request actually is on this model

The request depends on the instance: bits per weight K, rows M, and plain or residual mode. The arithmetic below is
from the 1.6.5 source (`gemv_int8_sq_rows_max`, `smem_for`, `gemv_int8_stage_bytes`); static shared memory is from
the W1 dump.

| path | instance | dynamic request | static | total | fits 64 KB |
|---|---|---|---|---|---|
| per-slice-scale | M = 1, plain, K = 2 (this model) | 512 x 32 + 512 x 64 + 0 + 1,024 = 50,176 B | 144 B | 50,320 B | yes |
| per-slice-scale | M = 1, plain, K = 3 or 5 (staged) | 56,320 / 60,416 B | 144 B | | yes |
| per-slice-scale | M = 2, plain, K = 2 (verify rows) | 83,968 B | 144 B | | no |
| per-slice-scale | M = 1, residual | 82,944 B or more | 144 B | | no |
| cooperative | any K, as upstream requests it | 49,152 + 16,384 (the K = 8 stage) = 65,536 B | 656 B | 66,192 B | no |
| cooperative | K = 2, its own stage | 49,152 + 0 B | 656 B | 49,808 B | yes |

So the single-row decode call on this 2.0 bpw model needs about 49 KB, which a 64 KB device can hold. Nothing in
the kernel has to change: only the host-side gate.

**Prior art (checked 2026-10-30 against the local mirrors, `engines/src-mirror`, fetched 03:26 PDT).**
- **Upstream.** exllamav3's `dev` branch, at f8807350 from 2026-10-29, still declines every int8 GEMV launch under
  80 KB.
- **Forks.** So does every branch of the Turing forks we mirror that contains the file: rafatxf (including its
  `turing` branch), rluisr (`sm75`, `sm75-dev`, `sm75-upstream`), Pakobbix (`sm75-prefill-tiles`) and Matt's own fork.
- **Unfusing.** rafatxf's fork does report unfusing among its Turing speedups (on a 22 GB card), so G1 has prior art.
  The int8 fit does not, as far as these sources show.

## The patch (`tools/turing/patches/exllamav3-1.6.5-w7-int8-gemv-fit.patch`, host code only)

- **The blanket decline goes.** `gemv_int8_smem_fits(fn, request, device)` checks each instance's own largest
  request plus its static shared memory (`cudaFuncGetAttributes`) against `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
  It runs once per kernel and device, and the result is cached. An instance that does not fit is declined and
  falls through, exactly as upstream declines: per-slice-scale, then cooperative, then the regular fp16 kernel.
- **The cooperative path requests its own stage size** instead of the K = 8 maximum.
- **Unchanged:** the kernels' constants, so the host and device work split stays in sync. The request is never
  larger than what the device code uses; it is the same upper bound over shapes, computed per instance.
- **`EXL3_INT8_GEMV_FIT=0`** restores the upstream decline in the same build, so one binary serves both arms.

**Build** (`tools/turing/build_exl3.sh`): sm_75 only, on the desktop under a capped unit.
- The source is v1.6.5 (commit cdc785db, tarball sha256 07b70fc1…) plus this patch, labelled `1.6.5+w7`.
- The toolchain is a lab copy of CUDA 12.5 with the six glibc 2.43 `noexcept` declarations added, using gcc-13.
- The build is unpacked as an overlay (`$L/overlay/exl3-w7`) that `tools/turing/with_engine.sh` puts in front of
  the stock wheel, after checking that Python imports it.
- The stock wheel and the lab venv are untouched.

## Pre-registered before any measurement (2026-10-30 13:05 EDT; the build is still compiling)

The arms all use the census procedure (`tools/turing/census.sh`: 3 prompts x 256 decoded tokens, undrafted and MTP,
16K Q8 cache), one after another on the 2080 Ti:

| arm | engine | switches |
|---|---|---|
| C0 | stock 1.6.5 wheel | none (step `census-w2`) |
| C1 | 1.6.5+w7 | `EXL3_INT8_GEMV_FIT=0` (upstream gate, our toolchain) |
| C2 | 1.6.5+w7 | none (int8 plain wherever it fits) |
| C3 | 1.6.5+w7 | `EXL3_INT8_GEMV=1` (residual mode; at M = 1 only the cooperative kernel fits) |
| C4 | stock 1.6.5 wheel | `EXL3_INT8_GEMV=0` (no int8; the fused MGEMM thresholds go to 0, so gate/up stay fused) |

- **T1 (toolchain).** C1 undrafted ms per token is within ±3% of C0. A larger gap is the compiler's effect and is
  reported as such. All W7 comparisons then use C1, not C0, as the baseline.
- **T2 (the gate).** C2 undrafted ms per token is at least 5% lower than C1. A smaller gain, or a loss, means the
  int8 path does not pay on Turing at this K. The likely cause would be one 49 KB block per SM, which leaves 8 warps
  to hide memory latency. The MTP arm is reported, not predicted: verification rows (M = 2..7) still use the fp16
  kernel.
- **T3 (numerics, before any quality run).** `numerics.py` under the arm that wins T2 is checked against the plan's
  E2 criterion: `default||ref` KL mean at most 2e-4 and top-1 agreement at least 99% on the E47 item set. If plain
  mode fails it, C3 (residual) is tried under the same criterion.
- **T4 (quality).** The adopted arm runs GSM8K-500, undrafted and MTP. It is adopted only if no McNemar test
  against the stock lanes is significant and the accuracy delta is at least -1.0 point.
- **W6 side question (C4 vs C0).** On the stock wheel, int8 enabled changes how the MGEMM bundles are fused
  (`config.use_mgemm`) even though the int8 kernels never run on Turing. C4 measures whether switching int8 off
  changes decode speed on stock 1.6.5. No direction is predicted.

## W6: dispatch switches on the stock wheel (pre-registered 2026-10-30 13:35 EDT, after the undrafted census, before any arm ran)

**What the undrafted census showed** (`testsuite/results/turing/census/w2`, stock wheel, 45.6 ms per token under
nsys):
- The GPU is busy 96% of the time.
- EXL3 matrix kernels take 92% of kernel time.
- Reading one token's kernel sequence against each tensor's bitrate (`tools/turing/layer_bits.py`; this
  "2.0 bpw" quant mixes K = 1, 2, 3 and 4 per tensor) gives the dispatch below. Per-byte rates are the matrix's
  bytes over its kernel time.

| matrix (K) | kernel on the 2080 Ti | time | rate |
|---|---|---|---|
| MLP gate/up, 5120 x 17408 (K = 2) | `exl3_gemv`, wide config (136 x 256) | ~80 us each | ~279 GB/s |
| MLP down, 17408 x 5120 (K = 2) | `exl3_gemv`, narrow config (68 x 512: one block per SM) | ~125 us | ~178 GB/s |
| GDN out_proj, 6144 x 5120 (K = 2) | `exl3_gemv`, narrow | ~50 us | ~157 GB/s |
| GDN qkv + z, sliced bundle (K = 2, slice width 2048) | `exl3_mgemm` (8 x 1 x 8) | ~171 us | ~123 GB/s |
| any K = 1 matrix (gate/up in 4 layers, two q_proj) | `exl3_gemm` (tensor-core, 68 x 512) | ~151 us for 11 MB | ~74 GB/s |
| K = 3 matrices (9 out_proj, 6 o_proj, 3 down, layer-63 gate/up) | `exl3_gemm` | e.g. out_proj ~67 us | ~176 GB/s |

The source (`exl3_gemv.cu`, `config.py`) explains each row:
- **No K = 1 GEMV.** GEMV instances exist for K = 2-4 only, so K = 1 falls back to the GEMM.
- **K = 3 gets no GEMV on Turing.** It gets GEMV only on Ada, or when the narrow grid fits one wave.
- **Narrow config for n ≤ 8,192.** It is chosen for every K = 2 matrix with n at or under 8,192. On this card it
  runs one 512-thread block per SM.
- **qkv + z stay fused.** They are fused whenever the slice width is under `EXL3_MGEMM_N_THRESHOLD` (8,192).

**Arms.** All run undrafted census captures only (`CENSUS_CFGS=nodraft`) on the stock wheel; G0 is `census-w2`.

| arm | switches | what changes |
|---|---|---|
| G1 | `EXL3_MGEMM_N_THRESHOLD=0` | GDN qkv + z unfused into two GEMVs (qkv wide, z narrow) |
| G2 | `EXL3_GEMV=4` | every eligible K = 2-4 matrix at M ≤ 8 takes the wide config, including the K = 3 matrices now on the GEMM |
| G3 | `EXL3_GEMV=2` | K = 2-4 eligible, narrow for n ≤ 8,192 and wide above (K = 3 moves to GEMV) |
| G4 | G1 + G2 | both |
| C4 | `EXL3_INT8_GEMV=0` | MGEMM thresholds 0: everything that can fuse is fused, including MLP gate/up (above) |

Predictions, from the rates in the table, in undrafted ms per token against G0 under the same capture:
- **G1** saves at least 2 ms (estimate 3.6: 48 bundles at ~171 us become ~95 us).
- **G2** saves at least 2 ms (down_proj alone: ~60 layers x ~45 us). The K = 3 matrices are an open question.
- **G4** saves at least 4 ms (9%). If G1 and G2 each save but G4 saves less than the larger of them, the two
  interact (for example through occupancy) and that is reported.
- **G3:** no prediction.
- **Numerics.** Any arm adopted goes through `numerics.py` (`default||ref` KL mean at most 2e-4, top-1 agreement
  at least 99%), then GSM8K-500 (T4's rule), before it is called a speedup.

### W6 results (2026-10-30 13:10-13:22 EDT, one undrafted capture each; `testsuite/results/turing/census/compare_w6.md`)

| arm | ms per token (code / math / prose) | mean | change | prediction |
|---|---|---|---|---|
| G0 stock | 45.22 / 44.18 / 47.35 | 45.58 | | |
| G1 `EXL3_MGEMM_N_THRESHOLD=0` | 44.44 / 40.16 / 42.90 | 42.50 | **-6.8%** (3.1 ms) | ≥ 2 ms: **holds** |
| G2 `EXL3_GEMV=4` | 44.97 / 45.96 / 41.31 | 44.08 | -3.3% (1.5 ms) | ≥ 2 ms: **fails** |
| G3 `EXL3_GEMV=2` | 45.86 / 46.13 / 41.89 | 44.63 | -2.1% | none |
| G4 G1 + G2 | 42.53 / 42.48 / 42.85 | 42.62 | -6.5% | ≥ 4 ms: **fails**; no gain over G1 alone |
| C4 `EXL3_INT8_GEMV=0` | 53.22 / 58.13 / 57.86 | 56.40 | **+23.7%** | none |

- **G1.** It removes both fused bundles, the GDN qkv + z (8 x 1 x 8) and attention k/v (34 x 1 x 2). Their work moves
  to GEMVs (wide for qkv, narrow for z, a 32-block GEMV for k and v).
- **G2.** The K = 3 matrices leave the tensor-core GEMM (6.9 → 2.0 ms per token). But the formerly narrow K = 2
  matrices at n = 5,120 get only 80 wide blocks (n / 64) on 68 SMs, which is no better than one narrow block per SM.
  So the net gain is small, and it vanishes on top of G1.
- **C4.** It answers the side question: fusing everything is much worse on Turing. MLP gate/up as one MGEMM
  (34 x 1 x 2) costs 24.1 ms per token against about 11.8 ms as two wide GEMVs. The `exl3_mgemm` kernel (204-221
  registers, 64 KB shared memory, one block per SM) is the Turing problem, and unfusing is the right direction.
- **Noise.** One capture per arm, and per-prompt differences run to ±10%: G0's prose capture is the slowest of all.
  G1 is therefore confirmed by interleaved repeats (`tools/turing/ab_speed.sh`) before adoption, and goes through the
  numerics and quality rules above.

### W7 + G1 (pre-registered 2026-10-30 13:30 EDT, before any W7 arm ran)

The W7 build is ready: `1.6.5+w7`, 169 sm_75 cubins, overlay `$L/overlay/exl3-w7`. Two things follow from the
census:
- **Coverage.** The int8 per-slice-scale kernel has K = 1 instances, so W7 also replaces the K = 1 GEMM fallback
  (about 74 GB/s) at M = 1. It covers K = 3 as well (per-arch cap 5).
- **Fusing.** With int8 on, the qkv + z bundle still fuses (slice width 2,048 < 8,192), so the best W7 arm is
  expected to need G1 as well.

| arm | switches |
|---|---|
| C2g1 | `1.6.5+w7` + `EXL3_MGEMM_N_THRESHOLD=0` |

- **T5.** C2g1 undrafted ms per token is at least 5% below G1's 42.50.
- **T2, T3 and T4** apply unchanged.
- **Fail-fast.** The W7 steps are labelled `x-…`, which `full_queue.sh` now gives 2 tries instead of 30.

### W7 results (2026-10-30 13:36-13:44 EDT, one undrafted capture each; `testsuite/results/turing/census/compare_w7.md`)

| arm | engine, switches | ms per token (code / math / prose) | mean | vs G0 stock | test |
|---|---|---|---|---|---|
| G0 | stock 1.6.5 | 45.22 / 44.18 / 47.35 | 45.58 | | |
| C1 | 1.6.5+w7, `EXL3_INT8_GEMV_FIT=0` | 49.49 / 44.43 / 48.11 | 47.34 | +3.9% | T1 (±3%): **fails narrowly** |
| C2 | 1.6.5+w7 | 35.21 / 35.03 / 35.18 | 35.14 | **-22.9%** | T2 (≥5% below C1): **holds**, -25.8% |
| C2g1 | 1.6.5+w7, `EXL3_MGEMM_N_THRESHOLD=0` | 29.96 / 25.48 / 27.90 | **27.78** | **-39.1%** | T5 (≥5% below G1's 42.50): **holds**, -34.6% |
| C3 | 1.6.5+w7, `EXL3_INT8_GEMV=1` (residual) | 42.40 / 42.16 / 42.63 | 42.40 | -7.0% | |

- **C2.** `exl3_gemv_int8_sq_kernel` replaces every GEMV and the GEMM fallback, at 273 launches per token, 15.7 ms,
  about 57 us each. The output head is a single 68-block launch of 2.5 ms (455 MiB at about 188 GB/s). The fused
  qkv + z MGEMM is untouched and is then the largest single item (10.6 ms per token).
- **C2g1.** Unfusing moves that bundle to int8 GEMVs too. 27.78 ms per token is about 36 tok/s undrafted under nsys,
  against 21.9 for the stock wheel: 1.64x the throughput. It is about 239 GB/s of weights, 39% of the card's
  616 GB/s (stock: 27%).
- **C3.** Residual mode's per-slice-scale instance needs about 83 KB, so M = 1 goes to the cooperative kernel, which
  is much slower here.
- **T1.** The same sources built with the lab toolchain (CUDA 12.5 + gcc-13), with the patch switched off, are 3.9%
  slower than the official wheel (CUDA 12.8) in one capture. That is just outside the ±3% band, so the W7 effects
  are stated against C1, not C0. A repeat decides whether the gap is the compiler or noise.
- **Not yet a speedup.** Plain int8 mode quantizes activations (upstream: about 0.9% output RMS deviation), so T3
  (numerics) and T4 (GSM8K-500) must pass first. Both are queued: `numerics-w4b` on the stock wheel, then
  `x-numerics-w7g1` on the same token streams, which gives KL(W7 + unfuse || stock) at every position as well as
  each run's KL against the fp32 oracle.

## W8 alongside: MTP draft settings (pre-registered 2026-10-30 13:20 EDT, before the sweep ran)

`tools/turing/draft_sweep.py` (queue step `draft-sweep`) loads the model once on the stock wheel and measures each
setting on the three smoke prompts after a warm-up. The `base` setting is repeated four times through the grid to
track drift.

- **Sparse draft head (`-sdh N`).** Upstream drafts through a dense copy of the N most frequent unembedding columns,
  and verification keeps the full head, so the output distribution cannot change. The full head is 248,320 columns.
  Reading it once per drafted token costs bandwidth this card lacks.
  - **D1.** With sdh 16,384, MTP decode tok/s is at least 5% above `base` (mean of the base repeats).
  - **D1.** Draft acceptance falls by no more than 0.03.
- **Draft confidence and length.** No prediction. The best setting by mean decode tok/s over the three prompts is
  reported, beside the spread of the `base` repeats. A setting counts as better only if its gain exceeds that
  spread.
- **Output.** Greedy replies are hashed. Settings other than sdh can change the verify batch shape, and with it the
  bits (section 1 of `TURING_QUALITY_2026-10-30.md`), so hash agreement is reported, not required.

### W8 results (2026-10-30 13:22-13:36 EDT, stock wheel; `testsuite/results/turing/draft_sweep/ds1.jsonl`, tag ds1-20261030T132218)

- **The drift is as large as the effects.** The four `base` repeats ran at 46.0, 41.6, 39.4 and 41.7 tok/s (mean of
  three prompts): a spread of 6.6 tok/s (15%), falling as the card heats from the start of the grid. Undrafted ran
  at 21.5 and 21.7 tok/s at the two ends.
- **D1 holds by its letter, and that is weak evidence.** sdh 16,384 ran at 47.4 tok/s, +12.4% over the base mean.
  Its acceptance was 0.551 against 0.552. But all three sdh settings ran early, on a cooler card, and against the
  adjacent `base` (46.0) the gain is +3%. It goes to the interleaved repeats before any claim.
- **Draft confidence and length.** No setting exceeds the base spread (best: dc 0.2 at +7.9%, ndt 8 at +6.2%), so
  the lane defaults stay.
- **MTP output is not reproducible across generator instances.** No two settings matched each other's text, not
  even two `base` repeats (0/3 hashes equal). Dynamic draft lengths and the online confidence calibrator change
  verify shapes from run to run, and with them the greedy path. Undrafted greedy on 1.6.5 is reproducible within
  a run; the numerics step measures that floor.
- **Method change.** From here on, speed claims come from `ab_speed.sh`: fresh processes, arm order rotated every
  round, three rounds, so heat spreads over all arms.

## Interleaved confirmation (pre-registered 2026-10-30 13:55 EDT, before it ran)

Step `x-ab1`: `ab_speed.sh ab1 3` with five arms: stock undrafted, G1, W7 + G1 undrafted, stock MTP and W7 + G1
MTP (MTP at the lane setting: 6, dynamic). Before it, step `x-census-c2g1-mtp` captures W7 + G1 under MTP with
nsys.

- **A1.** W7 + G1 undrafted is at least 25% faster (decode tok/s, mean of three prompts) than stock undrafted in
  every round.
- **A2.** G1 alone is faster than stock in at least 2 of 3 rounds.
- **A3.** W7 + G1 under MTP versus stock MTP: no prediction. Verification runs 2-7 rows, and the int8 path takes
  only M ≤ 2, so the MTP gain comes mostly from unfusing and the drafted single-row forwards.

## T3 baseline: stock numerics (2026-10-30 13:45-15:19 EDT, step `numerics-w4b`; `testsuite/results/turing/numerics/w4b-e47-n50.summary.json`)

Stock 1.6.5 on the 2080 Ti: 50 E47 items (25 GSM8K, 25 MMLU), 12,239 response positions, greedy streams generated by
the `default` mode. Each mode is scored on the same streams; `ref` is the fp32 same-card oracle.

| pair | KL mean | KL p99 | KL max | top-1 agreement | flips | flips at ref margin > 0.1 |
|---|---|---|---|---|---|---|
| default ‖ ref (served dispatch vs oracle) | 7.26e-6 | 1.12e-4 | 8.8e-4 | 99.935% | 8 | 0 |
| unfused ‖ ref (EXL3 linears alone, E2) | 7.42e-6 | 1.15e-4 | 9.4e-4 | 99.943% | 7 | 0 |
| default ‖ unfused (fused blocks alone) | 7.01e-6 | 1.16e-4 | 4.7e-4 | 99.975% | 3 | 0 |
| default ‖ default_repeat (run-to-run floor) | 0 | 0 | 0 | 100% | 0 | 0 |

- **E2 and the end-to-end check are met** with about 27x margin on the KL rule (2e-4). Every top-1 flip sits where
  the oracle's own top-two margin is under 0.1 nat.
- **Undrafted decode is bit-reproducible** within a process on the stock wheel (the repeat floor is exactly 0).
- **Timing under the harness:** 45.1 ms per position in `default`, 42.7 in `unfused`. The direction matches G1.
- **What W7 + G1 must show (T3, unchanged):** `default‖ref` KL mean ≤ 2e-4 and top-1 ≥ 99% under the W7 overlay with
  `EXL3_MGEMM_N_THRESHOLD=0`. Step `x-numerics-w7g1` runs on these streams (`--streams`), so its `--xref` against
  this run also gives KL(W7 + G1 ‖ stock) at every position. Reported beside T3, not gated: the ratio of W7 + G1's
  `default‖ref` KL to the stock 7.26e-6. Plain int8 activations should raise it, and by how much is the cost of the
  speedup.


## W9: the int8 kernel's work split on a 64 KB part (pre-registered 2026-10-30 15:35 EDT, before the build or any W9 run)

**What the W7 + G1 census shows** (`census/c2g1-w7-mgemmn0`, token 100 in `token100_sequence_nodraft.txt`). Weight
bytes over kernel time per matrix:
- MLP gate/up (K = 2) run at about 409 GB/s (54.5 us each);
- MLP down at about 330 GB/s (66-70 us) for the same bytes;
- GDN qkv + z at about 300 GB/s;
- the 248K-column output head at about 196 GB/s (2.43 ms, 9% of the token) on a 68-block grid, against 272 everywhere
  else.

**Why.** The m = 1 int8 kernel (`exl3_gemv_int8_sq_kernel`) derives its slice height (`rows_per`) from its own grid
size. The host sizes that grid from the occupancy of a guessed slice height. On the 2080 Ti two things follow:
- **The output head gets one slice of all 320 rows.** Its activation staging is then about 37 KB of shared memory per
  block, so only one 8-warp block fits on each SM.
- **Two shapes round past one wave.** MLP down (17408 x 5120) and GDN qkv (5120 x 10240) round `rows_per` and
  `ksplit` up separately and land at 280 units on 272 blocks. Eight blocks then run a second, nearly empty wave.

`tools/turing/sq_fit_sim.py` mirrors both rules with sm_75's occupancy limits (4 blocks of 256 threads, 64 KB per
SM). Its kernel-rule column reproduces the grids measured in the census.

**The patch** (`tools/turing/patches/exllamav3-1.6.5-w9-int8-sq-fit.patch`, applied after W7; build `1.6.5+w9`):
- **The host chooses `rows_per` and passes it to the kernel** as a new trailing argument. 0 keeps the kernel's own
  rule, so host and kernel cannot disagree. CUDA-graph updates patch only the pointer arguments they recorded.
- **How it chooses.** Each `ksplit` is scored with the smallest 8-row multiple that gives it. The score is the
  per-block critical path at the grid that slice height's shared memory allows: waves x slice weight bytes, plus the
  epilogue's partial reads, times max(blocks per SM, 4). The lowest score wins.
- **Switches.** On by default where opt-in shared memory is under 80 KB. `EXL3_INT8_SQ_FIT=0` restores the kernel's
  rule in the same build. `EXL3_INT8_SQ_FIT_LOG=1` prints each shape's choice once.
- **Numerics are expected to move.** Activation scales are per slice, so they follow the slice height: five slices
  instead of one for the head (finer), and slightly taller slices for down and qkv (coarser).

Predicted choices at M = 1 (simulation, sm_75, 68 SMs):

| matrix | k x n | K | per token | kernel rule rows/ksplit/grid (units, waves) | W9 host rows/ksplit/grid (units, waves) | changed |
|---|---|---|---|---|---|---|
| GDN qkv | 5120 x 10240 | 2 | 47 | 48/7/272 (280, 1.03) | 56/6/272 (240, 0.88) | yes |
| GDN qkv | 5120 x 10240 | 1 | 1 | 48/7/272 (280, 1.03) | 56/6/272 (240, 0.88) | yes |
| GDN z | 5120 x 6144 | 2 | 48 | 32/10/272 (240, 0.88) | 32/10/272 (240, 0.88) |  |
| GDN out_proj | 6144 x 5120 | 2 | 39 | 32/12/272 (240, 0.88) | 32/12/272 (240, 0.88) |  |
| GDN out_proj | 6144 x 5120 | 3 | 9 | 32/12/272 (240, 0.88) | 32/12/272 (240, 0.88) |  |
| MLP gate/up | 5120 x 17408 | 2 | 118 | 80/4/272 (272, 1.00) | 80/4/272 (272, 1.00) |  |
| MLP gate/up | 5120 x 17408 | 1 | 8 | 80/4/272 (272, 1.00) | 80/4/272 (272, 1.00) |  |
| MLP gate/up | 5120 x 17408 | 3 | 2 | 80/4/272 (272, 1.00) | 80/4/272 (272, 1.00) |  |
| MLP down | 17408 x 5120 | 2 | 61 | 80/14/272 (280, 1.03) | 88/13/272 (260, 0.96) | yes |
| MLP down | 17408 x 5120 | 3 | 3 | 80/14/272 (280, 1.03) | 88/13/272 (260, 0.96) | yes |
| attn q | 5120 x 12288 | 2 | 14 | 64/5/272 (240, 0.88) | 64/5/272 (240, 0.88) |  |
| attn q | 5120 x 12288 | 1 | 2 | 64/5/272 (240, 0.88) | 64/5/272 (240, 0.88) |  |
| attn k/v | 5120 x 1024 | 3 | 16 | 16/20/272 (80, 0.29) | 16/20/272 (80, 0.29) |  |
| attn k/v | 5120 x 1024 | 4 | 14 | 16/20/272 (80, 0.29) | 16/20/272 (80, 0.29) |  |
| attn o | 6144 x 5120 | 2 | 10 | 32/12/272 (240, 0.88) | 32/12/272 (240, 0.88) |  |
| attn o | 6144 x 5120 | 3 | 6 | 32/12/272 (240, 0.88) | 32/12/272 (240, 0.88) |  |
| lm_head | 5120 x 248320 | 3 | 1 | 320/1/68 (970, 14.26) | 64/5/272 (4850, 17.83) | yes |

**Arms.** One undrafted nsys census each, on the `1.6.5+w9` build with G1 (`EXL3_MGEMM_N_THRESHOLD=0`):

| arm | switches |
|---|---|
| F0 | `EXL3_INT8_SQ_FIT=0` (the W7 + G1 decomposition in the new build) |
| F1 | default (fit on), with `EXL3_INT8_SQ_FIT_LOG=1` |

**Predictions:**
- **S1 (decomposition).** F1's log changes exactly the three shapes the table marks: the head (320/1/68 → 64/5/272),
  MLP down (80/14 → 88/13) and GDN qkv (48/7 → 56/6). No other shape changes.
- **S2 (head).** The output-head launch takes at least 30% less time in F1 than in F0.
- **S3 (total).** F1 undrafted ms per token is at least 3% below F0. The estimate is about 7%: the head about 1.2 ms,
  down about 0.5 ms and qkv about 0.25 ms of F0's ~27.8.
- **S4 (build reproducibility).** F0 is within ±3% of C2g1 (27.78 ms), the same decomposition in the W7 build.
- **Reported, not predicted:** the mean launch time of MLP down and GDN qkv.

**Numerics (T3 for W9).** One `default`-only pass on the stock streams (`--modes default`, `--xref` against
`w4b-e47-n50`, full logits every 16th position) gives:
- the top-20 KL bound against the stock fp32 oracle at every position;
- the exact KL on the stride-16 sample (`numerics.py --compare`).

The criterion is T3's: KL mean ≤ 2e-4 and top-1 agreement ≥ 99% against the oracle. Reusing the stored oracle is
valid only if `x-numerics-w7g1` shows the oracle reproducible across builds: KL(stock ref ‖ W7 ref) mean ≤ 1e-6.
Otherwise W9 gets a full numerics run like W7's.

**Adoption** follows W7 + G1. It needs S3 confirmed by interleaved repeats (`ab_speed.sh`, F0 against F1, three
rounds, F1 faster in every round), T3, and then GSM8K-500 under the adopted build (T4's rule).

**Queue (rev 10, 2026-10-30 15:55 EDT, before any W9 run).** The build `1.6.5+w9` is ready: 169 sm_75 cubins, wheel
sha256 f0c4735d…, `testsuite/results/turing/builds/exl3-w9_build_summary.txt`. The W9 steps run after `x-numerics-w7g1`
in this order:
1. `x-census-f0-w9fit0` and `x-census-f1-w9` (F0, F1);
2. `x-numerics-w9g1` (default only);
3. then, after `x-ab1`, `x-ab2`: `ab_speed.sh` with three arms, three rounds: W7 + G1 on the W7 build, F0 and F1.

The W7 + G1 arm repeats S4 under interleaving: F0 within ±3% of it in the round means.

## T3 result: W7 + G1 numerics (2026-10-30 15:20-16:44 EDT, step `x-numerics-w7g1`; `testsuite/results/turing/numerics/w7g1-e47-n50.summary.json`)

The run used `1.6.5+w7` with `EXL3_MGEMM_N_THRESHOLD=0`, on the stock run's 50 streams (12,239 positions). The guard stopped
it once at item 20, when a Chrome on-device model took the GPU, and it resumed from its shards.

| pair | KL mean | KL p99 | KL max | top-1 agreement | flips | flips at margin > 0.1 / > 0.5 |
|---|---|---|---|---|---|---|
| default ‖ ref (T3) | **8.89e-5** | 1.38e-3 | 9.6e-3 | **99.796%** | 25 | 1 / 0 |
| unfused ‖ ref (E2) | 9.54e-5 | 1.35e-3 | 1.27e-2 | 99.804% | 24 | 2 / 0 |
| default ‖ unfused | 1.56e-4 | 2.30e-3 | 1.01e-2 | 99.632% | 45 | 6 / 0 |
| default ‖ default_repeat | 0 | 0 | 0 | 100% | 0 | 0 / 0 |
| stock default ‖ W7 + G1 default (cross-run, top-20 bound) | 9.03e-5 | 1.39e-3 | 1.02e-2 | 99.763% | 29 | |
| stock ref ‖ W7 ref (cross-run) | 1.9e-17 | | | 100% | 0 | |

- **T3 holds:** KL mean 8.9e-5 ≤ 2e-4, and top-1 agreement 99.80% ≥ 99%.
- **The cost is real, though.** Plain int8 activations raise the served path's KL to the oracle about 12x
  (7.3e-6 → 8.9e-5), and the p99 about 12x. Top-1 flips rise from 8 to 25. All but one sit where the oracle's own
  top-two margin is under 0.1, and none above 0.5. Whether that moves task accuracy is T4's question.
- **The oracle is reproducible across builds,** bit for bit to 1.9e-17. The stored stock oracle therefore serves
  W9's `default`-only numerics, as pre-registered.
- **Speed under the harness:** 26.1 ms per position in `default` against 45.1 on stock (-42%).

**T4 queued** (2026-10-30 16:50 EDT, before either run): GSM8K-500 on the E47 lanes under `TURING_LANE_ENV=
"PYTHONPATH=$L/overlay/exl3-w7 EXL3_MGEMM_N_THRESHOLD=0"`:
- MTP first, labelled `TURING-exl3-mtp-w7g1`;
- then undrafted, labelled `TURING-exl3-nodraft-w7g1`.

Each is compared with `compare_quality.py` against the stock lane of the same mode (`TURING-exl3-mtp`,
`TURING-exl3-nodraft`, both 97.6%). The rule is unchanged: a delta of at least -1.0 point, and no significant
McNemar test. The steps follow `x-ab2` and run ahead of the MMLU replica. If W9 is adopted, it gets its own GSM8K-500
afterwards.

### W9 results (2026-10-30 16:45-16:56 EDT; `census/f0-w9fit0-mgemmn0`, `census/f1-w9-mgemmn0`, `census/compare_w9.md`, `numerics/w9g1-e47-n50.summary.json`, `numerics/compare_exact_w9.txt`)

**S1 holds.** F1's log (`census/f1-w9-mgemmn0/sq_fit_log.txt`) changes exactly the three predicted shapes, at
every K they occur with:
- the head, 320/1/68 → 64/5/272;
- MLP down, 80/14/272 → 88/13/272;
- GDN qkv, 48/7/272 → 56/6/272.

No other shape changes.

**Whole-token census** (one undrafted capture each, ms per token, code / math / prose):

| arm | ms per token | mean | vs F0 |
|---|---|---|---|
| C2g1 (W7 build) | 29.96 / 25.48 / 27.90 | 27.78 | +7.9% |
| F0 (W9 build, fit off) | 25.79 / 25.58 / 25.90 | 25.75 | |
| F1 (W9 build, fit on) | 24.27 / 24.57 / 23.56 | **24.13** | **-6.3%** |

- **S3 holds:** F1 is 6.3% below F0, against the ≥ 3% required.
- **S4 fails by its letter:** F0 is 7.3% below C2g1, outside ±3%. The kernel-level medians below show that F0 and C2g1
  run the same decomposition at the same speed. The whole-token gap comes from C2g1's slow code capture (29.96) and its
  head, which is latency-bound and clock-sensitive. Single captures on this card vary by up to ±8%, so x-ab2 settles
  this.

**Per-matrix medians over all captured tokens** (`tools/turing/census_head.py`, `census_shapes.py`; medians, because
display time-slicing on this desktop GPU stretches a few launches):

| matrix | C2g1 | F0 | F1 | F1 vs F0 | F1 rate |
|---|---|---|---|---|---|
| output head (5120 x 248320, K = 3, 477 MB) | 2194-2530 us | 2151-2165 us | **989-992 us** | **-54%** | ~482 GB/s (78% of peak) |
| MLP down (17408 x 5120, K = 2) | 67.7-69.0 us | 67.6-67.8 us | 60.2-60.7 us | -11% | ~370 GB/s |
| GDN qkv + z (two launches) | 69.7-71.3 us | 70.0-70.2 us | 62.5-63.0 us | -11% | |
| MLP gate + up (unchanged split) | 105.8-108.8 us | 107.4-107.9 us | 109.2-110.3 us | +1.5% | ~409 GB/s each |

- **S2 holds:** the head takes 54% less time, against the ≥ 30% required.
- **The pre-registered estimates held per matrix:** head about 1.2 ms (measured 1.16), down about 0.5 (0.48), qkv
  about 0.25 (0.36).
- **The other int8 time per token is unchanged** (about 18.3 ms), so the whole gain is these three matrices.

**Numerics (T3 for W9).** The stored oracle holds, since the fp32 oracle is bit-identical across builds:

| against the stock fp32 oracle | top-1 agreement | top-20 KL bound, all positions | exact KL, 768 sampled positions (mean / p99 / max) |
|---|---|---|---|
| W9 + G1 default | **99.869%** (16 flips) | 8.35e-5 | **8.69e-5** / 1.33e-3 / 6.8e-3 |
| W7 + G1 default | 99.796% (25 flips) | 8.86e-5 | 1.02e-4 / 1.63e-3 / 4.8e-3 |
| stock default | 99.935% (8 flips) | 7.24e-6 | 7.15e-6 / 1.19e-4 / 2.3e-4 |

- **T3 holds for W9:** exact KL mean 8.7e-5 ≤ 2e-4, and top-1 agreement 99.87% ≥ 99%.
- **W9's finer head slices** (five scales across the 5,120 inputs instead of one) leave it slightly closer to the
  oracle than W7.
- **W7 and W9 differ from each other by 1.5e-4.** They are two independent int8 noise draws around the oracle, not one
  path. Neither is bit-identical to stock.

**T4 amended (2026-10-30 17:05 EDT, before either GSM8K run started).** The two queued GSM8K-500 runs move from W7 + G1
to W9 + G1 (`PYTHONPATH=$L/overlay/exl3-w9`, fit on by default), labelled `TURING-exl3-mtp-w9g1` and
`TURING-exl3-nodraft-w9g1`.

Why: W9 contains W7. It passes T3 with a lower KL. Its gains are confirmed per matrix, not only per token. It is the
build that would be adopted, so the quality test should run on it.

The rule is unchanged: against `TURING-exl3-mtp` and `TURING-exl3-nodraft`, a delta of at least -1.0 point and no
significant McNemar test. If x-ab2 does not confirm S3 (F1 faster than F0 in every round), W9 is still not adopted
for speed, and T4's result then speaks for the int8 path in general. W7 alone gets its own run only if W9 is set
aside.

### W7 + G1 under MTP (2026-10-30 16:56 EDT, step `x-census-c2g1-mtp`; `census/compare_w7_mtp.md`)

This is one nsys capture at the lane setting (6 draft tokens, dynamic), against stock MTP (`census/w2`):

| arm | ms per token (code / math / prose) | mean | acceptance (code / math / prose) |
|---|---|---|---|
| stock MTP | 21.39 / 20.43 / 37.02 | 26.28 | 0.657 / 0.700 / 0.318 |
| W7 + G1 MTP | 18.33 / 18.66 / 29.36 | 22.12 (**-15.8%**) | 0.651 / 0.668 / 0.374 |

- **Where the gain comes from.** It comes from unfusing and from the drafted single-row forwards, as A3's note
  expected:
  - the `exl3_mgemm` bundles (4.5 ms per token) are gone;
  - the K = 1 and K = 3 tensor-core GEMM falls from 5.7 to 2.7 ms;
  - the M = 1 draft forwards and the head now run int8.
- **What remains.** The verify forwards (2-7 rows) stay on the fp16 `exl3_gemv` kernels, about 12 ms of the
  22 ms per token. The head still runs on 68 blocks at 2.5 ms per token, which W9 cuts by about half.
- **The next lever under MTP** is therefore the verify rows: int8 instances for M = 3-8 (upstream compiles M = 1-2
  only). That is W10. It is not pre-registered yet, and comes after T4.
- **Acceptance** moves with the greedy path (math 0.700 → 0.668, prose 0.318 → 0.374), as expected when int8 changes
  the bits. The interleaved A/B (x-ab1, A3) gives the speed claim. T4 gives the quality.

## W10a: plain int8 for 3-4 rows (MTP verification) (pre-registered 2026-10-30 17:25 EDT, before the build or any run)

**Why.** Under MTP, verification forwards run 2-7 rows. Upstream sends m ≤ 2 to the int8 kernel. Larger m goes to
the fp16 kernels, because on a 3090 they won there. On the 2080 Ti, the W7 + G1 MTP census shows those fp16
`exl3_gemv` launches at about 12 of 22 ms per token.

The int8 kernel is generic in M, and rows share the decoded weights. Upstream's own limits allow m ≤ 4:
- the sq epilogue runs a warp pair per row on 8 warps;
- the per-slice scale region holds ksplit x m ≤ 4 x 64.

The patch (`tools/turing/patches/exllamav3-1.6.5-w10-int8-sq-m4.patch`, after W7 and W9; build `1.6.5+w10`) does three
things:
- **Instances.** It compiles plain (non-residual) instances for M = 3 and 4, for every K and half-K. On sm_75 they
  compile to 64 registers, the same as M = 1-2, and nothing spills to local memory.
- **Switch.** `EXL3_INT8_SQ_MAX_M` (1-4, default 2, which is upstream's behaviour) sets the largest m.
- **Workspace bound.** The W9 slice search now skips any `ksplit` whose partials would overflow the fixed 16 MB
  workspace. That case arises for the head at m = 4.

**Arms.** One MTP census each (lane setting: `-mtp -ndt 6 -dds`), on `1.6.5+w10` with G1 and the W9 split on:

| arm | switches |
|---|---|
| H0 | `EXL3_INT8_SQ_MAX_M=2` (the W9 + G1 path) |
| H1 | `EXL3_INT8_SQ_MAX_M=4`, with `EXL3_INT8_SQ_FIT_LOG=1` |

**Predictions:**
- **V1.** H1's MTP ms per token (mean of the three prompts) is at least 4% below H0's.
- **V2.** H1 runs fewer fp16 `exl3_gemv` launches per token, and more int8 launches. Reported.
- **V3 (numerics, T3's rule).** A teacher-forced pass at four tokens per forward is scored against the stock fp32
  oracle: `numerics.py --step 4 --modes default` on the stock streams, under H1. The rule is KL mean ≤ 2e-4 (exact,
  on the stride-16 sample) and top-1 agreement ≥ 99%.
- **Undrafted is unchanged** (m = 1), so W10a does not enter undrafted claims.

**Adoption.** V1 must be confirmed by interleaved repeats (H0 against H1, three rounds), and V3 must hold. GSM8K-500
under MTP is then rerun if W10a is adopted, because it changes the bits of every verification forward.

**Queue (rev 13, 2026-10-30 17:30 EDT, before any T4 run).** T4's order changes so that MTP's GSM8K-500 runs once, on the
right build:
1. undrafted (W9 + G1) right after x-ab2;
2. then the W10a arms H0, H1 and the step-4 numerics;
3. then T4's MTP run.

The MTP run's build is W9 + G1 as queued, or W10a + G1 (`EXL3_INT8_SQ_MAX_M=4`) if V1 and V3 hold. Either choice is
recorded here before the run starts. W10a does not change one-row decoding, so the undrafted run serves both builds.

## Interleaved confirmation: results (2026-10-30 16:58-17:22 EDT; `testsuite/results/turing/ab/ab1.md`, `ab2.md`, raw rows beside them)

Method: fresh processes, three rounds, arm order rotated each round, no profiler. The figures are decode tok/s,
averaged over the three smoke prompts.

**x-ab1** (stock wheel and the W7 build):

| arm | round 1 | round 2 | round 3 | mean | ms per token |
|---|---|---|---|---|---|
| stock undrafted | 23.60 | 23.13 | 24.46 | 23.73 | 42.14 |
| G1 | 24.25 | 24.49 | 26.06 | 24.93 | 40.10 |
| W7 + G1 undrafted | 35.93 | 36.71 | 39.12 | **37.25** | 26.84 |
| stock MTP | 47.47 | 44.61 | 48.80 | 46.96 | 21.30 |
| W7 + G1 MTP | 47.86 | 47.34 | 52.18 | 49.12 | 20.36 |

- **A1 holds.** W7 + G1 undrafted is +52%, +59% and +60% over stock in the three rounds, against the ≥ 25% required.
  The mean is **+57% (1.57x)**.
- **A2 holds.** G1 alone is faster in 3 of 3 rounds, by +5.1% on average.
- **A3 (no prediction).** W7 + G1 under MTP is +4.6% on average (+0.8, +6.1, +6.9). That is far less than the one
  capture's -15.8% per token. The verification rows still run fp16, and acceptance varies from instance to instance.
  W10a targets exactly this.

**x-ab2** (the W7 build and the W9 build, all with G1):

| arm | round 1 | round 2 | round 3 | mean | ms per token |
|---|---|---|---|---|---|
| W7 + G1 | 40.25 | 40.73 | 40.71 | 40.56 | 24.65 |
| F0 (W9 build, fit off) | 39.82 | 37.96 | 40.79 | 39.52 | 25.30 |
| F1 (W9 + G1) | 42.98 | 43.66 | 44.04 | **43.56** | 22.96 |

- **S3 is confirmed.** F1 beats F0 in 3 of 3 rounds: +8.0%, +15.0%, +8.0% (mean +10.3%). Against W7 + G1, F1 is
  +7.4% (3 of 3).
- **S4 holds under interleaving.** F0 is within -2.6% of W7 + G1 on average, inside ±3%. The single-capture gap was
  noise.
- **Sessions drift.** The same W7 + G1 arm ran 37.25 tok/s in x-ab1 and 40.56 in x-ab2. Each comparison is therefore
  made only within its own session.
- **Chained estimate:** F1 is about 1.57 x 1.074 ≈ **1.69x** the stock wheel's undrafted decode. That is roughly
  40 tok/s against 24 at short context. The ratio multiplies across two sessions, so it is an estimate. A direct stock
  against F1 repeat goes into the depth runs.

**Status:**
- W9 + G1 meets its speed rules (S1-S3, confirmed) and T3.
- T4 (GSM8K-500, undrafted first) decides adoption, and is next in the queue.
- Under MTP the int8 path pays only a few percent so far. W10a, for the verify rows, is being built.

## D: decode at context depth (pre-registered 2026-10-30 17:30 EDT, before any depth run)

**Why.** On Turing, decode attention runs on Triton's FMA path; `tl.dot` has no tensor-core lowering on sm_75. Its
cost grows with context, while the int8 savings are per weight byte. Two figures to beat at depth:
- the smoke run's stock EXL3, 19.1 tok/s at 30K;
- llama.cpp's best undrafted, 21.2 tok/s at depth 32768 (UD-IQ2_XXS, E1).

**Tool.** `census.py --fill N` puts a fixed synthetic document of about N tokens ahead of each smoke prompt (fill 16384
gives 16,452 prompt tokens). Step `x-depth1` runs `ab_speed.sh depth1 3` with six arms at cache 32768, undrafted:
stock and F1 (W9 + G1), each at fill 0, 8192 and 28000. That is three rotated rounds, 18 processes.

**Predictions:**
- **D1.** At fill 28000, F1's decode tok/s is at least 1.3x stock's in every round.
- **D2.** At fill 28000, F1's mean over the rounds is at least 25 tok/s, above llama.cpp's 21.2 at 32K.
- **D3 (reported).** Each arm's ms per token at 28000 minus at 0 is its attention-and-cache cost at depth. Expected
  to be about the same for both arms, since W7 and W9 do not touch attention.
- **Fill 0** gives the direct stock against F1 ratio in one session. It replaces the chained 1.69x estimate.

**Queue (rev 15, 2026-10-30 17:45 EDT).** `x-depth1` now runs before T4's MTP GSM8K-500. That leaves its 45 minutes or so
to read W10a's census and step-4 numerics, and to record the MTP run's build here before it starts. The W10a build
`1.6.5+w10` is ready: `builds/exl3-w10_build_summary.txt`, wheel sha256 57bebd97…. The undrafted T4 run is at item 23 of
500 at about 40 tok/s.

## T4, undrafted: result (2026-10-30 18:45 EDT; `quality/compare_t4_nodraft_w9g1.md`, rows in `quality/TURING-exl3-nodraft-w9g1/`)

GSM8K-500 on the E47 undrafted lane under W9 + G1 (`TURING-exl3-nodraft-w9g1`), against the stock lane of the same mode:

| label | accuracy [95% CI] | mean tokens | wall per item | decode tok/s (median) |
|---|---|---|---|---|
| TURING-exl3-nodraft (stock) | 97.6% [95.9, 98.6] | 325.8 | 14.22 s | 23.6 |
| TURING-exl3-nodraft-w9g1 | 97.0% [95.1, 98.2] | 328.5 | 8.32 s | 40.9 |

- **T4 holds for the undrafted lane.** The delta is -0.60 points, inside the -1.0 bound. The three discordant items all
  go to stock (3 / 0), so the exact McNemar test gives p = 0.25, not significant. The final answers agree on 497 of
  500 items, and 69.2% of the outputs are byte-identical.
- **Speed in a real workload:** 1.71x less wall time per item, with a 1.73x median decode rate. That matches the
  chained 1.69x estimate. The stock MTP lane is still faster per item (6.45 s against 8.32 s).
- **One guard stop.** The desktop's GPU guard paused the run at item 398 (18:18 EDT, the Chrome on-device model) and the
  queue resumed it at 18:20 on a fresh server with the same lane environment. Both segments decode at the overlay's
  speed (median 39.2 and 40.4 tok/s, against stock's 23.6), so both ran W9 + G1. The 500 ids are unique.

## W10a: results, and the MTP run's build (2026-10-30 18:50 EDT; `census/compare_w10a.md`, `numerics/compare_w10m4s4.txt`)

| arm | code | math | prose | mean ms/token | int8 launches/token | fp16 `exl3_gemv` launches/token (136- and 68-block) |
|---|---|---|---|---|---|---|
| H0 (`EXL3_INT8_SQ_MAX_M=2`) | 17.21 | 17.12 | 28.08 | 20.80 | 10.31 | 71.17 + 63.33 |
| H1 (`EXL3_INT8_SQ_MAX_M=4`) | 16.30 | 15.61 | 28.04 | 19.98 | 10.08 | 68.42 + 60.90 |

- **V1 fails.** H1 is 3.9% below H0, short of the 4% required. The mechanism never ran at all. Every kernel family
  shrinks by the same ~4% per token: launches per token fall with acceptance, and the int8 share does not grow. The
  fit log (`EXL3_INT8_SQ_FIT_LOG=1`) lists 18 shapes, all at M = 1.
- **V2: no shift.** The int8 and fp16 launch counts move together.
- **V3 passes, but it tested the wrong path.** Exact KL is 7.08e-6 and top-1 agreement 99.95%: stock-level numerics,
  because the multi-row forwards ran the fp16 kernel. It says nothing about int8 at four rows.
- **Why nothing ran.** The sq kernel's shared layout is 32 + 64·M bytes per slice row, plus its stage and epilogue
  buffers. `gemv_int8_sq_rows_max()` caps the slice at 512 rows within an 80 KB budget. W7's fit check sizes each
  instance at that cap: about 49 KB at M = 1, but 80 KB or more at every M ≥ 2, beyond the 2080 Ti's 64 KB opt-in limit.
  So W7 declined every multi-row instance. On Turing no verify row has ever taken the int8 path, upstream's m = 2
  included. `sq_fit_sim.py --m 2 3 4` (extended for W10b below) shows "declines" for every shape.

**The MTP run's build: W9 + G1, as queued.** V1 did not hold, so under the rev 13 rule step `x-q-exl3-mtp-w9g1-gsm8k_500`
runs unchanged after `x-depth1`. It is recorded here before that run starts.

## W10b: a slice cap that fits 64 KB, so verify rows reach int8 (pre-registered 2026-10-30 19:00 EDT, before the build finished or any run)

**Change** (`tools/turing/patches/exllamav3-1.6.5-w10b-int8-sq-rowcap.patch`, sha256 c2bb2c32…, after W7, W9 and W10;
build `1.6.5+w10b`). Host only; the kernel is unchanged. Since W9, the host passes the slice height to the kernel,
which builds its shared layout from it. W10b uses that:
- **Cap.** With W9's host choice on (`EXL3_INT8_SQ_FIT`, default on below 80 KB), slices are capped per instance at
  the tallest multiple of 8 rows that fits the device's opt-in shared memory, after the instance's static shared
  memory, stage and epilogue buffers.
- **Sizing.** The fit check and the dynamic shared memory attribute use that cap instead of the 80 KB one. W9's slice
  search stays within it. The kernel-rule mirror that seeds the grid is clamped to it as well, so no occupancy query
  asks for more than the device holds.
- **Decline.** If the search finds no slice height, the launch is declined (fp16). The kernel's own rule would size
  for 80 KB.
- **Unchanged.** At M = 1 the cap equals the old one, 512 rows, for every instance up to K = 7. One-row launches and
  undrafted decoding are therefore unchanged. `EXL3_INT8_SQ_FIT=0` restores W10's behaviour.

The host file compiles for sm_75 with no new warnings. The simulation's caps on the 2080 Ti (68 SMs, 144 B static):

| M | cap, K = 1, 2, 4 | cap, K = 3 | host choices (full table: `sq_fit_sim.py --m 2 3 4`) |
|---|---|---|---|
| 2 | 392 rows | 352 rows | 16-88 rows, 4 blocks per SM throughout; head 40 rows x 8 slices |
| 3 | 272 | 248 | 16-80 rows, 2-4 blocks per SM; head 64 x 5 |
| 4 | 208 | 184 | 16-64 rows, 2-4 blocks per SM; head 160 x 2 (55 KB, 1 block per SM; W10's workspace bound allows at most 4 slices) |

**Arms.** MTP census (`-mtp -ndt 6 -dds`) on `1.6.5+w10b` with G1 and the W9 split on:

| arm | switches |
|---|---|
| J0 | `EXL3_INT8_SQ_MAX_M=1` (verify rows on fp16: W9 + G1's behaviour) |
| J2 | `EXL3_INT8_SQ_MAX_M=2` (upstream's m ≤ 2, for the first time on Turing), with the fit log |
| J4 | `EXL3_INT8_SQ_MAX_M=4`, with the fit log |
| J4n | J4's switches, undrafted |

**Predictions:**
- **R0 (mechanism, a precondition).** J4's fit log lists shapes at M = 2, 3 and 4, with exactly the host choices the
  simulation table gives. J4 runs more int8 and fewer fp16 `exl3_gemv` launches per token than J0. If R0 fails, R1-R3
  are not evaluated.
- **R1.** J4's MTP ms per token (mean of the three prompts) is at least 4% below J0's.
- **R1b (reported).** J2 against J0 and against J4.
- **R2 (numerics, T3's rule).** A teacher-forced pass at four tokens per forward under J4, scored against the stock fp32
  oracle (`numerics.py --step 4`): exact KL mean ≤ 2e-4 on the stride-16 sample, and top-1 agreement ≥ 99%. The same
  at two tokens per forward under J2 (`--step 2`).
- **R3 (one-row decoding unchanged).** J4n's 100-token undrafted sequence is identical to F1's (W9 + G1,
  `census/f1-w9-mgemmn0`).

**Adoption.** The faster of J2 and J4 must beat J0 in every round of interleaved MTP repeats (step `x-ab3`, three
rotated rounds). R2 must hold for it. Then GSM8K-500 under MTP is rerun on that setting, under T4's rule against
`TURING-exl3-mtp`.

**Queue (rev 16, 2026-10-30 19:00 EDT).** After T4's MTP run, ahead of the MMLU replica: `x-census-j0-w10b`, `-j2-`,
`-j4-`, `-j4n-`, `x-numerics-w10bm4s4`, `x-numerics-w10bm2s2`, then `x-ab3` (J0, J2, J4 under MTP, three rounds).

**Build.** `1.6.5+w10b` was built at 19:11 EDT: 169 sm_75 cubins, wheel sha256 a32219f2…, overlay `$L/overlay/exl3-w10b`.

## D: results (2026-10-30 18:42-19:08 EDT, step `x-depth1`; `ab/depth1.md`, raw rows `ab/depth1.jsonl`)

Undrafted decode tok/s, averaged over the three smoke prompts. Fresh processes, cache 32768, arm order rotated each
round, no profiler. F1 is W9 + G1.

| arm | prompt tokens | round 1 | round 2 | round 3 | mean | ms per token |
|---|---|---|---|---|---|---|
| stock, fill 0 | 31-50 | 24.68 | 23.57 | 24.59 | 24.28 | 41.18 |
| F1, fill 0 | 31-50 | 43.49 | 43.21 | 43.16 | **43.29** | 23.10 |
| stock, fill 8192 | 8242-8261 | 21.47 | 22.48 | 21.55 | 21.83 | 45.80 |
| F1, fill 8192 | 8242-8261 | 35.27 | 37.03 | 34.83 | 35.71 | 28.00 |
| stock, fill 28000 | 28050-28069 | 19.03 | 19.82 | 19.88 | 19.58 | 51.08 |
| F1, fill 28000 | 28050-28069 | 28.85 | 28.60 | 30.42 | **29.29** | 34.14 |

- **Direct ratio at short context: 1.78x.** F1 beats stock by +76.2%, +83.3% and +75.5% (mean +78.4%). That is above
  the 1.69x chained from two sessions, and it replaces it.
- **D1 holds.** At fill 28000, F1 beats stock by +51.6%, +44.3% and +53.0%: at least 1.3x in every round.
- **D2 holds.** At fill 28000, F1 averages 29.3 tok/s, against the 25 required. llama.cpp's best undrafted figure at
  depth 32768 is 21.2.
- **D3: the cost of depth is about the same.** From fill 0 to 28000, stock's ms per token grows by 9.90 and F1's by
  11.04. F1's growth is 1.1 ms (11%) more, though W7 and W9 do not touch attention. Neither W7 nor W9 makes depth
  cheaper, so the ratio shrinks with context: 1.78x, 1.64x, 1.50x. At 28K, about a third of F1's token time is the
  context's cost. Decode attention, which runs on Triton's FMA path on sm_75, is the next lever at depth.
- **Recording bug (rows only).** `census.py` wrote `prompt_tokens = 1` in every row: `encode` returns a (1, n)
  tensor, and the field took its length. The prompt lengths in the table were recounted on the desktop's CPU with
  the same tokenizer and filler. The measurement itself is unaffected. The field is fixed for rows from 19:15 EDT
  (`census.py`, sha256 60045dcb…), so the W10b arms record it correctly.

## W11 scouting: what decode pays at depth (diagnostic, recorded 2026-10-30 19:20 EDT before either capture)

**Floor.** The model has 16 full-attention layers (of 64). Each has 4 KV heads of dimension 256, so one cached token is
2,048 K and V elements per layer: 2 KB in the Q8 cache. At 28,050 tokens a decoded token reads about
16 x 28,050 x 2 KB ≈ 0.92 GB of cache. At the 2080 Ti's ~550 GB/s that takes about 1.7 ms, against the 10-11 ms
that depth costs (D3). The gap is roughly 6x. Two candidate causes:
- the Q8 pages are expanded to fp16 before attention every step;
- the Triton split kernel (FMA path on sm_75) runs far from bandwidth.

**Captures.** Two undrafted nsys censuses under F1 (W9 + G1), cache 32768, the same settings except depth:
- `x-census-d0-f1` at fill 0;
- `x-census-d28-f1` at fill 28000.

Their per-family difference attributes the ~11 ms. No prediction: this decides what W11 targets, and W11 is
pre-registered after it. Queue rev 17 puts them after `x-census-j4n-w10b`.

**Code reading (19:30 EDT).**
- Decode reads the Q8 pages directly ("online dequant"), so there is no fp16 staging pass. The whole cost is in
  the Triton split kernel (and its combine).
- The cached sm_75 PTX of `_paged_attn_decode_split_kernel` holds no `mma.sync` and 3,074-6,148 `fma.rn.f32`,
  although both of its dots are fp16 x fp16. Triton 3.6 has no tensor-core lowering for this target.
- Undrafted, a KV head's 6 query heads are packed into a 16-row program, the `tl.dot` minimum. On FMA units 10 of
  the 16 rows are wasted arithmetic.
- The grid is 4 heads x 34 splits = 136 programs of 4 warps. That is at most about 8 of an SM's 32 warp slots, and
  shared memory may cap it there anyway.

**Bench (step `x-attn-bench1`, queue rev 18, after the depth censuses).** `tools/turing/attn_bench.py` times the
split-decode call alone at context 512, 8192 and 28050 and query lengths 1 and 7. It sweeps the KV tile (16, 32,
64), warps (2, 4, 8), stages (1, 2) and splits (the rule, 68, 136, 272), checking each output against the default
launch's. This shows how much of the gap is the launch shape (free to change, Python only) and how much is the FMA
arithmetic (which needs a tensor-core kernel: the 1.6.5 tree already has one for Turing prefill, `turing/fa75.cu`).

## W11: fd75d, a tensor-core decode attention kernel for Turing (pre-registered 2026-10-30 20:05 EDT, before any timing of it)

Matt approved building it on 2026-10-30 ("go for the attention kernel"). This is recorded before the depth censuses and
the launch-shape bench have run, and before any fd75d timing.

**Kernel** (`tools/turing/fd75d/fd75d.cu`; kept in this repository until the paper is published). It is a drop-in
for phase 1 of the Triton split kernel when the cache is 8-bit K and V, head_dim 256, causal, with no window, softcap
or sinks: the same grid (programs x split cap), the same row packing and the same partial layout, so the Triton
combine finishes unchanged. Per 16-key tile each warp computes:
- S^T = K Q^T (keys x query rows) with `mma.m16n8k8`, so a 6-head group fills 6 of 8 columns instead of 6 of 16 rows;
- O^T += V^T P^T, with P^T transposed in-warp (`movmatrix`).

Fragments are built in registers straight from the packed cache. Each thread owns the dims it loads, which is
valid because the dot products are invariant under a consistent dim permutation. q is rotated with a fast
Walsh-Hadamard transform. Dequantization is one fp16 rounding of the exact value, as in the Triton loaders. A
program's 4 warps split its keys four ways and merge through shared memory. sm_75 build: 198 registers, no
spills, 25.6 KB shared memory, 2 blocks per SM.

**Test** (`tools/turing/fd75d_test.py`, standalone extension, CUDA graphs). Cases: batch 1 and 2, query lengths 1, 2,
5 and 7, context 37 to 28,050. Each compares fd75d + combine, and the engine's Triton path, against an fp64
reference attention over the dequantized cache.

**Predictions:**
- **C1 (correctness, gate).** In every case and split count, fd75d's output has no NaN. Its mean error against the
  fp64 reference is at most 1.5x the Triton path's + 1e-5, and its max error at most 2x Triton's + 1e-4.
- **K1.** At context 28,050, query length 1: fd75d (split + combine) is at least 3x faster than the Triton path.
- **K2.** At context 28,050, query length 7 (an MTP verify step): at least 4x faster.
- **K3 (reported).** At contexts 37 and 300 the two are expected within ±20%: both are launch- and combine-bound there.
- After integration into `BC_Attention` (patch W11, switch `EXL3_FD75D`, default on for sm_75 when the conditions
  hold):
  - **E1.** Undrafted decode at fill 28000 is at least 20% faster than F1 (W9 + G1) in every round of an
    interleaved repeat.
  - **E2.** At fill 0 it is within ±3% of F1.
  - **E3 (reported).** MTP at fill 28000.
  - Then T3's numerics rule and T4's GSM8K-500 rule before adoption.

## T4, MTP: result, and adoption (2026-10-30 20:00 EDT; `quality/compare_t4_mtp_w9g1.md`, rows in `quality/TURING-exl3-mtp-w9g1/`)

GSM8K-500 on the E47 MTP lane under W9 + G1 (`TURING-exl3-mtp-w9g1`, 19:09-19:57 EDT, no guard stop), against the stock
MTP lane:

| label | accuracy [95% CI] | mean tokens | wall per item | decode tok/s (median) | draft acceptance |
|---|---|---|---|---|---|
| TURING-exl3-mtp (stock) | 97.6% [95.9, 98.6] | 327.2 | 6.45 s | 53.4 | 0.622 |
| TURING-exl3-mtp-w9g1 | 97.4% [95.6, 98.5] | 324.7 | 5.62 s | 61.1 | 0.622 |

- **T4 holds under MTP.** The delta is -0.20 points, with 1 / 0 discordant items (McNemar p = 1.0). The final answers
  agree on 498 of 500 items, and 88.6% of the outputs are byte-identical.
- **Speed in a real workload:** 1.15x less wall time per item, and the median decode rate goes from 53.4 to 61.1 tok/s.
  That is more than the +4.6% of the smoke-prompt repeats (A3), on GSM8K's longer, more predictable outputs.
- **W9 + G1 is adopted** for both modes on the 2080 Ti. It has passed S1-S3 and T2-T4, and A1-A2 for its W7 base.

### W11: kernel test results (2026-10-30 20:00 EDT, step `x-fd75d-test1`; `attn/fd75d_test1.jsonl`)

GPU time per call inside CUDA graphs, split + combine, the engine's split rule; one layer:

| batch | query len | context | Triton path | fd75d | speedup | error vs fp64 ref (max): Triton / fd75d |
|---|---|---|---|---|---|---|
| 1 | 1 | 37 | 61.5 us | 44.5 us | 1.38x | 2.13e-5 / 2.22e-5 |
| 1 | 1 | 300 | 178.1 | 57.2 | 3.12x | 6.91e-6 / 6.87e-6 |
| 1 | 1 | 2048 | 548.7 | 72.8 | 7.54x | 2.86e-6 / 2.67e-6 |
| 1 | 1 | 8242 | 482.0 | 92.4 | 5.22x | 1.66e-6 / 1.66e-6 |
| 1 | 1 | 28050 | 732.7 | 146.3 | **5.01x** | 8.68e-7 / 7.78e-7 |
| 1 | 2 | 8242 | 539.8 | 141.5 | 3.82x | 1.55e-6 / 1.36e-6 |
| 1 | 5 | 8242 | 875.8 | 201.2 | 4.35x | 1.63e-6 / 1.58e-6 |
| 1 | 7 | 300 | 362.4 | 98.8 | 3.67x | 9.40e-6 / 9.66e-6 |
| 1 | 7 | 8242 | 1175.5 | 252.2 | 4.66x | 1.68e-6 / 1.72e-6 |
| 1 | 7 | 28050 | 2294.1 | 382.6 | **6.00x** | 1.01e-6 / 8.45e-7 |
| 2 | 1 | 8242 | 781.6 | 107.8 | 7.25x | 2.16e-6 / 2.08e-6 |
| 2 | 7 | 2048 | 1198.4 | 251.2 | 4.77x | 4.72e-6 / 4.52e-6 |

- **C1 holds** in all 29 case and split-count combinations (12 cases; splits 2, 34 and the rule): no NaN, and the error
  matches the Triton path's to within a few percent, often lower.
- **K1 holds:** 5.01x at 28,050 tokens, undrafted (3x required). **K2 holds:** 6.00x for a 7-token verify step
  (4x required).
- **K3 was wrong, in fd75d's favour:** 1.38x at 37 tokens and 3.12x at 300, not ±20%. The Triton path is not launch-bound
  there. Its split count comes from the block table's bound (32K), not the live length, so at short context one
  program per kv head walks every live key. At 300 tokens that is 4 programs on a 68-SM GPU, each paying FMA dots
  per key. fd75d inherits that split rule but splits each program's keys four ways across warps, on HMMA.
- **The engine figure checks out.** 732.7 us x 16 layers = 11.7 ms per token at 28K, which matches D3's depth cost
  (10-11 ms). fd75d would make it 2.3 ms, about 9 ms per token saved at 28K: from about 34 ms to about 25 ms
  (~29 -> ~40 tok/s), before end-to-end confirmation.

**Integration (patch `tools/turing/patches/exllamav3-1.6.5-w11-fd75d-decode.patch`, sha256 b2681d1f…, after W7-W10b;
build `1.6.5+w11`).**
- `BC_Attention::set_fd75d()` swaps the split launch in the graph-captured decode block for fd75d, which has its own
  graph parameter sites (block table, seqlens, split length, pages, splits). The Triton combine is unchanged.
- `bc_attn.py` enables it on sm_75 for an 8-bit K/V cache at head_dim 256 without window, softcap or sinks.
  `EXL3_FD75D=0` keeps Triton.
- The modified C++ and bindings pass a syntax check against torch 2.10's headers, and the patch dry-runs on the
  W10b tree.

**Queue (rev 20, after `x-ab3`).** Every W11 arm runs `EXL3_INT8_SQ_MAX_M=1`, so its int8 path is exactly W9 + G1's:
- `x-census-w11-d28`: nsys census at fill 28000, against `x-census-d28-f1`.
- `x-numerics-w11`: T3's rule; default mode runs the graph-captured decode blocks, so fd75d is in the scored forwards.
- `x-depth2`: three rotated rounds of F1 against W11 at fill 0, 8192 and 28000 (E1, E2). It adds a same-build
  `EXL3_FD75D=0` control at 28000, and MTP at 28000 for both (E3).
- GSM8K-500 undrafted and MTP under W11 (T4's rule).

### W11 scouting: depth censuses (2026-10-30 20:08 EDT; `census/compare_depth_f1.md`)

F1 undrafted, cache 32768. The prompts are 28,050-28,069 tokens at fill 28000, and the fixed `prompt_tokens` field
records them.

| family | fill 0, us per token | fill 28000, us per token |
|---|---|---|
| int8 sq GEMV (all weight matrices) | 18,495 | 18,162 |
| `_paged_attn_decode_split_kernel` | 1,030 | **11,140** |
| `_paged_attn_decode_combine_kernel` | 183 | 211 |
| GDN recurrent rule | 618 | 612 |
| total (mean ms per token) | 22.97 | 32.75 |

- **The split kernel is the whole cost of depth:** +10.1 ms of the +9.8 ms. That is 699 us per call at 28K, matching
  the bench's 733 us for the JIT path. Everything else is flat. This is what W11 replaces.
- **The next target after W11:** at fill 0, the int8 weight kernels take 18.5 of 23.0 ms. 6.63 GB in 18.5 ms is about
  358 GB/s, 58% of the 616 GB/s peak, so about 7 ms per token is left on the weights. The trellis decode's
  arithmetic is the likely limit. That will be W12.

## W10b: census results (2026-10-30 19:57-20:03 EDT; `census/compare_w10b.md`, `census/compare_w10b_nodraft.md`, fit logs beside them)

| arm | code | math | prose | mean ms/token | fp16 `exl3_gemv` + `exl3_gemm` us/token | int8 sq us/token (all grids) |
|---|---|---|---|---|---|---|
| J0 (`MAX_M=1`) | 17.69 | 16.23 | 27.27 | 20.40 | 14,814 | 1,377 |
| J2 (`MAX_M=2`) | 17.24 | 14.62 | 27.25 | 19.70 | 13,449 | 2,266 |
| J4 (`MAX_M=4`) | 18.93 | 15.58 | 26.49 | 20.34 | 2,722 | 13,316 |

- **R0, the mechanism, holds:** verify rows now reach int8. J4 cuts fp16 GEMV launches from about 160 to 29 per token,
  and the fit log lists 72 shapes at M = 1-4.
- **R0's exact-choice clause:** all 18 M = 2 shapes match the simulation exactly. Only 14 of the 36 M = 3-4 shapes
  match. The misses run 3 blocks per SM (grid 204) where the simulation assumed 4 (272): the M = 3-4 instances hold
  more than 64 registers per thread in this build, and the simulation's occupancy model assumed 64.
  **J4 therefore fails R0 and is not evaluated further. J2 passes R0.**
- **R1 fails** in any case: J4 is -0.3% against J0 (4% required).
- **R1b (reported):** J2 is -3.4% in one capture, inside the ±8% single-capture noise.
- **Why there is no gain:** per launch, the int8 kernel at 3-4 rows costs about what the fp16 kernel did (~89 against
  87-93 us). The fp16 path runs its rows on tensor cores, which take extra rows almost for free. The int8 path does
  per-row integer dots on the ALUs, so its cost grows with M. Upstream's 3090 finding ("larger m loses to the fp16
  tensor-core kernel") holds on Turing as a tie.
- **R3 cannot be tested as written:** `census.py` rows do not record the generated text. The evidence for an unchanged
  one-row path is structural: J4n launches the same grids and counts as F1 (399.43 int8 launches per token, every
  family the same).
- **What remains of W10b:** J2 only (upstream's own m = 2 rule, active on Turing for the first time). Queue rev 21
  drops `x-numerics-w10bm4s4` and cuts `x-ab3` to J0 against J2. J2 is adopted only if it wins every round there and
  passes the step-2 numerics.
- **Default caution:** builds containing W10b run J2's rule by default (`EXL3_INT8_SQ_MAX_M` defaults to 2). Every
  W11 arm sets `MAX_M=1` until J2 is decided.

## W11b: splits over the live context (pre-registered 2026-10-30 20:20 EDT, before any W11b run)

**Why.** fd75d inherits the Triton rule: the splits are cut from the block table's bound (32K), not the live length.
At 2,048 tokens only 3 of 34 splits hold keys, so 12 blocks work on a 68-SM GPU. In the kernel test fd75d sits about
10x above its memory floor at 2,048 tokens, but only ~1.5x at 28,050.

**Change.** A kernel flag. With it set, each split takes an equal, 16-key-aligned share of the live context: it reads
`cache_seqlens` on the device, so the graph needs no new patch sites. The combine is unaffected: it merges partials
whatever their boundaries. Host switch `EXL3_FD75D_LIVE` (test harness: both variants on the same data).

**Predictions** (standalone test, step `x-fd75d-test2`):
- **L1.** At contexts 2,048 and 8,242, query length 1: live splits are at least 1.5x faster than bound splits.
- **L2.** At 28,050: within ±10% (29 of 34 splits are already live there).
- **C1** (as for W11) holds for the live variant in every case.

### W11b: kernel results (2026-10-30 20:17 EDT, step `x-fd75d-test2`; `attn/fd75d_test2.jsonl`)

fd75d (+ combine), the engine's split count, block-table splits against live splits; same data as test 1:

| batch | query len | context | Triton path | fd75d, bound splits | fd75d, live splits | live / bound |
|---|---|---|---|---|---|---|
| 1 | 1 | 37 | 52.9 us | 38.1 | 38.1 | 1.00x |
| 1 | 1 | 300 | 175.0 | 48.4 | 44.9 | 1.08x |
| 1 | 1 | 2048 | 513.0 | 89.7 | 53.1 | **1.69x** |
| 1 | 1 | 8242 | 497.6 | 94.0 | 79.1 | **1.19x** |
| 1 | 1 | 28050 | 709.6 | 145.2 | 145.2 | 1.00x |
| 1 | 7 | 8242 | 1145.5 | 248.0 | 136.6 | 1.82x |
| 1 | 7 | 28050 | 2223.7 | 381.7 | 355.1 | 1.07x |
| 2 | 7 | 2048 | 1173.0 | 267.5 | 110.7 | 2.42x |

- **L1 holds at 2,048 tokens (1.69x) and fails at 8,242 (1.19x, 1.5x required).** At 8K the bound rule already
  leaves 9 of 34 splits live. What remains is per-block overhead, plus the Triton combine walking 16 rows per program
  for 6 live ones.
- **L2 holds** (1.00x at 28,050).
- **C1 holds** for the live variant in all 12 cases.
- **Live is never slower**, and it gains most on verify steps (1.82x for 7 tokens at 8K) and with batch 2.
- **Variation between the tests:** test 1 and test 2 measured the same bound-split case at 2,048 tokens as 72.8 and
  89.7 us. Single kernel timings on this desktop vary by up to ~20% (display time-slicing), so the end-to-end
  repeats decide.
- Engine patch `exllamav3-1.6.5-w11b-fd75d-live-splits.patch` (sha256 2e469042…, after W11) adds the switch
  `EXL3_FD75D_LIVE`, off by default. `x-depth2` (rev 23) now runs on the `1.6.5+w11b` build. Its live-off arms take
  W11's exact code path; it adds live-on arms at 0, 8K and 28K and live-on MTP at 28K, and it waits for the build.

### The launch-shape bench (step `x-attn-bench1`; `attn/bench1.jsonl`)

The best of 72 Triton launch settings (KV tile, warps, stages, splits) per case, against the default and against fd75d:

| context | query len | Triton default | Triton best (setting) | fd75d (test 2, live) | memory floor |
|---|---|---|---|---|---|
| 512 | 1 | 276.5 us | 138.2 (tile 16, 8 warps, 1 stage, 272 splits) | ~45 | 2 |
| 8192 | 1 | 501.4 | 167.2 (32, 8, 2, 68) | 79.1 | 32 |
| 28050 | 1 | 703.9 | 469.8 (16, 4, 1, 136) | 145.2 | 111 |
| 28050 | 7 | 2239.1 | 1476.5 (32, 8, 2, 68) | 355.1 | 111 |

Tuning Triton's launch shape recovers 1.5-3x. The tensor-core kernel is another 2-4x beyond the best setting, and at
28K it is within 1.3x of the memory floor. On Turing the lever is the arithmetic unit, not the configuration.

## W11 in the engine, W10b J2 decided (2026-10-30 20:40 EDT)

**Census at fill 28000** (`census/compare_w11_d28.md`; build `1.6.5+w11`, `EXL3_INT8_SQ_MAX_M=1`, otherwise F1):

| family | F1, us per token | W11, us per token |
|---|---|---|
| `_paged_attn_decode_split_kernel` (Triton) | 11,140 | - |
| `fd75d_split_kernel` | - | **2,149** (135 us per call) |
| `_paged_attn_decode_combine_kernel` | 211 | 222 |
| int8 sq GEMV | 18,162 | 19,533 |
| total (mean ms per token) | 32.75 | **25.35 (-22.6%)** |

- In the engine fd75d takes 135 us per call against Triton's 699: 5.2x, matching the standalone test.
- The rest of the token is unchanged within this desktop's capture-to-capture variation (int8 +7.5% here, i.e. the two
  captures ran at slightly different clocks). The interleaved repeat `x-depth2` gives the speed claim (E1, E2).

**Numerics (T3's rule; `numerics/compare_w11-e47-n50.txt`).** W11 default mode against the stock fp32 oracle: exact KL
mean 7.32e-5 on the stride-16 sample (n = 768), top-1 agreement 99.78%. Both are inside 2e-4 and 99%. W9 + G1 had
8.69e-5 and 99.87%. Default mode runs the graph-captured decode block, so every scored forward used fd75d.

**W10b J2:**
- **R2 holds** (`numerics/compare_w10bm2s2-e47-n50.txt`): two tokens per forward under `EXL3_INT8_SQ_MAX_M=2`,
  exact KL 7.83e-5, top-1 99.85%.
- **x-ab3** (`ab/ab3.md`, MTP, three rotated rounds): J2 beats J0 in every round, +0.4%, +1.9% and +3.2% (mean +1.8%;
  56.38 against 55.38 tok/s).
- **J2 meets W10b's adoption rule** up to the GSM8K-500 rerun under MTP.

**Amendment (recorded before either W11 GSM8K-500 run starts).** The two queued GSM8K-500 runs move to the `1.6.5+w11b`
build:
- The **MTP** run adds `EXL3_INT8_SQ_MAX_M=2`, so one run carries W11 and J2. Both passed numerics on their own. A
  pass adopts both for MTP. A fail leaves both unadopted until they are run separately.
- The **undrafted** run keeps `MAX_M=1` (J2 does not touch one-row decoding).
- **Live splits.** Both runs take `EXL3_FD75D_LIVE=1` if, in `x-depth2`, the live arm's mean is no more than 1% below
  the bound arm's at every fill, and the live MTP arm finishes. Otherwise they keep the bound splits, W11 as built.
  The choice is recorded here before the runs start.

## W12 scouting: the int8 weight kernel is bound by per-weight work, not bytes (2026-10-30 20:55 EDT)

`tools/turing/census_bits.py` tags every int8 launch of the fill-0 census (`census/d0-f1-mgemmn0/int8_by_bits.md`) with
its role, from the layer's kernel order, and its bit width K, from the kernel's template arguments:

| role (k x n) | K | median us | weight MB | GB/s | M weights per us |
|---|---|---|---|---|---|
| MLP gate/up (5120 x 17408) | 1 | 53.5 | 11.14 | 208 | 1.66 |
| MLP gate/up | 2 | 54.8 | 22.28 | 406 | 1.63 |
| MLP gate/up | 3 | 72.6 | 33.42 | 460 | 1.23 |
| MLP down (17408 x 5120) | 1 | 59.1 | 11.14 | 189 | 1.51 |
| MLP down | 2 | 60.5 | 22.28 | 368 | 1.47 |
| MLP down | 3 | 79.3 | 33.42 | 421 | 1.12 |
| GDN out_proj (6144 x 5120) | 2 | 29.3 | 7.86 | 269 | 1.07 |
| GDN out_proj | 3 | 37.7 | 11.80 | 313 | 0.83 |

- **At K = 1 and K = 2 a matrix takes the same time**, though K = 1 is half the bytes: time follows weights, not bytes.
  The 2080 Ti's int8 GEMV is bound by per-weight decode work. Fewer bits buy VRAM, not speed. That confirms the
  research survey's "[Reported, medium confidence]" claim (§5.4 of `TURING_EXL3_RESEARCH_2026-10-30.md`) with this
  lab's own data.
- **Per weight, the kernel does:** an extraction of the 16-bit trellis window (shifts and masks), a 32-bit multiply
  by the mul1 hash `0x83DCD12D`, and one `dp4a` against the splatted int8 activation per row. On top come
  shared-memory loads of the splats every 8 weights, and address and loop overhead.
- **Throughput:** 1.63 M weights per us is about 14 weights per clock per SM. At an estimated ~8 instructions per
  weight that is near Turing's issue limit (4 warp-instructions per clock per SM, i.e. 128 thread-instructions).
- **The bar for W12.** To hit the memory floor (a 2-bit MLP matrix in 36 us, not 55), the kernel needs about 5.5
  instructions per weight or fewer, all-in. Next: count the inner loop's instructions in the sm_75 machine code
  (`cuobjdump -sass`), then design a cheaper decode. The MLPs alone are ~10.6 ms of the 23 ms token, so a 1/3 cut
  there is about 3.5 ms per token.

### W12 scouting: the hot loop's machine code (`census/d0-f1-mgemmn0/sq_k2m1_hot_loop.sass`)

The sm_75 SASS of `exl3_gemv_int8_sq_kernel<2, 1, false, false, false>` (W10 source; the one-row path is W9's) has a
hot loop from 0x2f30 to 0x39d0: 171 instructions for 32 weights per thread, i.e. 5.3 per weight.

| per 32 weights | count | pipe on Turing |
|---|---|---|
| `IDP.4A.U8.S8` (codebook value x activation) | 32 | FMA |
| `IMAD` (the mul1 hash x * 0x83DCD12D, plus addressing) | 36 (+7 wide/X) | FMA |
| `SHF` + `SGXT` (16-bit window extraction) | 52 | ALU |
| `IADD3`, `LEA`, `LOP3`, `PRMT` (addressing, permutes) | ~25 | ALU |
| `LDG` (8 x the trellis words), `LDS.64` (4 x the splats) | 12 | memory |

- **Correction (2026-10-30 21:05 EDT).** An earlier version of this section put `IDP` on the integer ALU and called
  the loop integer-pipe bound. Nsight Compute's pipe documentation says otherwise: its FMA pipe executes `IMAD`,
  `IMUL` and integer dot products, and the ALU executes shifts, logic, `IADD3`, `LEA` and `PRMT`. Re-tallied per
  32 weights: ~75 FMA-pipe and ~82 ALU instructions, 171 issued.
  - The three ceilings are close: ~25 weights per clock per SM for each pipe (16 lanes per sub-core) and ~24 for issue
    (one warp-instruction per clock per sub-core).
  - Measured: ~14 (1.63 M weights per us at ~1.75 GHz), about 58% of each. **No single pipe is saturated.**
  - Memory and dependency latency, partial waves and per-launch overhead are as likely the limit. The hot loop alone
    cannot settle it.
- **Measuring it needs GPU counters.** The desktop's driver has `RmProfilingAdminOnly: 1`, so Nsight Compute (pipe
  utilization, stall reasons) needs one sudo session from Matt. Changing the setting would need a module reload or
  reboot, which is not done. Without counters, kernel variants can isolate the limit: one with the dp4a removed,
  one with the extraction removed, one with the loads replaced by constants.
- **The tensor-core idea, assessed against the corrected tally.**
  - `mma.sync.m8n8k16.s32.s8.u8` fits the current lane mapping with no data movement. Lane (g, c)'s products
    w0..w3 have k in {2c, 2c+1, 2c+8, 2c+9}, depending only on c, and output column g. Each thread's hash product is
    its B fragment, and the per-lane splat word is its A fragment. The integer sums are exact, so results would be
    bit-identical.
  - **One row:** it issues the same number of warp instructions as the dp4a (one per product per thread), so it
    relieves the FMA pipe but not issue. Worth it only if the counters show FMA pressure.
  - **Several rows:** A's rows 0-7 can carry 8 different activation rows at no extra cost, where the dp4a path pays
    per row. That is exactly what W10b lacked: verify rows at int8 tied the fp16 tensor-core kernel because the int8
    cost grew with M. The multi-row case is the stronger motivation, and it is a design candidate for W12, to be
    pre-registered before any build.

### W12 scouting: fixed plus per-weight cost (`census/d0-f1-mgemmn0/int8_by_bits.md`, all roles)

With every role classified (GDN qkv/z/out, attention q/kv/o, MLP gate/up/down, head), the per-role median durations
fit t = a + b x weights per bit width:

| K | roles | fixed a (us per launch) | b (us per million weights) |
|---|---|---|---|
| 1 | 4 | 8.1 | 0.543 |
| 2 | 7 | 12.0 | 0.508 |
| 3 | 5 | 14.9 | 0.691 |

- **Per token:** ~27 billion weights x ~0.51 us per million ≈ 14 ms of per-weight work. ~400 int8 launches x ~12 us ≈
  4.8 ms of fixed cost. The sum, ~18.8 ms, matches the census's 18.5.
- **Small matrices are nearly all fixed cost:** attention k/v (5.2 M weights) take 13.8-16.3 us.
- **Per-weight rate:** ~1.97 M weights per us at K = 2, ~16.5 weights per clock per SM, about 69% of the issue ceiling.
- **Two W12 levers, in order of tractability:**
  - **W12a, multi-matrix int8 launch.** One launch for the matrices that share an input: MLP gate + up, GDN qkv + z,
    attention q + k + v. The activation is staged once, and the fixed cost is paid once per bundle instead of per
    matrix. Estimated saving: 48 x 12 + 64 x 12 + 16 x 24 us ≈ 1.7 ms per token (~7% of an undrafted token). G1
    unfused exactly these bundles, because the fused path ran the fp16 cooperative kernel. An int8 bundle would have
    both benefits.
  - **W12b, fewer instructions per weight:** cheaper window extraction (`PRMT` does shift and mask in one instruction
    for byte-aligned windows), and the multi-row tensor-core path for verify rows.
  - What the ~12 us fixed cost is made of (activation staging, the serialized epilogue, fences and atomics, wave
    tails) decides W12a's design. Kernel variants measure it before anything is pre-registered.

## W11 / W11b: interleaved depth repeats (2026-10-30 20:46-21:31 EDT, step `x-depth2`; `ab/depth2.md`, raw rows `ab/depth2.jsonl`)

Undrafted decode tok/s, averaged over the three smoke prompts, cache 32768, 12 arms in rotated order, three rounds.
W11 arms run the `1.6.5+w11b` build with `EXL3_INT8_SQ_MAX_M=1` and live splits off, which is W11's code path:

| fill | F1 (W9 + G1) | W11 | W11, live splits | W11 against F1, per round |
|---|---|---|---|---|
| 0 | 41.58 | 43.24 | 42.65 | -1.1%, +11.8%, +2.0% (mean +4.2%) |
| 8192 | 34.94 | 40.93 | 40.87 | +16.0%, +17.0%, +18.6% (mean +17.2%) |
| 28000 | 28.90 | **39.91** | 39.67 | +36.2%, +45.0%, +33.0% (mean +38.1%) |
| 28000, MTP | 37.46 | - | **52.13** | +66.5%, +32.2%, +24.6% (mean +41.1%) |

Same-build control at fill 28000: fd75d off is 29.64 tok/s, +2.6% over F1 (3 of 3 rounds). So the gain is the kernel,
and the build is otherwise F1.

- **E1 holds:** at fill 28000, W11 is at least 20% faster than F1 in every round (+33-45%). Undrafted decode at 28K is
  now 39.9 tok/s: 2.04x the stock wheel's 19.6 (D), and 1.9x llama.cpp's best undrafted 21.2 at 32K.
- **E2 misses its band, in W11's favour:** the fill-0 mean is +4.2%, outside ±3%. F1's slow round 2 (38.99) drives
  it. The other rounds are -1.1% and +2.0%, and there is no regression.
- **E3 (reported):** MTP at fill 28000 gains +41.1%, from 37.5 to 52.1 tok/s. MTP's acceptance varies from process to
  process (F1's rounds 32.2-42.2), but W11 wins every round.
- **The 8K point (not predicted):** +17.2%, every round.
- **W11b (live splits) gives nothing end to end:** -1.4%, -0.1% and -0.5% against bound splits at fill 0, 8K and 28K.
  The kernel-level gain at 2-8K (L1) is under 1% of a token. The amended rule (live only if no fill is more than 1%
  below bound) fails at fill 0. **The two W11 GSM8K-500 runs keep bound splits**, W11 as built: decision file
  `w11_live.decision = 0`, written at 21:34 EDT. The undrafted run's step had started at 21:31 but was holding on that
  file before starting its server, so no item ran before the decision.

## T4 for W11 (and J2 under MTP): result, and adoption (2026-10-30 23:28 EDT; `quality/compare_t4_w11.md`, rows in `quality/TURING-exl3-nodraft-w11/` and `quality/TURING-exl3-mtp-w11j2/`)

GSM8K-500 on the `1.6.5+w11b` build with bound splits (`EXL3_FD75D_LIVE=0`, the amendment's decision), each against the
stock lane of the same mode. Both runs went through without a guard stop:

| label | build and switches | accuracy [95% CI] | mean tokens | wall per item | decode tok/s (median) |
|---|---|---|---|---|---|
| TURING-exl3-nodraft (stock) | 1.6.5 wheel | 97.6% [95.9, 98.6] | 325.8 | 14.22 s | 23.6 |
| TURING-exl3-nodraft-w9g1 | W9 + G1 | 97.0% [95.1, 98.2] | 328.5 | 8.32 s | 40.9 |
| TURING-exl3-nodraft-w11 | W11, `MAX_M=1` | 96.8% [94.9, 98.0] | 328.5 | 7.95 s | 42.6 |
| TURING-exl3-mtp (stock) | 1.6.5 wheel | 97.6% [95.9, 98.6] | 327.2 | 6.45 s | 53.4 |
| TURING-exl3-mtp-w9g1 | W9 + G1 | 97.4% [95.6, 98.5] | 324.7 | 5.62 s | 61.1 |
| TURING-exl3-mtp-w11j2 | W11 + J2, `MAX_M=2` | **97.6%** [95.9, 98.6] | 323.8 | **5.44 s** | **63.2** |

- **Undrafted: T4 holds.** The delta is -0.80 points, inside the -1.0 bound. The discordant items are 4 / 0, so the
  exact McNemar test gives p = 0.125, not significant. The final answers agree on 495 of 500 items, and 66.6% of the
  outputs are byte-identical. **W11 is adopted for undrafted decoding.** Wall time per item is 1.79x lower than stock.
- **MTP: T4 holds with no discordant item.** 97.6% against 97.6% (0 / 0, p = 1). The final answers agree on 499 of
  500 items, and 88.6% of the outputs are byte-identical. **W11 and J2 are both adopted for MTP**, as the amendment
  set out. Wall time per item is 1.19x lower than stock, and the median decode rate is 63.2 tok/s (stock 53.4).
- **GSM8K is short context**, so W11 adds little here: +4% median decode over W9 + G1 undrafted, +3% under MTP. That
  matches `x-depth2`'s fill-0 point (+4.2%). W11's gain grows with context (+17% at 8K, +38% at 28K).
- **The build in the servers.** The run records hold `PYTHONPATH=.../overlay/exl3-w11b` and the switches above in
  `lane_env`. TabbyAPI prints the wheel's metadata version (1.6.5), not the overlay's. The servers had exited before
  this check, so there is no process map as in T4 undrafted. The decode rates (42.6 and 63.2 against stock's 23.6 and
  53.4) can only come from the overlay.

**Where the undrafted misses fall (diagnostic, after the result).** Every discordant item in both patched undrafted runs
goes to stock (3 / 0 for W9 + G1, 4 / 0 for W11). Each run passes on its own, and the two runs share their items, so
they cannot be pooled into one test. The item lists show what the direction is:
- Stock undrafted and stock MTP miss the same 12 items. Every patched lane misses those 12 too.
- The extras are W9 + G1's gsm8k_12, gsm8k_505 and gsm8k_1059, and W11's gsm8k_12, gsm8k_154, gsm8k_252 and gsm8k_1190.
  Only gsm8k_12 recurs, so most extras differ from run to run. That fits near-ties that flip whenever the arithmetic
  changes, not a fault in either kernel.
- **The laptop misses the same items.** The laptop runs upstream's int8 GEMV path for undrafted decoding: its sm_89
  part has 100 KB of shared memory, which passes upstream's check, and E47's exllamav3 1.4.7 already carries the int8
  sq kernels (`engines/exl3-1.4.7-lab/.../exl3_gemv_int8_inst_sq_k*.cu`). Its E47 undrafted lane `EFF-h38a_nodraft` misses the 12 shared items plus gsm8k_12 and
  gsm8k_505 (97.2%). It misses gsm8k_12 in 7 of its 8 undrafted runs that cover it, and gsm8k_154 in both
  `h38e_nodraft_dev` runs. Its MTP lane misses exactly the 12 shared items, like every Turing MTP lane here.
- So stock Turing undrafted was the outlier: upstream declined int8 on 64 KB parts, and it ran the fp16 GEMV. W7 puts
  Turing on the int8 activation path that upstream ships on larger GPUs, and those near-tie items go with it. The
  int8 path quantizes the activation vector, which is the visible numerics change. T3 bounds it (exact KL 7.3e-5,
  top-1 99.78%), and the laptop shows the same items under upstream's own kernel.
- Under MTP the verify forwards (2-7 rows) carry most of the weight work, and rows of 3 or more stay on the fp16
  tensor-core kernels. int8 covers the one-row draft forwards, the head and, with J2, two-row verify forwards. That
  keeps MTP's numerics closer to stock: 88.6% byte-identical against 66.6% undrafted.

**Adopted configuration on the 2080 Ti from here:** the `1.6.5+w11b` build (W7, W9, W10b's slice cap, W11's fd75d
attention, live splits off), `EXL3_MGEMM_N_THRESHOLD=0`, and `EXL3_INT8_SQ_MAX_M=2` under MTP (`1` undrafted is
equivalent, since one-row decoding never uses M=2). The queue's later quality, speed and sweep steps keep the
configuration they were queued with. A final interleaved repeat of this configuration against stock is the speed figure
of record (queued after W12's bench).

## Final speed of record: adopted configuration against stock (pre-registered 2026-10-30 23:36 EDT, before the step runs)

Step `x-final1` (queue rev 26, after `x-gemv-bench1`): `ab_speed.sh final1`, three rotated rounds, cache 32768, eight
arms. Stock is the 1.6.5 wheel with no switches. "Adopted" is the `1.6.5+w11b` overlay with `EXL3_MGEMM_N_THRESHOLD=0`,
`EXL3_FD75D_LIVE=0`, and `EXL3_INT8_SQ_MAX_M=1` undrafted or `2` under MTP (6 draft tokens, dynamic). Each mode runs at
fill 0 and fill 28000. Predictions, from D, `x-depth2` and `x-ab3`:
- **S1:** undrafted at fill 0, adopted at least 1.6x stock in every round (D and depth2 put it near 1.78x).
- **S2:** undrafted at fill 28000, adopted at least 1.8x stock in every round (near 2.0x expected).
- **S3:** MTP at fill 0, adopted faster than stock in every round, mean at least 1.05x.
- **S4:** MTP at fill 28000, adopted faster than stock in every round, mean at least 1.3x.
A miss is reported as a miss. It changes no adoption, since T3 and T4 already decided those, but the paper quotes these
rows as the speed of the adopted configuration.

## W12 scouting: the int8 GEMV alone, and a configuration-switch check (2026-10-30 23:30-23:36 EDT; `w12/gemv_bench1.jsonl`, `w12/gemv_bench1.log`)

`tools/turing/gemv_bench.py` (step `x-gemv-bench1`, w9 overlay) times `ext.exl3_gemm` on synthetic mul1 tensors, one
activation row, as a CUDA graph of 20 back-to-back calls. Try 1 failed before writing a row (`exllamav3.__version__`
does not exist; it is now read from `exllamav3.version`), and try 2 ran the fixed script. Microseconds per call:

| shape (k x n) | K = 1 | K = 2 | K = 3 | K = 4 | in-model census, same K |
|---|---|---|---|---|---|
| tiny 128 x 256 | 7.08 | 5.93 | 6.68 | 5.64 | - |
| attn k/v 5120 x 1024 | 19.39 | 11.80 | 13.43 | 14.57 | 16.3 (K=3), 13.8 (K=4) |
| GDN z 5120 x 6144 | 31.61 | 24.95 | 31.62 | 50.63 | 26.2 (K=2) |
| GDN out 6144 x 5120 | 34.02 | 25.04 | 31.64 | 50.85 | 29.3 (K=2) |
| GDN qkv 5120 x 10240 | 43.83 | 36.35 | 46.10 | 63.91 | 36.4 (K=2) |
| attn q 5120 x 12288 | 48.73 | 46.00 | 55.70 | 65.86 | 44.2 (K=2) |
| MLP gate 5120 x 17408 | 59.24 | 54.10 | 72.48 | 85.69 | 54.8 (K=2) |
| MLP down 17408 x 5120 | 64.06 | 61.10 | 77.62 | 97.81 | 60.5 (K=2) |
| fit over the unfused shapes | 12.89 + 0.563/M | 7.42 + 0.570/M | 8.53 + 0.743/M | 12.52 + 0.918/M | 12.0 + 0.508/M (K=2) |

Fused shapes (one launch, one input transform) against the sum of their separate launches, K = 2:

| bundle | separate launches | one fused launch | saving |
|---|---|---|---|
| GDN qkv + z (5120 x 16384) | 61.30 | 53.33 | 7.97 (13%) |
| attention q + k + v (5120 x 14336) | 69.60 | 53.10 | 16.50 (24%) |
| MLP gate + up (5120 x 34816) | 108.20 | 107.78 | 0.42 (0%) |

- **The bench reproduces the engine.** At K = 2 the large shapes agree with the census within 2 us (GDN qkv 36.35
  against 36.4, MLP gate 54.10 against 54.8, MLP down 61.10 against 60.5). GDN out is 4 us faster alone. Attention
  k/v at K = 3 is faster alone too (13.4 against 16.3): its 2 MB of weights stay in the 5.5 MB L2 when one matrix
  repeats 20 times, which a decode never gets.
- **The "~12 us fixed cost" was mostly the fit's extrapolation.** The census fit spans 5-90 M weights. With the tiny
  and small shapes the K = 2 intercept drops to 7.4 us, and the floor for a near-empty launch is 5.6-7.1 us. Time is
  not linear in weights: small matrices cost more per weight (too few units to fill 68 SMs), large ones less.
- **W12a's estimate falls from ~1.7 to ~0.6 ms per token.** Gate + up gain nothing fused: at 22 MB per matrix the
  launch is already a small part. GDN qkv + z saves 8 us x 48 layers = 0.38 ms, and attention q + k + v up to
  16.5 us x 16 = 0.26 ms. That is an upper bound, for two reasons:
  - each real matrix has its own input sign flips (`suh`), so a fused launch still stages every matrix's own
    transformed slice. The sq kernel stages per (slice, 256-column) unit
    (`exl3_gemv_int8_kernel.cuh`, `gemv_int8_stage_slice`), so only the launch and the wave tail are saved;
  - attention's q and k/v differ in K (2 against 3-4), so one launch would need per-unit K dispatch.
  **W12a is set aside:** at most ~2.5% undrafted, for a multi-matrix kernel with per-matrix K.
- **Per-weight cost by K:** K = 1 and K = 2 cost the same per weight (0.563 and 0.570 us per million), i.e. a compute
  ceiling of ~1.76 M weights per us. K = 3 runs at 0.743 (505 GB/s) and K = 4 at 0.918 (545 GB/s), near the memory
  floor. **K = 2, most of this model, runs at 439 GB/s, 71% of the memory floor**, limited by per-weight work. If the
  per-weight time fell to the memory floor (0.406 us per million), the K = 2 matrices would save ~28% of their time,
  roughly 3 ms of a 23 ms token. That is W12b's prize, and why W12b comes next.

**Configuration-switch check (diagnostic, census `d0-f1-mgemmn0`, three captures, 768 tokens).** Upstream's int8 host
sets `cudaSharedmemCarveoutMaxShared` and its comment reports ~4 us per launch for carveout switches in graph replay.
CUPTI's `sharedMemoryExecuted` shows that on this card every int8 launch runs at the 64 KB shared configuration and
nearly every other decode kernel at 32 KB (RMS norm, act_mul, the GDN kernels, rope, cache quantization; Triton
attention runs at 64 KB). So 62% of consecutive kernel pairs switch, ~574 switches per token. The cost is small:
- the gap before a kernel grows from 0.22 to 0.49 us (32 to 64 KB) and from 0.30 to 0.77 us (64 to 32 KB), means;
- inside the kernel nothing shows: attention k, launched right after a 32 KB kernel, and attention v, right after k,
  take 16.45 against 16.06 us (K = 3) and 13.76 against 13.79 (K = 4), medians over 12,240 layer instances.
So switches cost ~0.2 ms per token (~1%) on this card. One uniform carveout would be a one-line context setting, but
it buys at most ~1%, and it is noted, not pursued.

## W12b limiter study: the per-weight loop alone (pre-registered 2026-10-31 00:07 EDT, before any timing of it)

Times in this section and the two above are the commit times (corrected at 00:15 EDT). The first stamps ran ahead
of the clock: 23:50, 00:20, 00:45 and 01:05 were written for entries committed at 23:36, 00:07, 00:10 and 00:13.
Neither `x-final1` nor `x-w12-sqmb1` had started at any of them.

`tools/turing/w12/sqmb.cu`, driven by `tools/turing/w12_sqmb.py` (step `x-w12-sqmb1`), runs the sq kernel's
(k-slice x 256-column) units for K = 2, one row. Splats are copied from memory instead of transformed, and there is no
epilogue, so the time is the per-weight loop plus a small per-unit cost. It includes `exl3_gemv_int8_kernel.cuh`
from the `w11b` source tree and calls upstream's own units for V0 and V1. Grid and slice height follow the sq rule.
Variants:

| V | what | tests |
|---|---|---|
| 0 | upstream narrow unit (the engine's K = 2 path) | baseline |
| 1 | upstream cp.async-staged unit (upstream routes K = 3/5/7 there) | load latency |
| 2 | narrow, PRMT window extraction | ALU / issue count |
| 3 | staged + PRMT | both |
| 4 | narrow, words from a register hash instead of loads | compute-only time |
| 5 | narrow, dp4a replaced by adds | FMA-pipe share |
| 6 | narrow, no hash multiply | FMA-pipe share |
| 7 | loads only, 16-byte loads of the same rows | memory floor of this layout |

Compiled on the desktop CPU (`logs/build-w12-sqmb.log`). The machine code (`cuobjdump`, whole kernels) shows the PRMT
form barely changes the count: 536 instructions against 544. PRMT replaces the 36 `SGXT` and 15 `SHF`, but costs 48
`PRMT`, since the compiler already extracts eight windows in ~14-15 instructions per tile. V0 has 144 `IMAD` for 48
hash multiplies: the Turing compiler issues moves and adds as `IMAD` to balance its pipes.

**Checks and decision rules (per shape: MLP gate, MLP down, GDN qkv, attention q):**
- **C0:** V1-V3 must equal V0's partials bit for bit. A variant that differs is a bug and is not timed further.
- **L1, latency:** if V4 is at least 20% faster than V0 on every shape, and V7 at least 20% faster too, the loop is
  not bound by its own arithmetic. Load latency or bandwidth then dominates.
- **L2, staging:** if V1 beats V0 by at least 5% on every shape, **W12b-1** follows: route K = 1 and K = 2 to the
  staged unit on sm_75. It is a one-line host and kernel routing change, pre-registered for the engine before any
  build.
- **L3, PRMT:** if V2 beats V0 by at least 3% on every shape (or V3 beats V1), PRMT extraction joins W12b. Otherwise
  it is dropped.
- **L4, pipes (reported):** V5 and V6 against V0 show how much the FMA pipe (dp4a, hash) holds. If both are within 5%
  of V0, the FMA pipe is not the limit, and the IMMA one-row path is not worth building.
- The slice-height sweep (MLP gate, 16-320 rows) shows whether the sq rule's slice choice matters on Turing.

**Amendment (00:10 EDT, before any timing; the step had not started).** Two variants are added.
- **Why:** V0's machine code recomputes each row's 64-bit tile address from `kb` (`IADD3`, `LEA`, `IMAD.WIDE`,
  `IMAD.X`: ~15 instructions per row, for 16 weights per lane). Each lane's two word offsets are constant.
- **V8**, narrow with running pointers: two pointers step one row, and tile B is an immediate +64 bytes.
- **V9:** V8 + PRMT.
- **C0 extends to V8 and V9.**
- **L5:** if V8 (or V9) beats V0 by at least 5% on every shape, running pointers join W12b.
- **Instruction counts after the change** (whole kernels): V0 544, V8 528 (`LEA` 21 to 8), V9 512, V1 416, V3 400.
  Counted between the first and last `IDP` and scaled to 16 weights, the hot loop takes about 67 instructions in V0,
  62 in V8, 56 in V9, 53 in V1 and 46 in V3. That is a rough measure, since address work can sit outside the span.

**Second amendment (00:13 EDT, before any timing): the M-row part, W12c scouting.** The same step also times verify-row
shapes (MLP gate, MLP down, GDN qkv; M = 1, 2, 3, 4, 6, 8):
- **The form:** upstream's dp4a narrow unit at M rows (`gemv_int8_unit_narrow<2, M>`) against an integer tensor-core
  form (`mb_unit_imma<M>`, `mma.sync.m8n8k16.s32.s8.u8`, 288 `IMMA` in the build).
  - Lane (g, c)'s hash words w0..w3 / w4..w7 are its B fragments as they stand.
  - Row g's splats at k = 2c, 2c+1, 2c+8, 2c+9 are its A fragments (zero for g >= M).
  - One instruction covers 4 weights x 8 columns x 8 rows, so the row count costs nothing extra up to 8.
- **C0:** the IMMA partials must equal the dp4a partials bit for bit (exact int32 sums), at every M.
- **R-rule (scouting; decides whether W12c is built for the engine):**
  - The IMMA form at M = 4 must be at least 25% faster than dp4a at M = 4 on all three shapes.
  - IMMA at M = 8 must stay within 1.3x of IMMA at M = 1.
  - If both hold, W12c (IMMA int8 for verify rows, replacing the fp16 tensor-core GEMV under MTP) gets a
    pre-registration with engine-level predictions. Otherwise it is dropped.
- **IMMA at M = 1 against V8 is reported:** it relieves the FMA pipe of the dp4a for one-row decoding.

**W12c engine patch, drafted ahead of the decision (00:24 EDT).** The overlay build takes about an hour of CPU, so it
runs now in a capped unit (`turing-build-w12c`). If the R-rule fails, the overlay is not used.
`tools/turing/patches/exllamav3-1.6.5-w12c-int8-imma-verify.patch` (on top of the w11b chain):
- `gemv_int8_unit_imma<bits, M, HALF>`: the microbenchmark's IMMA unit in the narrow mapping, for every K. K = 4's
  wide unit pairs lanes differently and cannot form the fragments.
- The sq kernel runs it for M >= 3 without residual; M = 1 and 2 keep dp4a. M = 3 and 4 move from W10's dp4a, which
  tied fp16 in W10b.
- Instances for M = 5-8.
- The sq epilogue loops its 4 warp pairs over the rows.
- The workspace's per-slice scale region is sized for M <= 8.
- The host takes `EXL3_INT8_SQ_MAX_M` up to 8, and M >= 3 only on compute capability 7.2+.
- **The default (`MAX_M = 2`) is unchanged,** so one environment variable gives the A/B.

Nothing is measured with it before its own pre-registration.

### W12c: engine-level exactness and per-launch timing (pre-registered 2026-10-31 00:26 EDT, before the build finished)

Step `x-w12c-check1` (after `x-w12-sqmb1`; it waits for `overlay ready` in `build-w12c.log`):
- **Exactness, `tools/turing/w12c_check.py`.** Synthetic mul1 tensors, seeded per (shape, K), so every process builds
  the same ones. Shapes: attention k/v, GDN z, GDN qkv, MLP gate, MLP down. K = 1-4, M = 1-7. `ext.exl3_gemm` is
  dumped under four configurations:
  - w11b with `MAX_M=4` (W10's dp4a at M = 3-4);
  - w11b with `MAX_M=2` (fp16 for M >= 3, today's MTP path);
  - w12c with `MAX_M=4`;
  - w12c with `MAX_M=8`.
- **Per launch, `gemv_bench.py --m 1-7`:** seven shapes, K = 1-4. w11b `MAX_M=2` (fp16 for M >= 3) against w12c
  `MAX_M=8` (IMMA for M >= 3).

**Rules:**
- **X1:** w12c `MAX_M=4` equals w11b `MAX_M=4` bit for bit in every case. At M = 3-4 that is IMMA against dp4a,
  with the same partials and epilogue. At M = 1-2 nothing changed.
- **X2:** w12c `MAX_M=8` is finite everywhere. Its largest relative difference from the fp16 path at M = 5-7 is
  reported, and is expected near the int8 activation error of the M = 3-4 cases against fp16. A bug shows as
  differences of order 1.
- **X3, speed:** take MLP gate, MLP down and GDN qkv at K = 2. For each shape, the median over M = 3-7 of
  fp16 time / IMMA time must be at least 1.15, and no single (shape, M) may be slower under IMMA.
- **If X1-X3 hold,** W12c goes to the engine checks, each pre-registered before it runs: census and numerics under
  MTP, interleaved MTP A/B at fill 0 and 28K, and GSM8K-500 MTP for T4. If X1 or X2 fails, it is a bug, fixed
  before anything else. If X3 fails, W12c stops at the microbenchmark.

## Final speed of record: results (2026-10-31 01:38-02:10 EDT, step `x-final1`; `ab/final1.md`, rows `ab/final1.jsonl`)

The first attempt (01:33) was stopped by the desktop's GPU guard at 01:35: Chrome's on-device model took the GPU.
The queue restarted the step from round 1 at 01:38. The six rows the stopped attempt wrote (stock-0 and adopt-0,
round 1) are dropped (`ab/final1.clean.jsonl`). Including them changes no per-round figure in the report. Decode tok/s,
three rotated rounds, cache 32768:

| | stock 1.6.5 | adopted (w11b) | per round | mean |
|---|---|---|---|---|
| undrafted, fill 0 | 23.40 | 42.33 | +72.6%, +92.9%, +77.7% | **1.81x** |
| undrafted, fill 28000 | 18.78 | 40.01 | +105.6%, +106.3%, +129.5% | **2.13x** |
| MTP, fill 0 | 46.84 | 53.24 | +12.6%, +13.0%, +15.7% | **1.14x** |
| MTP, fill 28000 | 36.78 | 50.77 | +38.3%, +34.8%, +41.1% | **1.38x** |

**S1-S4 all hold, in every round:**
- S1: at least 1.6x; 1.73-1.93x measured.
- S2: at least 1.8x; 2.06-2.30x measured.
- S3: faster in every round, mean 1.14x (at least 1.05x required).
- S4: faster in every round, mean 1.38x (at least 1.3x required).

These are the speeds of the adopted configuration that the paper quotes. On this desktop's 2080 Ti, undrafted decode
at 28K context now runs faster than stock at short context.

## W12b limiter study: results (2026-10-31 02:10 EDT, step `x-w12-sqmb1`; `w12/sqmb1.jsonl`, `w12/sqmb1.log`)

Every variant runs at 58-64 registers and 4 blocks per SM. Grid 272. Slice height from the sq rule (80 rows for the
MLPs, 48 for GDN qkv, 64 for attention q). Microseconds per launch, single timings (a graph of 20 launches, 30
replays):

| V | variant | MLP gate | MLP down | GDN qkv | attn q |
|---|---|---|---|---|---|
| 0 | upstream narrow | 64.63 | 63.40 | 47.67 | 39.03 |
| 1 | upstream staged | 76.60 | 76.11 | 51.81 | 47.11 |
| 2 | narrow + PRMT | 48.74 | 71.26 | 37.29 | 38.67 |
| 3 | staged + PRMT | 66.99 | 81.46 | 49.23 | 52.77 |
| 4 | no loads (compute only) | 37.69 | 41.92 | 27.49 | 29.81 |
| 5 | no dp4a | 49.96 | 55.22 | 37.59 | 37.89 |
| 6 | no hash multiply | 58.59 | 56.39 | 42.00 | 38.23 |
| 7 | loads only | 40.92 | 44.33 | 33.31 | 31.30 |
| 8 | running pointers | 45.94 | 65.22 | 44.20 | 35.74 |
| 9 | pointers + PRMT | 49.14 | 66.59 | 36.69 | 37.26 |
| | memory floor at 616 GB/s | 36.17 | 36.17 | 21.28 | 25.53 |

**The rules:**
- **C0 holds:** V1-V3, V8 and V9 equal V0's partials bit for bit on every shape.
- **L1 misses by a hair.** V4 is 24-42% faster than V0 on every shape. V7 is 30-37% faster on three shapes and 19.8%
  on attention q, against the 20% required.
- **L2 fails:** the staged unit is 9-21% slower than narrow on every shape.
- **L3 fails:** PRMT is 22-25% faster on MLP gate and GDN qkv, 12% slower on MLP down, and level on attention q.
- **L4:** removing the dp4a saves 3-23% and removing the hash multiply 2-12%. Both exceed 5% on most shapes, so the
  FMA pipe holds part of the time.
- **L5 fails:** running pointers are 7-29% faster on three shapes and 2.9% slower on MLP down.

**What the numbers support, and what they do not:**
- **Measurement noise is the first finding.** These are single timings. The MLP gate sweep moves V0 between 53 and
  64 us across slice heights of 16-160 rows, with no trend. W12c's same-code M = 1 pairs (below) differ by up to
  13%. Rules at 3-5% cannot be decided this way. The next microbenchmark repeats each variant in rotated order and
  takes medians.
- **The robust result is the overlap gap.** On MLP gate:
  - loads alone take 41-48 us (about 80-88% of peak bandwidth);
  - arithmetic alone takes 36-51 us;
  - together they take 53-65 us.

  Each alone sits near the 36 us floor. The kernel loses its time because the two do not overlap, not to either
  one. W12b's target is therefore overlap: prefetch the next rows' words into registers while the current row
  computes. Upstream's cp.async staging is the other overlap mechanism, and it is slower on Turing.
- **Trimming instructions** (PRMT, running pointers) gives shape-dependent results within the noise. No single change
  is a clear win.
- **The slice-height sweep** shows only that 320-row slices are bad (one wave, 2x slower). Between 16 and 160 rows,
  nothing is resolved.

## W12c: results, and the decision (2026-10-31 02:12 EDT; `w12/sqmb1.jsonl` M-row part, `w12/w12c_check1.jsonl`, `w12/w12c_bench_{fp16,imma}.jsonl`)

**Microbenchmark, IMMA against dp4a at M rows (K = 2):**

| M | MLP gate dp4a / IMMA | MLP down | GDN qkv |
|---|---|---|---|
| 1 | 61.0 / 59.2 (1.03x) | 59.3 / 61.2 (0.97x) | 37.3 / 38.3 (0.98x) |
| 2 | 98.5 / 71.2 (1.38x) | 99.3 / 65.0 (1.53x) | 54.6 / 39.8 (1.37x) |
| 3 | 104.4 / 66.0 (1.58x) | 104.9 / 65.3 (1.61x) | 63.0 / 42.1 (1.50x) |
| 4 | 120.8 / 73.9 (1.64x) | 110.9 / 70.1 (1.58x) | 76.3 / 45.8 (1.67x) |
| 6 | 150.7 / 83.2 (1.81x) | 155.3 / 82.5 (1.88x) | 88.5 / 53.9 (1.64x) |
| 8 | 222.2 / 112.3 (1.98x) | 258.2 / 121.3 (2.13x) | 108.2 / 64.1 (1.69x) |

- **C0:** bit-identical at M = 1-6 on all three shapes. At M = 8 the dp4a instance holds 80 registers and fits 3
  blocks per SM against IMMA's 4, so the sq rule picks a different slice height (112 against 80 rows). The partials
  then differ in layout, and M = 8 is untested rather than failed.
- **R-rule:**
  - its first clause holds: IMMA at M = 4 is 1.58-1.67x faster than dp4a;
  - its second clause fails: IMMA at M = 8 costs 1.68-2.13x its M = 1 time, against 1.3x allowed. Rows are not free.

**Engine, `x-w12c-check1`:**
- **X1 fails as written:** 124 of 140 cases are bit-identical. All 16 misses are at M = 3-4, with a relative
  difference of ~0.7%. Three things point to the cause:
  - w11b with `MAX_M=4` differs from its own fp16 path in all 40 M = 3-4 cases, so it ran int8 dp4a, not an fp16
    fallback;
  - in the microbenchmark, IMMA equals dp4a whenever both use the same slices;
  - the sq kernel quantizes each k-slice with its own scale, and in the engine each build's W9 fit picks the slice
    height from its own instance's occupancy.

  The likely cause is that the two builds chose different slice heights for those 16 shapes. That is unverified: a
  fit-log run of the 16 cases would settle it, and it is queued (`x-w12c-fitlog`).
- **X2 holds:** finite everywhere. Against fp16 at M = 3-7, the largest relative difference is 0.74-0.95% (median
  per M), the size of the int8 activation rounding.
- **X3 fails.** fp16 time / IMMA time, K = 2:

| shape | M = 3 | 4 | 5 | 6 | 7 | median |
|---|---|---|---|---|---|---|
| MLP gate | 1.19 | 1.03 | 0.94 | 0.66 | 0.77 | 0.94 |
| MLP down | 1.88 | 1.51 | 1.37 | 1.37 | 1.05 | 1.37 |
| GDN qkv | 1.65 | 1.08 | 1.06 | 0.86 | 0.75 | 1.06 |

  The same-code M = 1 pairs vary 0.87-1.04, so that is this timing's noise. Beyond it, IMMA wins at M = 3 and loses
  at M = 6-7 on most shapes (GDN z, GDN out, attention q too). Attention k/v loses at every M.

**Decision: W12c stops here, as pre-registered.** Both the R-rule's scaling clause and X3 fail. The overlay
`exl3-w12c` is not used further.

**Why it loses at high M, and what that points to.** In the engine, every (slice, 256-column) unit re-stages its slice
for all M rows: Hadamard transform, max, quantize and splat per row. For an MLP matrix the same slice is staged 68
times over, once per column group. At M = 7 that is 7x the staging, and it outgrows what IMMA saves on the products.
The microbenchmark copied ready splats, so it never paid this. A shared staging pass would put it back to once per
slice: a small kernel, or the cooperative path's grid-wide phase, quantizing the M rows into global splats that every
unit reads. That is a separate design (W12d). It needs its own microbenchmark with real staging before any
pre-registration.

## W12b round 2: register prefetch, repeated timings (pre-registered 2026-10-31 02:19 EDT, before any timing of it)

Step `x-w12-sqmb2`, queued after `x-w12c-fitlog`, runs `w12_sqmb.py --part repeat --rounds 7`. It addresses the two
sqmb1 findings: overlap, and noise.
- **New variants:**
  - **V10, V11:** running pointers plus a register ring that issues each row's four word loads 1 or 2 rows before its
    arithmetic.
  - **V12:** V11 + PRMT.
  - Registers: V0 63, V8 61, V10 63, V11 64, V12 64. All keep 4 blocks per SM.
- **Timing:** each shape times V0, V2, V4, V7, V8, V9, V10, V11 and V12 once per round, in an order rotated every
  round, for 7 rounds. The medians decide, and min/max are reported.

**Rules:**
- **C0:** V2 and V8-V12 equal V0's partials bit for bit on every shape.
- **P1, prefetch:** V10 or V11 qualifies if, on all four shapes, its median is at least 8% below V0's and its slowest
  round is faster than V0's median. A qualifying variant becomes **W12b-pf**, the same prefetch in the engine's narrow
  unit. W12b-pf gets its own pre-registration (exactness, census, interleaved A/B, T3/T4) before any build.
- **P2:** PRMT joins W12b-pf if V12's median is at least 3% below V11's on all four shapes.
- **P3 (reported):** V8 and V9 against V0 by median. This settles sqmb1's single-timing L5 and L3 readings.
- **P4 (reported):** V4 and V7 medians against the memory floor (the overlap gap measured again).

## L1: energy per token from the speed of record (pre-registered 2026-10-31 04:09 EDT, before any energy number was computed)

This is the first of the lessons carried over from the laptop papers (Paper 7: tokens per joule from board power).
No new run is needed. The desktop already logs board power, SM clock and throttle reasons at 1 Hz
(`$L/logs/gpu_telemetry.csv`, from `nvidia-smi`). `x-final1` ran its eight arms against that log.

**Already seen before this was written.** Two stock rows' end-of-row snapshots: 256 W and throttle flag `0x4`
(software power cap). They suggest the card decodes at its power limit. EJ1 is written knowing this.

**Method** (`tools/turing/energy.py`):
- Each `final1.clean` row covers the window [row time - `decode_wall_s`, row time].
- Joules = mean board power of the telemetry samples in the window x `decode_wall_s`.
- Tokens per joule = `new_tokens` / joules. Speed = `new_tokens` / `decode_wall_s`, from the same rows.
- Per arm: the median over its 9 rows (3 tasks x 3 rounds). Rows with fewer than 3 samples are flagged and kept.
- The telemetry slice is copied to `testsuite/results/turing/ab/final1_telemetry.csv`.

**Predictions:**
- **EJ1.** In every one of the eight arms, mean decode power is at least 235 W, and the power-cap flag (`0x4`) is
  set in at least half the samples. The card decodes at its power limit whichever build runs.
- **EJ2.** In each of the four conditions (undrafted/MTP x fill 0/28000), adopted-over-stock tokens per joule is
  within ±10% of the same condition's adopted-over-stock speed ratio, since power is pinned.
- **EJ3 (reported).** Median tokens per joule per arm, beside Paper 7's laptop figure (0.559 tokens/J, EXL3 drafted,
  80 W cap).
- **EJ4 (reported).** Median SM clock per arm. Under a fixed power cap, a busier kernel can run at a lower clock.

**If EJ1 holds**, the power limit is a speed lever on this card, as the 80 W cap was on the laptop. Raising it needs
Matt's sudo (`nvidia-smi -pl`). That test gets its own pre-registration.

### L1: results (2026-10-31 04:11 EDT; `ab/final1_energy.md`, `ab/final1_energy.json`, telemetry `ab/final1_telemetry.csv`)

| arm | tok/s (row median) | mean W | tokens/J | SM MHz (median) | power-cap share |
|---|---|---|---|---|---|
| stock, undrafted, fill 0 | 23.48 | 256.3 | 0.0922 | 1785 | 100% |
| adopted, undrafted, fill 0 | 42.60 | 253.7 | 0.1687 | 1680 | 100% |
| stock, undrafted, fill 28000 | 19.35 | 256.3 | 0.0759 | 1785 | 100% |
| adopted, undrafted, fill 28000 | 39.84 | 253.6 | 0.1570 | 1650 | 100% |
| stock, MTP, fill 0 | 49.85 | 253.1 | 0.2004 | 1785 | 100% |
| adopted, MTP, fill 0 | 56.70 | 253.3 | 0.2250 | 1740 | 100% |
| stock, MTP, fill 28000 | 41.06 | 256.3 | 0.1600 | 1800 | 100% |
| adopted, MTP, fill 28000 | 57.11 | 251.2 | 0.2270 | 1725 | 100% |

The tok/s column is the median over an arm's 9 rows. The speed of record quotes per-round means over the three
prompts, which is lower for MTP because prose drafts worst. No row had a short telemetry window.

- **EJ1 holds.** Every arm decodes at 251-256 W, and every sample in every window has the power-cap flag set. The
  card's limit is 260 W (default; `nvidia-smi` reports a 320 W maximum). So decode on this card is power-limited
  whichever build runs, as the laptop was at 80 W.
- **EJ2 holds.** Adopted-over-stock tokens per joule tracks the speed ratio within ±2% in all four conditions:
  1.83x vs 1.81x, 2.07x vs 2.06x, 1.12x vs 1.14x, 1.42x vs 1.39x. With power pinned, every speedup in this
  campaign was also an equal energy saving.
- **EJ3.** The adopted build gives 0.16-0.17 tokens/J undrafted and about 0.23 with MTP. Paper 7's laptop gave
  0.559 drafted at its 80 W cap. The 2080 Ti spends about 2.5x the energy per token, as expected from 12 nm against
  4 nm silicon at three times the power.
- **EJ4.** The adopted build runs 3-8% lower SM clocks than stock under the same cap: 1650-1740 MHz against
  1785-1800. The int8 path does more arithmetic per byte, so it draws more power per clock, and the cap takes it
  out of the clock. W12b found that the int8 loop is limited by per-weight work, so its speed should follow the SM
  clock.
- **Memory clock.** All arms read 6,800 MHz against the card's 7,000 MHz maximum. That is the P2 state CUDA
  processes get on GeForce boards, about 3% below full memory bandwidth.

**What this opens (L2, needs Matt's sudo; gets its own pre-registration before any run):** raising the power limit
from 260 W to 320 W with `nvidia-smi -pl 320`. It is reversible with `-pl 260` and resets at reboot. The expected
gain is largest for the adopted int8 path, which is the most clock-starved. The P2 memory clock is a second, smaller
lever, if the driver allows locking it on Turing.

## L3: the overthinking-marker penalty on the 2080 Ti (pre-registered 2026-10-31 04:14 EDT, before either run)

This lesson comes from the laptop's efficiency campaign: ideas ledger rows 18 and 34, H36/E45. There, a logit bias of
-0.5 on 49 "overthinking marker" tokens ("Wait", "Alternatively", ...) cut about 5% of completion tokens and about
10% of wall time on the EXL3 lane, with no accuracy change at n = 100. The bias is applied per request, so no
engine change is involved. The file is `serving/logit_bias_marker_penalty_-0.5.json`, copied to the desktop as
`$L/tools/quality/logit_bias_marker_penalty_-0.5.json`. The token ids are the same because the tokenizer is the
same.

**Change to the runner.** `run_quality.sh` gains `TURING_EXTRA_BODY`: a JSON file whose fields are merged into each
request body through `run_eval_turing.py --extra-body-file`. The file's path and sha256 go into the run record. With
it unset, the request bodies are exactly as before. The desktop copy was swapped in by rename, because a run of the
old file was in progress.

**Runs** (queue rev 31, after `x-w12-sqmb2`), on the adopted configuration, GSM8K-500 with E47's flags:
- `x-q-exl3-nodraft-w11-mp05`: label `TURING-exl3-nodraft-w11-mp05`, paired with `TURING-exl3-nodraft-w11`
  (same build and switches, no penalty).
- `x-q-exl3-mtp-w11j2-mp05`: label `TURING-exl3-mtp-w11j2-mp05`, paired with `TURING-exl3-mtp-w11j2`.

**Predictions, per mode:**
- **M1 (length).** Total completion tokens over the 500 items are at least 3% below the paired unpenalized run, and
  the median per-item difference is below zero.
- **M2 (accuracy, T4's rule).** Accuracy is no more than 1.0 point below the paired run, and McNemar on the
  discordant items is not significant (p >= 0.05).
- **M3 (reported).** Run wall time, mean decode tok/s, and the share of items that hit the 4,096-token cap.

**Decision.** If M1 and M2 both hold in a mode, the penalty becomes part of that mode's adopted serving
configuration. It is a request default, not an engine change, and the speed of record is unaffected.

**Caveat.** The unpenalized runs are from 2026-10-30, a day earlier, on the same build. Greedy decoding in this
engine is not bit-reproducible on this model (E49; upstream #353), so the pairs include rerun noise. T4's rule was
set with that noise in mind.

## L2: the power limit, 260 W to 320 W (pre-registered 2026-10-31 04:15 EDT; needs Matt's sudo, nothing has run)

This lesson comes from the laptop: its 80 W cap was the largest single lever on local-model speed. L1 showed this
card decodes at its 260 W limit in every arm, with the power-cap flag set in 100% of samples. The adopted build runs
3-8% lower SM clocks than stock.

**Matt's step, one of two forms:**
- **(a) Once:** on the desktop, `sudo nvidia-smi -pl 320`. It is undone with `sudo nvidia-smi -pl 260`, and it
  resets at reboot.
- **(b) A narrow sudoers rule** letting the lab user run only `/usr/bin/nvidia-smi -pl *` without a password. The
  board itself bounds the value to 100-320 W. This lets the queue switch limits per arm, so the comparison is
  interleaved like every other A/B here. Form (b) is preferred for that reason; it is Matt's call.

**Run** (`x-power1`, `ab_speed.sh power1 3`):
- The same eight arms and prompts as `x-final1`.
- Under (b), each arm runs at both 260 W and 320 W, in rotated order.
- Under (a), it runs at 320 W only and is compared with `x-final1`'s rows. That comparison is not interleaved, and
  the note will say so.
- L1's telemetry method gives power, SM clock, memory clock, temperature and tokens per joule.

**Predictions:**
- **PW1.** Adopted undrafted at fill 0 and fill 28000 is at least 4% faster at 320 W than at 260 W, in every round.
- **PW2.** The gain is larger for adopted than for stock, undrafted at fill 0. The int8 path is the most clock-starved.
- **PW3.** Median SM clock rises at least 80 MHz in every adopted arm. Mean power stays at or below 320 W.
- **PW4 (reported).** Tokens per joule at 320 W against 260 W. It is expected to fall, as clocks rise faster than
  speed.
- **Safety.** The guard stops any run at 87 °C (`TURING_MAX_TEMP_C`). The night's peak at 260 W was 78 °C with the
  fan at 57%. If any arm reaches 87 °C, the 320 W setting is not adopted, whatever the speed.

**Also tried in the same sudo session (reported only):** `nvidia-smi -lmc 7000,7000`, to lift the P2 memory clock
from 6,800 to 7,000 MHz. It may be unsupported on Turing GeForce, and either answer is recorded.

### L2: amendment, form (b) installed (recorded 2026-10-31 04:32 EDT, before any power run)

- **Rule installed.** At 04:31 EDT Matt installed `/etc/sudoers.d/turing-power-limit` on the desktop. It allows
  exactly `nvidia-smi -pl 260`, `-pl 290` and `-pl 320` without a password: no wildcards, no other commands. The line
  was checked first with the desktop's own `visudo` (sudo-rs 0.2.13), which reported "parsed OK".
- **Tested.** Setting the current value with `sudo -n` (260 to 260 W) worked from a non-interactive session, so it
  works inside queue units too.
- **The run is interleaved.** `ab_speed.sh` gains `TURING_POWER_W=N` per arm: it sets the limit right before that
  arm. Arms without the variable run at 260 W, and an EXIT trap restores 260 W however the script ends, including a
  guard stop.
- **Insurance.** Every queued step after `x-power1` starts with `sudo -n nvidia-smi -pl 260`, in case a stop ever
  bypasses the trap.
- **Arms.** The eight `x-final1` configurations at 260, 290 and 320 W: 24 arms, three rotated rounds, about 1.5 h.
  The ordering puts a different power limit on every consecutive arm.
- **The 290 W arms are added and reported only (PW5)**, as a midpoint for the dose response. PW1-PW4 compare 320 W
  with 260 W as pre-registered.
- **Analysis.** `energy.py` per arm on the step's own telemetry slice, and `ab_report.py` with pairs `X-p260:X-p320`
  for every configuration X.

## Corrections from the 2026-10-31 related-work search (recorded 2026-10-31 04:46 EDT; the text above is kept as written)

The full search, its sources and verdicts are in `research/TURING_RELATED_WORK_2026-10-31.md`. The raw results are
in `research/raw/workflows/2026-10-31-p8-novelty-search-wf_882443ca-7fe/`.

- **W7's "The int8 fit does not [have prior art], as far as these sources show" is wrong.** Three projects ran this
  kernel class on 64 KB Turing parts before us:
  - weicj/exllamav3-turing commit 6f90340 (2026-09-06), with a compile-time 64 KiB budget shared by host and device;
  - vcruz305/exl3-turing (2026-09-02), with a host-only rows_max clamp, on a base from before the 80 KB gate existed;
  - spiritbuun/buun-llama-cpp (2026-09-06 to 09-18), an EXL3 int8/dp4a GEMV in a llama.cpp fork on SM75, with a
    per-device cap from the opt-in limit minus static memory.
  Our mirrors missed weicj because it is a separate repository, not a fork. What stays ours is the runtime
  per-instance fit in current upstream, and the measured int8-vs-fp16 result on a stock 11 GB card.
- **W10b's "upstream's own m = 2 rule, active on Turing for the first time" is wrong.** vcruz305's MTP setup
  (draft 1, so 2-row verify) very likely ran the 2-row int8 kernel on a 64 KB Quadro.
- **"exllamav3's dev branch at f8807350 from 2026-10-29" is wrong.** f8807350 is a 2026-10-04 ROCm commit, included
  from v1.6.0. The 2026-10-29 dev tip was a0932069; dev is now b658e643 (2026-10-30), which still declines under
  80 KB.
- **W11 (fd75d) cites these related kernels:**
  - LMDeploy TurboMind (2024: sm_75 decode over int8/int4 KV with the same S^T orientation and movmatrix);
  - rafatxf's fdq4 (2026-09-27: sm_75 tensor cores, register-direct from exllamav3's 4-bit cache);
  - kyojin's qdec (2026-10-23, sm_80+, split-phase drop-in over exllamav3's caches);
  - cookjohn/vllm-turing fdec.cu (2026-10-23, sm_75, FP8 KV, Qwen3.8-27B).
  None of them was known when W11 was designed. fd75d remains, to our knowledge, the first tensor-core decode kernel
  for exllamav3's 8-bit cache on Turing.

## R1: head-to-head with rafatxf's Turing branch, to learn what to borrow (pre-registered 2026-10-31 04:49 EDT, build running, nothing timed)

This follows from the related-work search (`research/TURING_RELATED_WORK_2026-10-31.md`). Its skeptics asked for this
run before any "fastest" wording, and Matt's direction is to borrow good ideas with credit.

- **Source.** rafatxf/exllamav3, branch `turing` at 15654d3 (2026-09-30, v1.5.3 base, MIT, Copyright Turboderp). It
  was archived from our mirror as `exllamav3-rafatxf-turing-15654d3.tar.gz` (sha256 2d18adc0…).
- **Build.** Unpatched, built as overlay `exl3-rafatxf` (`1.5.3+rafatxf`) by `build_exl3.sh`, which now takes
  `EXL3_SRC_TAR` / `EXL3_SRC_LABEL`. It runs in a capped CPU unit, `turing-build-rafatxf`.
- **The branch's sm_75 paths are all on by default:**
  - fdq4: tensor-core decode attention straight from a 4-bit cache, with cubins compiled by nvcc at first use. Our
    runs give it `CUDA_HOME` = the lab toolchain and `XDG_CACHE_HOME=$L/cache`, so the cubins stay on ssd2.
  - CUDA kernels for the gated-delta-rule layers (GDN_REC75, GDN_H/WY/O);
  - fp16-accumulate reconstruct GEMMs;
  - unfused multi-projections;
  - a split-K GEMV.

**Step `x-rafatxf1`** (queue rev 33, after `x-power1`):
1. Wait for the overlay.
2. Check the import with `with_engine.sh`.
3. Run `smoke.py`: Q4 cache, the three smoke prompts, plus a 28K prefill. Each reply's first 160 characters are
   recorded.
4. Run `ab_speed.sh rafatxf1 3` with 16 arms, three rotated rounds: (ours adopted at Q8, ours adopted at Q4,
   rafatxf at Q4, rafatxf at Q8) x (undrafted, MTP 6 dynamic) x (fill 0, fill 28000). Cache size is 32768 for all.

**Step `x-census-raf`:** nsys censuses of rafatxf at Q4, undrafted, at fill 0 and fill 28000. Its per-kernel time
goes against our `d0-f1` / `d28-w11` censuses, to see which of its components beats ours.

**Gate.** If any smoke reply is not readable English or code, the speed rows are reported as invalid, not compared.

**Predictions:**
- **R1a.** Undrafted at fill 0, ours (adopted, Q8) beats rafatxf (Q4, its defaults) by at least 5% in every round.
  At 2.0 bpw our int8 GEMV should beat an fp16 split-K GEMV.
- **R1b.** At fill 28000, rafatxf at Q4 beats rafatxf at Q8 by at least 20% (fdq4 against the Triton path), in
  every round.
- **R1c (reported, no direction).**
  - Ours against rafatxf undrafted at 28000, where fdq4 reads half the cache bytes fd75d reads.
  - Both MTP comparisons.
  - Ours at Q4 against ours at Q8: fd75d does not cover Q4, so ours at Q4 falls back to Triton.
- **R1d (reported).** The census's per-kernel families: GDN step, GEMV, attention, norms.

**Decision.** None. This run is diagnostic. Any component we borrow gets its own pre-registration and the usual
exactness and quality gates, with the source credited.

### R1: the rafatxf overlay is built (recorded 2026-10-31 05:10 EDT, nothing timed)

- **Build.** `turing-build-rafatxf` finished at 05:05 EDT: wheel `exllamav3-1.5.3+rafatxf` (sha256 a57c28c2…), 164
  sm_75 ELF images, overlay `$L/overlay/exl3-rafatxf`. Log: `$L/logs/build-rafatxf.log`.
- **Checks without the GPU.** `with_engine.sh` imports it from the overlay. The extension has `fa75_fwd_win` and
  `fdq4_decode`, the check the branch's own README gives.
- **Deviations from the branch's documented setup:**
  - torch 2.10.0+cu128 and the lab's CUDA 12.5 nvcc with gcc-13, where the branch documents torch 2.9.1 and CUDA 13.0;
  - our 2.0 bpw model, where the branch used 4.0 bpw;
  - a stock 11 GB card, where the branch used a 22 GB card.
  The step's smoke gate exists to catch a bad build.

## W12c fit log and W12b round 2: results (recorded 2026-10-31 06:42 EDT; steps `x-w12c-fitlog` and `x-w12-sqmb2`, which ended 2026-10-31 06:37 and 06:38 EDT; `w12/w12c_fitlog_{dp4a,imma}.txt`, `w12/sqmb2.jsonl`, `w12/run-x-w12-sqmb2.log`)

### W12c: the 16 non-identical cases are the 16 different slice choices

Both builds were rerun with `EXL3_INT8_SQ_MAX_M=4` and the fit log on (80 logged shapes: 5 matrices x K = 1-4 x M = 1-4).
Matched against X1's 140 rows (`w12c_dp4a_m4.pt` against `w12c_imma_m4.pt` in `w12c_check1.jsonl`):

- The host picks a different slice height in **16** of the 80 shapes. All 16 are at M = 3-4, because the IMMA instance's
  occupancy differs from the dp4a instance's at those row counts.
- **All 16 X1 misses are in those 16 shapes, and those 16 shapes are all misses.** No miss has the same slice height.
- In 10 more shapes, only the grid or k-split differs and the slice height is the same. All 10 are bit-identical.

So W12c's IMMA unit is exact. The X1 failure came from each build's fit choosing its slices from its own occupancy, and
the sq kernel quantizes activations per slice. The likely cause named at 02:12 is now confirmed. This does not reopen
W12c: it stopped on speed (the R-rule's scaling clause and X3), not on exactness. For any later int8 change, the
exactness check must pin the slice height, or compare against the same build's fp16 path, as X2 did.

### W12b round 2: register prefetch does not help; the medians replace sqmb1's single readings

Seven rounds, rotated order, medians in microseconds per launch, with the round minimum and maximum in brackets:

| V | variant | MLP gate | MLP down | GDN qkv | attn q |
|---|---|---|---|---|---|
| 0 | upstream narrow | 53.22 [52.39, 77.99] | 63.91 [59.89, 81.44] | 47.11 [38.37, 58.97] | 47.77 [45.00, 55.32] |
| 2 | narrow + PRMT | 51.85 (0.974x) | 64.01 (1.002x) | 44.75 (0.950x) | 47.01 (0.984x) |
| 4 | no loads (compute only) | 38.54 (0.724x) | 44.36 (0.694x) | 32.94 (0.699x) | 34.14 (0.715x) |
| 7 | loads only | 42.75 (0.803x) | 44.53 (0.697x) | 34.29 (0.728x) | 38.32 (0.802x) |
| 8 | running pointers | 47.98 (0.902x) | 58.68 (0.918x) | 47.06 (0.999x) | 43.41 (0.909x) |
| 9 | pointers + PRMT | 50.03 (0.940x) | 60.55 (0.948x) | 48.29 (1.025x) | 45.69 (0.956x) |
| 10 | prefetch 1 row | 53.30 (1.002x) | 66.73 (1.044x) | 45.40 (0.964x) | 48.76 (1.021x) |
| 11 | prefetch 2 rows | 54.00 (1.015x) | 64.69 (1.012x) | 40.46 (0.859x) | 50.45 (1.056x) |
| 12 | prefetch 2 + PRMT | 52.96 (0.995x) | 66.00 (1.033x) | 45.80 (0.972x) | 49.81 (1.043x) |
| | memory floor at 616 GB/s | 36.17 | 36.17 | 21.28 | 25.53 |

**The rules:**
- **C0 holds:** V2 and V8-V12 equal V0's partials bit for bit on every shape.
- **P1 fails.** V10 runs at 0.96-1.04x V0 and V11 at 0.86-1.06x. Only V11 on GDN qkv is 8% or more faster, and its
  slowest round (54.00) is slower than V0's median there. **No W12b-pf.**
- **P2 fails** (and is moot): V12 against V11 is 0.981x, 1.020x, 1.132x and 0.987x.
- **P3 (reported):** running pointers (V8) are 8-10% faster on MLP gate, MLP down and attention q, and level on GDN qkv.
  - That corrects sqmb1's MLP down reading (2.9% slower in one timing) and its GDN qkv reading (7% faster).
  - L5's rule (5% or more on every shape) still fails, on GDN qkv.
  - PRMT alone (V2) is within 5% everywhere. Its 22-25% gains in sqmb1 were noise.
- **P4 (reported): the overlap gap, measured again.**
  - Each part alone nears its floor. Loads alone (V7) reach 81-85% of peak bandwidth on the MLPs and 62-67% on GDN qkv
    and attention q. Arithmetic alone (V4) takes 0.69-0.72x V0's time.
  - Together, V0 takes 1.25-1.44x the slower of the two. It reaches 68% of peak on MLP gate, 57% on MLP down, 45% on
    GDN qkv and 53% on attention q.
  - The share of the shorter part hidden under the longer: 73% (MLP gate), 56% (MLP down), 61% (GDN qkv), 72% (attn q).

**What this means.** Issuing each row's loads one or two rows ahead did not shorten the kernel on three of four shapes.
So the part that fails to overlap is probably not load latency, which a register ring would hide. Two differences
between V0 and V7 remain candidates:
- V0 issues four 4-byte loads per row per lane, while V7 uses 16-byte loads;
- V0 computes each row's addresses, which V8's 8-10% gain shows is real work.

Both point to the loads' issue cost competing with the arithmetic for the same issue slots. The next scouting
microbenchmark (W12e, not pre-registered yet) tests that directly:
- V0's arithmetic fed by 16-byte loads, through a repacked tile layout or register shuffles;
- running pointers included;
- the same 7-round rotated medians.

Nothing goes into the engine from round 2.

## L2: results (recorded 2026-10-31 08:25 EDT; step `x-power1`, 2026-10-31 06:38-08:17 EDT; `ab/power1.md`, `ab/power1_energy.md`, `ab/power1_energy.json`, rows `ab/power1.jsonl`, telemetry `ab/power1_telemetry.csv`)

24 arms (8 configurations x 260/290/320 W), 3 rotated rounds, 216 rows, no short telemetry windows. The limit went
back to 260 W at exit, as designed. Speed changes are `ab_report.py` round means. Energy uses `energy.py` per-arm
medians. The two tools aggregate differently, so their ratios differ slightly.

**320 W against 260 W:**

| configuration | tok/s change per round | mean | tokens/J ratio | median SM clock change | power-cap share at 320 W |
|---|---|---|---|---|---|
| stock, fill 0 | -6.8 / +1.4 / +4.4% | -0.3% | 0.808 | +60 MHz | 79% |
| stock, 28K | -1.1 / -2.8 / -3.3% | -2.4% | 0.885 | +45 MHz | 21% |
| stock MTP, fill 0 | +9.9 / +14.8 / -2.2% | +7.5% | 0.859 | +82 MHz | 95% |
| stock MTP, 28K | +5.4 / +0.3 / -0.1% | +1.9% | 0.871 | +45 MHz | 42% |
| adopted, fill 0 | +12.1 / +0.2 / +6.2% | +6.2% | 0.806 | +120 MHz | 75% |
| adopted, 28K | +5.0 / +4.3 / +7.1% | +5.5% | 0.857 | +75 MHz | 32% |
| adopted MTP, fill 0 | +17.4 / +4.6 / -3.7% | +6.1% | 0.977 | +113 MHz | 92% |
| adopted MTP, 28K | -3.8 / +7.9 / +2.6% | +2.2% | 0.880 | +68 MHz | 45% |

**The rules:**
- **PW1 fails.** Adopted at 28K is at least 4% faster in every round (+5.0, +4.3, +7.1%). Adopted at fill 0 misses in
  round 2 (+0.2%).
- **PW2 holds.** Undrafted at fill 0, adopted gains +6.2% against stock's -0.3%. The int8 path is the one that
  responds to clock.
- **PW3 fails.** The median SM clock rises 120 and 113 MHz in the adopted fill-0 arms, but only 75 and 68 MHz at 28K.
  Mean power stays at or below 320 W (at most 313 W).
- **PW4: tokens per joule falls in every configuration**, by 2-19% (ratios 0.806-0.977).
- **PW5 (290 W, reported):** speed changes of -2.4% to +9.1% in round means, within this run's round-to-round spread.
  Tokens per joule falls 11-20%, except adopted MTP at fill 0 (1.07).
- **Safety:** the peak was 85 °C with the fan at 65%, against the 87 °C stop. No arm was stopped.
- The memory clock stayed at 6,800 MHz in every arm. `-lmc` was not tried, since the sudoers rule allows only `-pl`.

**Decision: stay at 260 W.**
- 320 W buys 0-6% speed in round means, with rounds spread as wide as the effect.
- It costs 2-19% in tokens per joule, and peaks at 85 °C, 2 °C from the stop.
- At 320 W the card stops being power-capped much of the time (21-95% of samples, against 100% at 260 W).
  - The SM clock rises only 2-7%, and speed rises less than the clock.
  - So once power is lifted, the remaining limit is the fixed 6,800 MHz memory clock and the kernels' memory path
    (W12b's overlap gap), not SM clock.

**What it points to:**
- On this card, decode speed responds weakly to SM clock, while energy grows steeply with it.
- The efficient point is therefore probably below 260 W. That is the L4 efficiency study: 200-260 W and SM clock locks
  of 1350-1650 MHz, measuring tokens/s, tokens/J and peak °C.
- L4 needs Matt's extended sudoers line (`research/LAB-POWER-HEAT-COST-2026-10-31.md`). It gets its own
  pre-registration before any run.

## W12e: one coalesced word per lane plus warp shuffles (pre-registered 2026-10-31 08:32 EDT, before any build or timing)

sqmb2 left the overlap gap unexplained by load latency, since register prefetch did not help. The narrow unit's load
pattern points at the loads' issue cost. Lane L reads words i0 and i1 of both tiles (i1 = L >> 1, i0 = i1 - 1 mod 16).
That is 4 loads per row per lane: 128 lane-loads for the tile pair's 32 words, so each word is read by 4 lanes. V7,
the loads-only floor, moves the same bytes with 8 lanes x one 16-byte load.

**New variants** (`tools/turing/w12/sqmb.cu`, `mb_unit_shfl`), each with a running pointer:
- **V13:** lane L loads only word L of the pair (tile A in lanes 0-15, tile B in lanes 16-31). That is one 128-byte
  transaction per warp, one load per lane per row. Each lane then takes its four words from the owning lanes with
  `__shfl_sync`.
- **V14, V15:** V13 with each row's single load issued 1 or 2 rows ahead.
- The words are identical to V0's, so the partials must be too.

**Run:** step `x-w12-e1`, `w12_sqmb.py --part repeat --rounds 7 --variants 0,4,7,8,13,14,15`. It uses sqmb2's
shapes, grid and slice rule, rotated order and medians. It goes in the queue after `x-census-raf`.

**Rules:**
- **C0:** V13-V15 equal V0's partials bit for bit on all four shapes. A variant that differs is a bug and is not
  timed further.
- **E1 (decision):** a variant among V13-V15 qualifies if two things hold on all four shapes:
  - its median is at least 8% below V8's median (V8, running pointers, is sqmb2's best exact variant);
  - its slowest round is faster than V8's median.

  A qualifying variant becomes **W12e-eng**, the same load scheme in the engine's narrow unit. W12e-eng gets its own
  pre-registration (exactness, census, interleaved A/B, T3/T4) before any build.
- **E2 (reported):** V13-V15 against V0, V4 and V7, as the share of the gap between V0 and max(V4, V7) that closes.
- **E3 (reported):** each kernel's registers and blocks per SM, plus its `LDG` and `SHFL` counts from `cuobjdump`.
- **Prediction:** if redundant load issue is the limiter, V13 lands within 10% of max(V4, V7) on both MLP shapes.
  If V13 is no faster than V8, issue cost is not the limiter either. The remaining suspects are then the `dp4a`/hash
  dependency chain and occupancy, the next scouting target.

### W12e: built and queued (recorded 2026-10-31 08:36 EDT, nothing timed)

- The extension built on the desktop CPU in a capped unit (`logs/build-w12e.log`). The first attempt failed because
  shell quoting dropped the include path; `tools/build_w12e.sh` fixes it.
- Step `x-w12-e1` is queued after `x-census-raf` (queue rev 34, `queue/steps-rev34.txt`).

**E3, from the machine code** (`cuobjdump`, whole kernels, so the counts include the splat copy and the tail):

| V | instructions | LDG | SHFL | IDP | registers |
|---|---|---|---|---|---|
| 0 | 544 | 43 | 8 | 48 | 63 |
| 8 | 528 | 43 | 8 | 48 | 61 |
| 13 | 520 | 34 | 20 | 48 | 63 |
| 14 | 536 | 34 | 20 | 48 | 64 |
| 15 | 544 | 35 | 20 | 48 | 63 |

- All five fit the same occupancy (64 registers or fewer).
- V13 trades 9 loads for 12 shuffles in the unrolled loop. Whether shuffles are cheaper to issue than redundant L1
  hits is what the timing decides.

## R1: results (recorded 2026-10-31 09:38 EDT; steps `x-rafatxf1` 08:18-09:23 and `x-census-raf` 09:23-09:27 EDT; `ab/rafatxf1.md`, rows `ab/rafatxf1.jsonl`, `ab/run-x-rafatxf1.log`, `census/raf-d0/`, `census/raf-d28/`)

**Deviation: fdq4 never ran.**
- The branch compiles its fdq4 cubins with nvcc at first use. nvcc picked up the system's default gcc, which is newer
  than 13, and CUDA 12.5 refuses it ("unsupported GNU version!").
- Every rafatxf arm printed "fdq4: graph-path kernels unavailable, falling back to Triton decode attention".
- So every rafatxf arm, Q4 and Q8 alike, used Triton decode attention. R1b's fdq4 test did not happen. The
  comparison is ours against rafatxf's other sm_75 paths (GDN kernels, fp16 split-K GEMV and GEMM) with Triton
  attention.
- A rerun with nvcc pointed at gcc-13 is the follow-up (R1-redo, below).

**Gate passes.** All four smoke replies are readable: code, math, prose, and a correct summary of the 28K prompt.

**Speed** (round means, tok/s; ms/token in brackets):

| | fill 0, undrafted | 28K, undrafted | fill 0, MTP 6 | 28K, MTP 6 |
|---|---|---|---|---|
| ours adopted, Q8 | 45.00 (22.2) | 41.74 (24.0) | 57.60 (17.4) | 53.07 (18.8) |
| ours adopted, Q4 | 43.98 | 30.93 | 55.65 | 41.20 |
| rafatxf, Q4 (Triton attention) | 31.85 (31.4) | 24.07 (41.6) | 67.29 (14.9) | 41.95 (23.8) |
| rafatxf, Q8 | 31.90 | 23.31 | 68.05 | 39.88 |

**Predictions:**
- **R1a holds.** Undrafted at fill 0, rafatxf is 29.2% slower than ours in every round (-29.2, -29.6, -28.9%). Ours is
  1.41x faster.
- **R1b is not tested** (fdq4 did not run). rafatxf Q4 against Q8 at 28K is +3.3% in every round, both on Triton.
- **R1c, reported:**
  - **Ours against rafatxf:** undrafted at 28K, rafatxf is -42.3%. With MTP at 28K it is -20.9%. **With MTP at fill
    0, rafatxf is +16.9% faster** (+19.0, +18.4, +13.2%).
  - **Ours at Q4 against ours at Q8:** -2.3% at fill 0 and -25.9% at 28K, because fd75d does not cover Q4 and Triton
    takes over. With MTP it is -3.4% and -22.4%.
- **Draft acceptance is the same everywhere:** median 0.62-0.66 in all eight MTP arms. So rafatxf's MTP advantage at
  fill 0 comes from its kernels, not its drafts.
  - rafatxf's MTP is 2.11x its undrafted speed; ours is 1.28x.
  - Its multi-row verify step is much cheaper relative to its single-row step than ours.
  - This is the borrowable finding of R1.
- **R1d, census, undrafted (ms per token):**

| family | ours (`d0-f1`) | rafatxf Q4 fill 0 (`raf-d0`) |
|---|---|---|
| weight GEMV/GEMM | 18.49 (int8 sq) | 26.43 (fp16 split-K 13.66 + gemv 10.80 + gemm 1.97) |
| decode attention (split + combine) | 1.21 | 1.30 |
| GDN recurrent step | 0.62 | 0.57 (its `_128r` kernel, 32x4 blocks) |
| RMS norms | 0.56 | 0.49 |
| total span | 23.0 | 31.5 |

  At 28K, rafatxf's Triton attention costs 12.3 ms per token (31% of the span), against fd75d in ours.

**What to borrow, and what is next:**
- **The MTP verify path is the prize.** At fill 0 our MTP gains only 1.28x over undrafted, against rafatxf's 2.11x.
  - The suspect is ours sending verify rows (M = 2-7) through the fp16 path. Our int8 path covers only M <= 2, and
    W12c's IMMA attempt lost at high M.
  - rafatxf's verify GEMV and GEMM handle those rows with fp16 accumulation.
  - Next step: **R2**, MTP censuses of both at fill 0, to see which kernel families make rafatxf's verify cheap. It
    gets its own pre-registration.
- **R1-redo:** rafatxf with nvcc pointed at gcc-13 (`NVCC_PREPEND_FLAGS=-ccbin gcc-13`), so fdq4 really runs. That
  retests R1b and compares fdq4 against our fd75d at 28K.
- The GDN `_128r` kernel is 7% faster than ours, but on 0.6 ms per token. It is not worth a borrow on its own.

## W12e: results (recorded 2026-10-31 09:38 EDT; step `x-w12-e1`, 2026-10-31 09:27-09:28 EDT; `w12/sqmb_e1.jsonl`, `w12/run-x-w12-e1.log`)

Seven rounds, medians in microseconds per launch. This run's spread is far tighter than sqmb2's: min/max mostly
within 2%.

| V | MLP gate | MLP down | GDN qkv | attn q |
|---|---|---|---|---|
| 0 upstream | 52.71 | 59.50 | 37.66 | 40.62 |
| 4 no loads | 38.05 | 43.41 | 27.37 | 29.36 |
| 7 loads only | 40.75 | 43.01 | 26.51 | 29.71 |
| 8 running pointers | 46.08 (0.874x) | 55.17 (0.927x) | 35.56 (0.944x) | 35.36 (0.870x) |
| 13 word/lane + shfl | 47.39 (0.899x) | 55.34 (0.930x) | 35.18 (0.934x) | 35.83 (0.882x) |
| 14 + prefetch 1 | 48.74 | 60.10 | 38.60 | 37.94 |
| 15 + prefetch 2 | 48.76 | 58.81 | 37.57 | 37.87 |

- **C0 holds:** V13-V15 are bit-identical to V0 on every shape.
- **E1 fails.** V13 is within -1.1% to +2.8% of V8, nowhere near 8% below it. Prefetch makes it slower.
- **The prediction fails.** V13 sits 16% (MLP gate) and 27% (MLP down) above max(V4, V7). Redundant load issue is
  **not** the limiter. As pre-registered, the remaining suspects are the dp4a/hash dependency chain within a row,
  and occupancy.
- **E2:** V13 closes 46% of the gap between V0 and max(V4, V7) on MLP gate, 26% on MLP down, 22% on GDN qkv and 44%
  on attn q. That is the same as V8 (52%, 26%, 19%, 48%), so the gain is address arithmetic, not load count.
- **A by-product: running pointers now clear the bar.** With this run's tight timings, V8 beats V0 by 12.6%, 7.3%,
  5.6% and 13.0% on the four shapes. That is at least 5% on every shape, the bar sqmb1's L5 set. sqmb2's medians
  had GDN qkv level, with much wider spread.
  - L5 belonged to sqmb1, so this is evidence, not a decision.
  - Running pointers in the engine's narrow unit (**W12b-ptr**) now gets its own pre-registration: exactness,
    census, interleaved A/B, T3/T4.
  - Expected size: the int8 GEMV is about 85% of undrafted decode. If the microbenchmark's 6-13% carries over,
    decode gains about 5-10%.

## R2 and R1-redo: where rafatxf's MTP gain comes from, and fdq4 for real (pre-registered 2026-10-31 09:39 EDT, before either step runs)

**R2, step `x-census-mtp2`:** nsys censuses with MTP 6 dynamic at fill 0 and cache 32768, for both builds:
- ours adopted (w11b, `EXL3_MGEMM_N_THRESHOLD=0 EXL3_FD75D_LIVE=0 EXL3_INT8_SQ_MAX_M=2`, Q8): tag `d0-w11b-mtp`;
- rafatxf at Q4 under R1's own condition (Triton attention), so it matches R1's +16.9%: tag `raf-d0-mtp`.

Both report kernel families per decoded token, and launch shapes split by row count where the census can tell.

- **R2a:** most of the gap (rafatxf 14.9 ms against ours 17.4 ms per token in R1) is in the weight kernels that run
  the verify rows. Ours sends M = 3-7 to the fp16 path; rafatxf uses its fp16-accumulate GEMV and GEMM.
- **R2b:** the GDN multi-token path differs by less than 1 ms per token.
- **Decision:** none. If R2a holds, the borrow candidate is rafatxf's verify-row weight kernel. It gets its own
  pre-registration with exactness and quality gates, and the source credited (MIT).

**R1-redo, step `x-rafatxf2`:** rafatxf with `NVCC_PREPEND_FLAGS="-ccbin gcc-13"`, so its runtime nvcc uses a
supported host compiler.
- **Gate:** the smoke run's log must not contain "kernels unavailable", and the replies must be readable. The step
  checks the log itself and skips the A/B if fdq4 still falls back.
- **Arms:** 7, in 3 rotated rounds:
  - ours adopted at 28K, Q8;
  - rafatxf Q4 (fdq4) and rafatxf Q8 (Triton), both at 28K;
  - ours MTP and rafatxf MTP (fdq4), at 28K;
  - rafatxf undrafted and MTP at fill 0, both with fdq4.
- **R1b' (R1b retested):** at 28K undrafted, rafatxf Q4 with fdq4 beats rafatxf Q8 (Triton) by at least 20% in
  every round.
- **R1e (reported):** ours with fd75d at Q8 against rafatxf with fdq4 at Q4, undrafted and with MTP at 28K. fdq4 reads
  half the cache bytes.
- **R1f (reported):** rafatxf at fill 0 with and without fdq4 (against R1's rows). It should not matter at fill 0.

Both steps go in right after the running quality step `x-q-exl3-nodraft-w11-mp05` (queue rev 35).

## W12b-ptr scouting: does a one-line stepped pointer get V8's gain? (pre-registered 2026-10-31 09:41 EDT, before any build or timing)

V8 makes two changes:
1. the row pointer steps instead of being recomputed from `kb`;
2. the lane's two word offsets are folded into two running pointers.

The engine's `ext8w` computes word indices from `t0` separately for each of the eight bit widths, so change 2 in the
engine means refactoring every K path. Change 1 alone is one line in `gemv_int8_unit_narrow` and covers every K.

- **V16** (`mb_unit_step`): upstream's own `gemv_int8_pair_row<2, 1>`, with only `blockA += row_stride` per row.
  Its partials must equal V0's.
- **Step `x-w12-ptr1`:** `w12_sqmb.py --part repeat --rounds 7 --variants 0,8,16`, queued after `x-rafatxf2`.

**Rules:**
- **C0:** V16 is bit-identical to V0 on every shape.
- **P5 (decision):** which engine patch W12b-ptr becomes.
  - If V16's median is within 3% of V8's on all four shapes, W12b-ptr is the one-line form.
  - If V16 recovers less than half of V8's gain over V0 on any shape, W12b-ptr is the full form: every K path's
    offsets hoisted, with V8 as the K = 2 reference.
  - In between, the one-line form goes first and the full form is scouted after.
- Either way, the engine patch gets its own pre-registration: exactness against w11b, census, interleaved A/B, T3/T4.

## Corrections found while drafting Volume 0 (recorded 2026-10-31 10:55 EDT; the text above is kept as written)

The Volume 0 drafters re-derived several numbers in this note from raw rows. These corrections are re-checked by
script against the files named. Decisions do not change.

- **L2: heat was a limit at 320 W, not only memory.** In the decode windows of `ab/power1_telemetry.csv`, the
  `0x20` throttle flag (software thermal slowdown) appears in:

  | limit | fill 0 | 28K |
  |---|---|---|
  | 260 W | 0 of 256 samples | 0 of 284 |
  | 290 W | 0 of 255 | 25 of 287 (9%) |
  | 320 W | 42 of 252 (17%) | 184 of 274 (67%) |

  - The 28K arms, where the clock rose least, are mostly heat-limited.
  - "Once power is lifted, the remaining limit is the fixed memory clock" was incomplete. Heat was the binding limit
    in most 320 W samples at 28K.
  - "Stay at 260 W" stands, and more firmly.
- **L2: the speed range.**
  - "320 W buys 0-6% speed in round means" was too narrow. Across the eight configurations the round means run from
    -2.4% to +7.5%. Adopted: +6.2, +5.5, +6.1, +2.2. Stock: -0.3, -2.4, +7.5, +1.9.
  - "Speed rises less than the clock" holds for stock undrafted. Adopted gained about as much as its clock: +6.2%
    for +7.1% at fill 0, and +5.5% for +4.5% at 28K.
- **L2: wording.** "2-19% fewer tokens per joule" is right. Some summaries said "energy per token 2-19% worse",
  which is wrong: energy per token rose 2.4-24%.
- **The int8 share of the token.** The census's 84.8% is a share of summed kernel time. As a share of the
  22.9 ms span, `exl3_gemv_int8_sq_kernel`'s 18.49 ms is 80.4%. Statements above that it is "about 85% of
  undrafted decode" mean the share of kernel time.
- **W12e: share of the gap closed.**
  - The values given were miscalculated. From `w12/sqmb_e1.jsonl` medians:
    - V13 closes 44.5% (MLP gate), 25.9% (MLP down), 24.1% (GDN qkv) and 43.9% (attn q);
    - V8 closes 55.4%, 26.9%, 20.4% and 48.2%.
  - The conclusion (V13 and V8 alike; address arithmetic, not load count) stands.
- **sqmb1: loads alone on MLP gate.** "41-48 µs" understates the spread. V7's single timings across the slice sweep
  run from 40.92 to 54.42 µs.

### Correction: drafted speeds were overstated by the mean of per-prompt rates (recorded 2026-10-31 10:56 EDT)

- **The bug.** `ab_report.py` averaged each prompt's tok/s. Under MTP, prose runs much slower than code and math, so
  the mean of rates exceeds pooled speed (total new tokens over total decode wall time).
- **The fix.** The tool now prints both, plus pooled changes for each pair; this was found by a Volume 0 drafter.
- **Recomputed from the same rows:**

  | arm | mean of rates | pooled |
  |---|---|---|
  | rafatxf1 adopt-mtp-0 | 57.60 | **54.02** |
  | rafatxf1 raf-mtp-0 | 67.29 | **63.02** |
  | rafatxf1 adopt-mtp-28k | 53.07 | **49.68** |
  | rafatxf1 raf-mtp-28k | 41.95 | **39.40** |
  | final1 stock-mtp-0 | 46.84 | **43.42** |
  | final1 adopt-mtp-0 | 53.24 | **49.32** |
  | final1 stock-mtp-28k | 36.78 | **33.90** |
  | final1 adopt-mtp-28k | 50.77 | **47.06** |

- **Undrafted arms agree within 0.3%** (for example, adopt-0 45.00 against 44.99).
- **Comparisons between builds barely move:**
  - rafatxf's MTP lead at fill 0 is +16.7% pooled, against +16.9%;
  - at 28K, -20.7% against -20.9%;
  - adopted over stock with MTP at fill 0 is 1.136x pooled, against 1.137x.
- **No decision changes.** Absolute MTP speeds quoted above and in chat are 6-8% high. The pooled values are the ones
  to quote.
- **Between sessions:** the same undrafted adopted arm read 42.57 tok/s pooled in `final1` (01:38-02:10) and 44.99 in
  `rafatxf1` (08:18-09:23). Absolute speeds drift between sessions by about this much. Only interleaved comparisons
  within one session are claims.

### More corrections from the Volume 0 drafters (recorded 2026-10-31 11:04 EDT; the text above is kept as written)

- **Stamps ran ahead on more headings than recorded.** Each heading's stamp was compared with the commit that first
  contains it.
  - Twelve headings from 30 October carry stamps 2.5-24 minutes later than their commits: W7, W6, W8, W7+G1, the
    interleaved confirmation, W9, W10a, D, W10b, W11 scouting, W11 and W11b. The 00:15 correction covered only the
    sections around midnight.
  - In eleven of these, the commit still precedes the first data row, W6 by only 16 seconds.
  - **W11's order cannot be shown.** Its first kernel-test row is stamped 20:00:20 by the desktop clock, and the
    commit adding its pre-registration (`bab8a4c`) is at 20:00:29 by the laptop clock. "Before any timing of it"
    has no record behind it.
  - The L1 pre-registration and result share one commit (`e7c2dc6`), so git cannot order them.
  - **Since the stamps have been generated by the writing command,** all 14 headings added have stamps at or before
    their commits.
- **W12c's scaling ratio.** "IMMA at M = 8 costs 1.68-2.13x its M = 1 time" is misstated. The rows give 1.68-1.98x;
  2.13 is another ratio from the same table. The decision stands.
- **R1's fallback line.** "Every rafatxf arm printed the fdq4 fallback" is too broad. The smoke run and the four
  Q4-cache arms printed it, 13 lines in all. The Q8 arms use Triton by design.
- **L3's laptop reference.** "About -5% tokens, about -10% wall at -0.5" comes from E45's math25 cells only
  (-7.5% tokens, -11% wall). Its GSM8K cells ran at -1.0 and -2.0. So the 2080 Ti's GSM8K run at -0.5 is a cell the
  laptop never measured.
- **L3's E49 caveat** (laptop 1.4.7 non-determinism) does not carry over to 1.6.5. The fix shipped in 1.5.0 (E52).
  The run-to-run variation recorded on 1.6.5 comes from MTP's changing verify shapes (W8).

## W12b-ptr scouting, R2 and R1-redo: results (recorded 2026-10-31 11:11 EDT; steps `x-census-mtp2` 10:33-10:36, `x-rafatxf2` 10:36-11:07, `x-w12-ptr1` 11:07 EDT)

### W12b-ptr scouting (`w12/sqmb_ptr1.jsonl`, `w12/run-x-w12-ptr1.log`)

Seven rotated rounds; medians in microseconds per launch:

| shape | V0 upstream | V8 running pointers | V16 stepped row pointer (one line) |
|---|---|---|---|
| MLP gate | 53.14 | 49.12 (0.924x) | 49.08 (0.924x) |
| MLP down | 60.82 | 57.47 (0.945x) | 56.36 (0.927x) |
| GDN qkv | 38.25 | 36.60 (0.957x) | 36.21 (0.947x) |
| attn q | 40.53 | 37.48 (0.925x) | 37.36 (0.922x) |

- **C0 holds:** V16 is bit-identical to V0 on every shape.
- **P5 decision: the one-line form.** V16 is within 3% of V8 on all four shapes; it is equal or faster on every one.
  W12b-ptr is the stepped row pointer in `gemv_int8_unit_narrow`, covering every K. It gets its engine
  pre-registration next.
- **Between sessions:** V8's gain over V0 is 4.3-7.6% here against 5.6-13.0% in W12e.

### R2: where rafatxf's MTP advantage comes from (`census/d0-w11b-mtp`, `census/raf-d0-mtp`)

MTP 6 dynamic, fill 0. Kernel families in ms per decoded token, summed over the three tasks (the census report's
"us/token" column, divided by 1,000):

| family | ours adopted (Q8) | rafatxf (Q4, Triton attention) |
|---|---|---|
| multi-row weight GEMV (`exl3_gemv_kernel`) | 10.50 (68 x 512 threads, 5 shapes) | 6.32 (136 x 256 threads, 7 shapes) |
| split-K GEMV (`exl3_gemv_sk_kernel`) | - | 4.86 |
| int8 sq GEMV | 2.29 | - |
| GEMM (`exl3_gemm_kernel`) | 2.27 | 0.81 |
| **weight kernels in total** | **15.06** | **11.99** |
| GDN recurrent step + state replay | 0.45 + 0.17 | 0.40 + 0.18 |
| decode attention (split + combine) | 0.27 + 0.12 (fd75d) | 0.71 + 0.06 |
| span per token: code / math / prose | 16.03 / 15.27 / 26.42 | 13.34 / 12.91 / 22.95 |

- **R2a holds.** The weight kernels differ by 3.07 ms per decoded token, which is essentially the whole gap. Our
  verify rows go to the fp16 multi-row GEMV at 68 blocks of 512 threads, and to GEMM more often. rafatxf splits the
  same work across a 136 x 256 GEMV and a split-K GEMV.
- **R2b holds.** The GDN multi-token path differs by 0.04 ms.
- **The borrow candidate is rafatxf's verify-row weight path:** its launch configuration, split-K and GEMM routing for
  M = 2-7 rows. It gets its own pre-registration (W13) after reading its code, with exactness and quality gates and
  credit (MIT).

### R1-redo: fdq4 for real (`ab/rafatxf2.md`, rows `ab/rafatxf2.jsonl`)

- **The gate passes.** No fallback line appeared; the smoke replies are readable; rafatxf's 28K decode in the smoke
  run rose from 24.22 tok/s (R1, Triton) to 31.65.
- **Draft acceptance is equal:** 0.64-0.66.
- **Pooled speeds, tok/s:**

  | arm | tok/s |
  |---|---|
  | ours 28K undrafted (Q8, fd75d) | 41.45 |
  | rafatxf 28K, Q4 with fdq4 | 31.18 |
  | rafatxf 28K, Q8 (Triton) | 23.24 |
  | ours MTP 28K | 50.68 |
  | rafatxf MTP 28K, fdq4 | 60.49 |
  | rafatxf fill 0, fdq4 | 32.53 |
  | rafatxf MTP fill 0, fdq4 | 63.56 |

- **R1b' holds.** fdq4 (Q4) beats rafatxf's Q8 Triton path by +34.0 to +34.5% in every round.
- **R1e: undrafted at 28K, ours leads.** rafatxf is -24.8% in every round (ours with fd75d at Q8 is 1.33x).
- **R1e: with MTP at 28K, rafatxf leads by +19.4% pooled** (+17.9 to +21.2% per round, faster in all three). In R1,
  with Triton attention, it was -20.7%. With fdq4 working, rafatxf wins MTP at both depths.
- **R1f holds:** at fill 0, fdq4 changes little (32.53 against 31.85 undrafted; 63.56 against 63.02 with MTP).

### What this means for P8

- **Our int8 path and fd75d are the best undrafted Turing configuration measured:** 1.41x rafatxf at fill 0 and
  1.33x at 28K.
- **rafatxf's Turing branch with MTP is the fastest configuration measured on this card overall:** +16.7% at fill 0
  and +19.4% at 28K over ours. That is the mode that matters for daily use, because MTP is accuracy-lossless here.
- **The related-work note's "fastest original-weights decode found on Turing at 12 GB or less" (C5a) does not hold
  for MTP decoding.** Corrected wording: the fastest undrafted. rafatxf's MIT branch is faster drafted, and it is
  cited as such.
- **The path forward is to borrow, with credit:**
  - W12b-ptr (ours, one line, every K) for the single-row int8 path;
  - W13: rafatxf's verify-row weight path for MTP;
  - possibly fdq4-style Q4 attention for MTP at depth.

  The target is a configuration that leads both modes.

## W12b-ptr in the engine (pre-registered 2026-10-31 11:12 EDT, before the build or any run)

- **Patch:** `tools/turing/patches/exllamav3-1.6.5-w12p-int8-stepped-pointer.patch` (sha256 6a489de6…), on top of
  the w11b chain. In `gemv_int8_unit_narrow` the row pointer steps by `row_stride` instead of being recomputed from
  `kb`. The words, the arithmetic and their order are unchanged. It covers every K and every row count, so it reaches
  both the single-row decode path and the M <= 2 verify rows.
- **Build:** overlay `exl3-w12p` (patch chain w7, w9, w10, w10b, w11, w11b, w12p) in a capped CPU unit.

**Steps** (queued after the running quality step, waiting for the overlay):
1. `x-w12p-check`: `w12c_check.py dump` with w11b and w12p, both at `EXL3_INT8_SQ_MAX_M=4`, and a bit-identity
   comparison.
2. `x-census-w12p`: an undrafted census at fill 0.
3. `x-w12p-ab`: an interleaved A/B with 3 rounds; adopted (w11b) against adopted with W12b-ptr (w12p), each
   undrafted and MTP, fill 0 and 28K. That is 8 arms.

**Rules:**
- **X (exactness):** every case of the dump (5 shapes x K 1-4 x M 1-7) is bit-identical between w11b and w12p.
  If any case differs, the patch is a bug and nothing else counts.
- **C (census):** `exl3_gemv_int8_sq_kernel` time per token falls at least 4% against `d0-f1-mgemmn0`. That census
  was a different session, so this is reported, with the A/B as the decision.
- **A1 (decision):** undrafted at fill 0, w12p is at least 3% faster than w11b in pooled tok/s in every round.
- **A2 (reported):** the same at 28K and under MTP. The expected gain is smaller under MTP, where the int8 path
  carries less of the time (R2: 2.29 of about 18 ms per token).
- **Quality:** if X holds, outputs are bit-identical to the adopted build, so the T3/T4 quality results carry over
  unchanged. No separate quality run is needed.
- **Adoption:** X and A1 hold.
- **Noise watch:** a headless Chrome swarm from another project (65 processes at nice 15) is loading the desktop CPU
  (load average 40-57 at 11:12). The A/B is interleaved, so contention lands on both arms. The load average is
  recorded with the result.

## W13: rafatxf's split-K GEMV for the verify rows (pre-registered 2026-10-31 11:14 EDT, before any port, build or run)

**What R2 points at.** Under MTP, our verify rows (M = 2-7) spend 10.50 ms per decoded token in upstream's
small-m `exl3_gemv_kernel` and 2.27 ms in GEMM. rafatxf's branch spends 6.32 + 4.86 + 0.81 ms on the same work.

**The borrowed component.** rafatxf/exllamav3, branch `turing` at 15654d3 (MIT, Copyright Turboderp):
- `exllamav3_ext/quant/exl3_gemv_sk_kernel.cuh` (471 lines), its instances (`comp_units/exl3_gemv_sk_inst.cu`),
  the fp32 workspace in `DevCtx`, and the sm_75 block of `exl3_gemv_try_launch`.

How it works, from its own header:
- The (column group x k-slice) work is cut into equal contiguous ranges, one per block and then one per warp, so
  every SM gets equal work at any shape. Upstream's one-block-per-column-group grid leaves a partial last wave at
  n = 5120, 6144 and 10240, and idle SMs at n = 1024.
- Tensor-core m16n8k8 pairs with fp16 accumulation, folded to fp32 once per prefetch-ring cycle.
- Deterministic sums: block partials are summed in block order after a grid sync.
- It covers M up to 16.
- rafatxf's measurement (4 bpw model, same card): weight time 30.4 to 26.6 ms at m = 1, and 32.1 to 30.1 ms at
  m = 8. Ours is a 2.0 bpw model.

**The port (W13 patch, on top of w11b + W12b-ptr):**
- Add the kernel, the instances, the workspace and the dispatch block to our 1.6.5 tree.
- Route on sm_75 only, behind `EXL3_GEMV_SK` (0 = off; default on in the patched build, so one variable gives the
  A/B):
  - M = 1 stays on our int8 sq path, which is faster single-row (R1: ours 1.41x rafatxf undrafted);
  - M = 2 stays on int8 (J2, adopted);
  - **M = 3-16 go to split-K.**
- Credit in the patch header and in any paper text.
- Nothing is pushed upstream before P8 is published.

**Gates, in order:**
1. **Build and import** (`with_engine.sh`).
2. **X (numerics):** split-K against the current fp16 path at M = 3-8, five shapes, K = 1-4. Finite everywhere;
   median relative difference at most 0.5%, and maximum at most 2%, per (shape, M). It is not bit-identical, since
   accumulation differs.
3. **Census:** MTP at fill 0. The weight kernels' total per decoded token falls at least 2.0 ms against
   `d0-w11b-mtp` (15.06 ms).
4. **A (decision):** an interleaved A/B with 3 rounds, w12p against w12p + W13, MTP at fill 0 and 28K. Pooled tok/s
   at least 8% faster in every round at fill 0. 28K is reported.
5. **T4 (quality):** GSM8K-500 with MTP on the W13 build against the adopted MTP rows. Accuracy change within the
   pre-registered -1.0 point bound, with the discordant pairs reported.
6. **Adoption:** all of 1-5 hold. If X fails, the port is wrong, and nothing after it counts.

**Prediction:** MTP pooled speed rises about 15% (most of R2's 3.07 ms gap), with undrafted unchanged (M = 1 is not
rerouted).

### W13: amendment, before any build or run (recorded 2026-10-31 11:27 EDT)

**The patch.** `tools/turing/patches/exllamav3-1.6.5-w13-gemv-sk.patch` (sha256 9397b6ec…), written by a porting
agent and reviewed here.
- **Contents:** rafatxf's kernel unchanged below a credit header; the M >= 2 instances only (68 of 102); a separate
  `exl3_gemv_sk_try_launch` called first in `exl3_gemv_try_launch`, so a declined call takes exactly its old path;
  a 10 MiB workspace allocated with the lock buffer, only on compute capability 7.5 and never during graph capture.
- **The switch, `EXL3_GEMV_SK`:**
  - 0 means off, identical to the tree without the patch;
  - unset or 1 means on with rafatxf's shape limits: K > 4, or M >= 9, or 2048 < n <= 12288;
  - **2** (added for testing) drops those limits.
- **Coverage under mode 1 at M = 3-8:**
  - **routed:** GDN qkv, GDN z, GDN out, attn o, attn q, MLP down;
  - **not routed:** MLP gate/up (n = 17408), attn k/v (n = 1024), K = 1 layers (no instance), lm_head.

**Amended gates.** The decision stays on mode 1 as pre-registered; mode 2 is added as a reported arm.
- **X (numerics).** `w12c_check.py`: dumps with `EXL3_INT8_SQ_MAX_M=2` and `EXL3_GEMV_SK` = 0, 1 and 2, then
  comparisons 0-1 and 0-2. The measure is `max_rel_diff` = max|Δ| / max|reference| per case (shape, K, M).
  - **The new path, M = 3-7:** every case finite, at most 2% each, and a median over those cases of at most 0.5%.
  - **Positive control, M = 1-2:** bit-identical between 0 and 1. Those rows never reach split-K, so any difference
    there is a bug.
- **Census.** MTP at fill 0, modes 1 and 2. Weight kernels down at least 2.0 ms per decoded token against
  `d0-w11b-mtp` (mode 1 decides).
- **A.** Six arms, 3 rounds, MTP at fill 0 and 28K: w12p, w13 mode 1, w13 mode 2. Mode 1 at least 8% faster than
  w12p in pooled tok/s in every round at fill 0. Mode 2 is reported.
- **T4.** GSM8K-500 with MTP on w13 mode 1, against the adopted MTP rows `TURING-exl3-mtp-w11j2`. The change is
  within -1.0 point, with discordant pairs reported.
- **Mode 2 follow-up:** if mode 2 beats mode 1 by at least 3% in every round, it needs its own X and T4 before it
  could be adopted.

**Build.** Overlay `exl3-w13`: the w12p chain plus W13. It starts in a capped unit after the w12p build reports
"overlay ready"; builds run one at a time.

## L3: results (recorded 2026-10-31 12:04 EDT; steps `x-q-exl3-nodraft-w11-mp05` 09:28-10:33 and `x-q-exl3-mtp-w11j2-mp05` 11:11-11:56 EDT; `quality/compare_l3_mp05.md`, rows `quality/TURING-exl3-*-mp05/`)

GSM8K-500, paired item by item against the unpenalized runs of 2026-10-30 (same build and switches):

| mode | completion tokens | median per-item difference | accuracy | discordant (lost / gained) | exact McNemar | wall | capped |
|---|---|---|---|---|---|---|---|
| undrafted | 164,236 to 160,588 (-2.22%) | 0 | 96.8 to 96.8 | 2 / 2 | p = 1.000 | -4.6% | 0 to 0 |
| MTP | 161,886 to 160,809 (-0.67%) | 0 | 97.6 to 97.0 (-0.6 pt) | 3 / 0 | p = 0.250 | -3.8% | 0 to 0 |

- **M1 fails in both modes.** Required: at least 3% fewer tokens and a negative median per-item difference. GSM8K on
  this model barely overthinks: no item hit the 4,096-token cap in any of the four runs.
- **M2 holds in both modes.** The undrafted change is 0.0 points; the MTP change is -0.6 points, within the -1.0
  bound and not significant. All three MTP discordants went to the unpenalized side, which is worth watching if the
  penalty is ever retested.
- **M3 (reported).** Wall time -3.8% to -4.6%, against runs from a different day, so it is not a claim. Token
  savings of 0.7-2.2% cannot explain it alone.
- **Decision: not adopted on the 2080 Ti.** The laptop's -5% tokens at -0.5 came from its math25 set (see the
  correction above). On GSM8K the effect is too small to clear the pre-registered bar.

## W12b-ptr in the engine: results (recorded 2026-10-31 12:30 EDT; steps `x-w12p-check`, `x-census-w12p`, `x-w12p-ab`, 11:38-12:24 EDT)

- **X fails as written.** 134 of 140 cases are bit-identical; 6 differ by 0.65-0.82% (`max_rel_diff`). All six are
  M = 3: gdn_z K2, gdn_qkv K1 and K2, mlp_gate K1 and K2, mlp_down K2.
  - Every M = 1, M = 2 and M = 4 case is bit-identical.
  - **Same-config control:** two dumps of the unchanged w11b build, 02:10 and 11:56 EDT, are bit-identical in all
    140 cases. The differences come from the w12p build, not from run-to-run noise.
  - The register counts of the K = 2 sq instances (M = 2, 3, 4) are equal in both builds (64-69), so the W12c cause
    (an occupancy-driven slice choice) is not established either.
  - **Next:** `x-w12p-fitlog` (queue rev 39) logs both builds' slice choices.
- **C holds (reported).** `exl3_gemv_int8_sq_kernel`: 17.41 ms per token (`census/d0-w12p`), against 18.49 in
  `d0-f1-mgemmn0`, -5.9%. That census was a different build and session. The w12p census shows 10 distinct launch
  shapes against 11.
- **A1, diagnostic only** (pooled, `ab/w12p1.md`):

  | mode | w11b | w12p | change per round |
  |---|---|---|---|
  | undrafted, fill 0 | 45.55 | 47.58 | +4.6 / +4.2 / +4.5% |
  | undrafted, 28K | 41.95 | 43.94 | +4.5 / +4.7 / +5.1% |
  | MTP, fill 0 | 55.20 | 54.23 | +0.2 / -3.0 / -2.5% |
  | MTP, 28K | 51.40 | 51.42 | +2.5 / -1.7 / -0.3% |

  - The undrafted gain would clear A1's 3% in every round. Under MTP there is no gain: the int8 path carries less
    there.
  - Load average 14.5 at the start and 5.1 at the end.
- **Decision: not adopted on this run, as pre-registered**, because X failed.
- **Note on production:** the adopted configuration routes only M = 1 (undrafted) and M = 2 (MTP verify) to int8
  (`EXL3_INT8_SQ_MAX_M` 1 and 2). Every case at those row counts is bit-identical. Any second attempt (W12b-ptr r2)
  gets its own pre-registration once the M = 3 cause is known: X over the M the configuration uses, plus a fresh
  interleaved A/B. This run's data does not decide it.

## W13: gates X and census (recorded 2026-10-31 12:30 EDT; steps `x-w13-check` and `x-census-w13`, 12:24-12:27 EDT; A/B and T4 still running)

**X holds** (`w12/w13_check_sk1.jsonl`, `w12/w13_check_sk2.jsonl`).
- **Positive control:** M = 1-2 bit-identical, 40 of 40, in both modes.
- **The new path, M = 3-7:** all finite; maximum `max_rel_diff` 6.39e-4 (limit 2e-2); median over the cases
  0 (mode 1) and 4.99e-4 (mode 2) (limit 5e-3).
- **Mode 1 changes exactly the routed shapes:** gdn_z, gdn_qkv and mlp_down at K2-K4 (45 of 100 cases). Mode 2 also
  changes mlp_gate, 75 of 100 cases.

**Census holds** (`census/d0-w13-mtp-sk1`, `census/d0-w13-mtp-sk2`). MTP at fill 0, ms per decoded token:

| build | weight kernels in total | small-m GEMV | split-K | int8 sq | GEMM | span code / math / prose |
|---|---|---|---|---|---|---|
| w11b (`d0-w11b-mtp`) | 15.06 | 10.50 | - | 2.29 | 2.27 | 16.03 / 15.27 / 26.42 |
| w13 mode 1 | **12.30** | 4.05 | 4.12 | 2.37 | 1.76 | 12.28 / 11.89 / 21.36 |
| w13 mode 2 | **10.70** | - | 7.10 | 2.88 | 0.72 | 11.25 / 10.97 / 18.36 |
| rafatxf (`raf-d0-mtp`) | 11.99 | 6.32 | 4.86 | - | 0.81 | 13.34 / 12.91 / 22.95 |

- **The rule needed at least 2.0 ms:** mode 1 is -2.76 ms; mode 2 is -4.36 ms.
- In this single capture, mode 2's span is below rafatxf's on every task. That is our int8 single-row path plus their
  split-K for everything else. The interleaved A/B decides.

## W13: A, T4 and the decision; W12b-ptr's fit log (recorded 2026-10-31 13:41 EDT; steps `x-w13-ab` 12:27-12:45, `x-w12p-fitlog` 12:46 and `x-q-w13-mtp` 12:46-13:24 EDT; `ab/w13ab1.md`, rows `ab/w13ab1.jsonl`, `quality/compare_t4_w13.md`, rows `quality/TURING-exl3-mtp-w13/`, `w12/w12p_fit_{w11b,w12p}.txt`, `w12/regs_full_exl3-{w11b,w12p}.txt`)

**A holds.** Pooled tok/s with MTP, 3 interleaved rounds:

| arm | fill 0 | 28K |
|---|---|---|
| w12p | 54.59 | 51.79 |
| w13 mode 1 | 67.62 | 61.49 |
| w13 mode 2 | 72.19 | 66.13 |

- **Mode 1 against w12p, fill 0 (the decision):** +20.9 / +23.2 / +27.5% per round, +23.9% over all rounds. The rule
  needed at least 8% in every round.
- **Mode 1, 28K (reported):** +21.2 / +18.5 / +16.6%, +18.7% overall.
- **Mode 2 against w12p (reported):** +29.8 / +28.3 / +38.8% at fill 0 (+32.2%), and +28.3 / +25.7 / +29.0% at 28K
  (+27.7%).
- **Mode 2 against mode 1:** +7.4 / +4.2 / +8.9% at fill 0 (+6.8%), and +5.9 / +6.1 / +10.6% at 28K (+7.5%). That is
  at least 3% in every round, so the pre-registered **mode 2 follow-up is triggered**.
- **The prediction** was about +15%. Mode 1 gave +23.9%.
- **Load average** was 3.75 at the start and 17.45 at the end (the other project's Chrome swarm). The rounds are
  interleaved, so the load fell on every arm. Round 3 is the widest, but every round clears the bar.

**T4 holds.** GSM8K-500 with MTP, `TURING-exl3-mtp-w13` (mode 1) against the adopted `TURING-exl3-mtp-w11j2`:
- **Accuracy:** 97.2% [95.4, 98.3] against 97.6% [95.9, 98.6]. The change is -0.40 points, inside the -1.0 bound.
- **Discordant items:** 2 / 0 (`gsm8k_12` and `gsm8k_505` correct only on the reference), McNemar p = 0.50.
  - Against stock MTP it is the same: -0.40 points, 2 / 0.
  - As in the int8 T4, every disagreement goes to the reference side. Each step stays inside its bound, so this is a
    watch item for stacked changes, not a failure.
- **Agreement with the reference:** 88.2% byte-identical [85.1, 90.7], with the same final answer on 498 of 500.
- **Tokens and speed:** mean tokens 323.8 against 330.5; draft acceptance 0.622 on both.
  - Wall per item 5.44 to 4.47 s (1.215x); accuracy per minute 10.77 to 13.03.
  - Median server decode 63.2 to 79.0 tok/s.
- **GPU during the run:** 80 C at most, 248.9 W on average.

**Decision: W13 mode 1 is adopted.** Build, X, census, A and T4 all hold.
- **The MTP configuration of record is now:** overlay `exl3-w13`, `EXL3_GEMV_SK=1`, `EXL3_INT8_SQ_MAX_M=2`,
  `EXL3_MGEMM_N_THRESHOLD=0`, `EXL3_FD75D_LIVE=0`.
- **The undrafted configuration is unchanged** (w11b, `EXL3_INT8_SQ_MAX_M=1`). W13 does not touch M = 1. The
  W12b-ptr r2 decision below can move undrafted onto the same build.
- **The w13 build contains W12b-ptr.** The W13 pre-registration named that stack ("on top of w11b + W12b-ptr"), so
  adopting it brings the stepped pointer in under MTP. What covers that:
  - At M = 2, the only int8 row count this configuration uses under MTP, every X case was bit-identical with w11b
    (`x-w12p-check`).
  - The fit log below shows no occupancy change at M <= 2.
  - T4 ran on the whole build.
  - What it does not settle: A1's MTP arms showed no gain from the pointer (pooled -1.8%; +0.2 / -3.0 / -2.5% per
    round), so the build may carry a small MTP cost from it. `x-final2` measures the whole chain against w11b
    directly.
- **Credit:** the split-K kernel is rafatxf's (rafatxf/exllamav3, branch `turing`, 15654d3, MIT). It stays credited
  in the patch header and in any paper text. Nothing is pushed upstream before P8 is published.

**W12b-ptr's six M = 3 differences are explained.**
- **The fit log:** each build logs 80 shapes. 10 differ, all at M = 3 with K = 1 or 2, and in all 10 the grid goes
  from 204 to 272 blocks (3 to 4 per SM).
  - In 6 of the 10 the host's slice height changes too. Those are exactly the six cases that differed in X: gdn_z K2,
    gdn_qkv K1 and K2, mlp_gate K1 and K2, mlp_down K2.
  - The other 4 keep their slice height and are bit-identical.
- **The cause is registers.** From `cuobjdump --dump-resource-usage` of both extensions, parsed per function:
  - The non-residual, non-half K = 1 and K = 2 instances at M = 3 fall from 66 to 64 registers.
  - At 256 threads per block, 64 registers allow 4 blocks per SM (65536 / (64 x 256)). 66 rounds up to 72 and allows
    3.
  - The grid is sized from occupancy, so the slice height follows, and so do the per-slice activation scales and the
    rounding. This is the W12c lesson again: the patch changed no arithmetic, but it changed occupancy.
- **The other register changes cross no occupancy line in any instance this model uses:**
  - K = 1 and 2 at M = 1: 58 to 64 (4 blocks either way);
  - K = 1 at M = 4: 73 to 72; K = 2 at M = 4: 67 to 71 (3 blocks either way);
  - K = 6 at M = 1: 64 to 74 (4 to 3 blocks), but this model has no K = 6 layer;
  - the residual K = 2 instance at M = 2: 81 to 80 (2 to 3 blocks), but residual mode is not used.
  - At M <= 2, no non-residual instance for K = 1-4 changes occupancy, so no shape's decomposition changes there.
- **Correction:** the 12:30 entry said the register counts of the K = 2 sq instances (M = 2, 3, 4) were equal in both
  builds (64-69). That was wrong. K = 2 goes from 66 to 64 at M = 3 and from 67 to 71 at M = 4.
- **What it means for the patch:** the M = 3 failure comes from the build, not the arithmetic, and it sits on a row
  count the configuration never sends to int8. That makes it a reason to judge W12b-ptr r2 on the configuration's own
  row counts. It is not a reason to rewrite r1's verdict, which stands as recorded.

## W13 mode 2, speed of record F2, R3 against rafatxf, and W12b-ptr r2 (pre-registered 2026-10-31 13:42 EDT, before any of these steps runs)

Queue rev 40 adds three steps after the running `speed-all`, in this order.

**1. `x-w13-x2`: X' (exactness), for W12b-ptr r2.**
- **What runs:** `w12c_check.py dump` on the adopted MTP configuration (w13 build, `EXL3_GEMV_SK=1`,
  `EXL3_INT8_SQ_MAX_M=2`), compared with `w12p_ref_w11b.pt` (w11b at `EXL3_INT8_SQ_MAX_M=4`, dumped 11:56 EDT).
- **Rule:** all 40 cases at M = 1 and 2 (5 shapes x K 1-4 x M 1-2) are bit-identical. These are the row counts the
  configuration sends to int8. M = 3-7 differ by design (split-K or fp16 against int8) and are not judged.

**2. `x-final2`: speed after W13.** 14 arms, 3 interleaved rounds:
- **Undrafted:** stock at fill 0; then w11b (adopted undrafted, `EXL3_INT8_SQ_MAX_M=1`) and the w13 build
  (`EXL3_GEMV_SK=1`, `EXL3_INT8_SQ_MAX_M=1`), each at fill 0 and 28K.
- **MTP:** stock at fill 0; then w11b (`EXL3_INT8_SQ_MAX_M=2`), w13 mode 1, w13 mode 2 and rafatxf, each at fill 0
  and 28K.
  - The rafatxf arms use exactly R1-redo's settings (`-cq 4,4`, gcc-13 bindir), its best configuration, so they
    carry a Q4 KV cache against our FP16. This is a comparison of best configurations, as in R1.
- Load average is logged at the start and the end.

The rules for `x-final2`:
- **A1' (W12b-ptr r2 decision):** undrafted at fill 0, the w13 build is at least 3% faster than w11b in pooled tok/s
  in every round. Undrafted, the w13 build differs from w12p only by code that M = 1 never reaches, so this measures
  the pointer.
- **W12b-ptr r2 adoption:** X' and A1' both hold.
  - The undrafted configuration then moves to the w13 build with `EXL3_INT8_SQ_MAX_M=1`, so one build serves both
    modes.
  - Quality carries over from T3/T4 through bit-identity at M = 1 (X'), as the r1 pre-registration said.
  - If either fails, undrafted stays on w11b.
- **The whole chain under MTP (reported):** w13 mode 1 against w11b at both fills, the pointer included.
- **R3 (reported; the question R1 opened):** w13 modes 1 and 2 against rafatxf with MTP, per round, at both fills.
  - "W13 closes rafatxf's MTP lead" is written only if mode 1 is faster in every round at fill 0.
  - Otherwise the per-round numbers are given as they are.
- **Speed of record (reported):** pooled tok/s for every arm, and adopted over stock at fill 0 in both modes.

**3. `x-q-w13s2-mtp`: mode 2's T4.**
- **What runs:** GSM8K-500 with MTP on the w13 build with `EXL3_GEMV_SK=2`, label `TURING-exl3-mtp-w13s2`.
- **Rule:** against `TURING-exl3-mtp-w11j2`, the accuracy change is within -1.0 point, with discordant pairs
  reported. It is also reported against `TURING-exl3-mtp-w13` (mode 1) and against stock MTP.
- **Mode 2's X is already in hand.** The follow-up rule asked for mode 2's own X and T4. Its X was part of the
  amended X gate, computed before any A/B, and it holds (`w12/w13_check_sk2.jsonl`: M = 1-2 bit-identical 40 of 40;
  M = 3-7 maximum 6.39e-4, median 4.99e-4). This T4 is the remaining part.
- **Mode 2 adoption:** X, the margin over mode 1 in A (held at 13:41), and this T4.
  - If all three hold, the MTP configuration of record moves to `EXL3_GEMV_SK=2`.
  - Mode 2 drops rafatxf's shape limits, so it also routes MLP gate/up (n = 17408) and attn k/v (n = 1024) to
    split-K.

**Predictions:**
- A1': +4 to 5% (the r1 diagnostic gave +4.5%).
- The chain under MTP: +20 to 25% over w11b at fill 0.
- R3: mode 1 about level with rafatxf at fill 0, and mode 2 ahead by 10% or more. Across sessions the numbers were
  67.6 and 72.2 against 63.6, which cannot be compared directly.
- T4 for mode 2: inside the bound.

## X', F2 and R3: results; W12b-ptr r2 adopted (recorded 2026-10-31 14:56 EDT)

Queue rev 40 ran the first two steps of the 13:42 pre-registration:
- `x-w13-x2` ran 14:08:35-14:08:45 EDT.
- `x-final2` ran 14:08:45-14:51:56 EDT. Load average was 3.19 at the start and 10.91 at the end, from the other
  project's headless Chrome swarm. The card ran at 260 W and 69-78 °C.

Files: `w12/w13_x2.jsonl` and `w12/run-x-w13-x2.log`; `ab/final2.jsonl`, `ab/final2.md`, `ab/run-x-final2.log` and
`ab/final2-load.log`. `x-q-w13s2-mtp` (mode 2's T4) started at 14:52:56 EDT and is still running.

**X' holds.** All 40 cases at M = 1 and 2 are bit-identical (20 of 20 at each), with maximum relative difference 0.
The cases outside the rule were not judged, and are recorded as found:
- M = 3-4: 0 of 20 identical, maximum 9.4e-3. The reference ran int8 there, under `EXL3_INT8_SQ_MAX_M=4`.
- M = 5-7: 11 of 20 identical at each M, maximum 6.39e-4. The differing cases are the shapes inside rafatxf's limits,
  where split-K replaces the fp16 path.

**A1' holds.** Undrafted at fill 0, the w13 build against w11b in pooled tok/s:
- per round: +4.5%, +5.0%, +4.3%, faster in 3 of 3 rounds and at least 3% in each;
- all rounds: +4.6% (47.53 against 45.45 tok/s);
- at 28K, reported: +4.7% (43.88 against 41.91), faster in 3 of 3 rounds.
- The prediction was +4 to 5%. It held.

**Decision: W12b-ptr r2 is adopted.** X' and A1' both hold.
- The undrafted configuration of record moves to overlay `exl3-w13` with `EXL3_GEMV_SK=1`, `EXL3_INT8_SQ_MAX_M=1`,
  `EXL3_MGEMM_N_THRESHOLD=0` and `EXL3_FD75D_LIVE=0`.
- One build now serves both modes; the MTP configuration of record differs only in `EXL3_INT8_SQ_MAX_M=2`.
- Undrafted quality carries over from T3 and T4 through X' (bit-identical at M = 1).
- This closes the caveat recorded at 13:41, that the w13 build contains W12b-ptr. The pointer is now adopted in its
  own right.

**The whole chain under MTP (reported).** w13 mode 1 against w11b (`EXL3_INT8_SQ_MAX_M=2`), in pooled tok/s:
- fill 0: +22.8%, +21.0%, +16.1% per round; all rounds +20.0% (65.50 against 54.60);
- 28K: +22.6%, +19.9%, +21.8%; all rounds +21.4% (61.33 against 50.51).
- The prediction was +20 to 25% at fill 0. It held, at its lower edge.

**R3 (reported).** Each arm against rafatxf with MTP, using R1-redo's settings (`-cq 4,4`, gcc-13 bindir), in pooled
tok/s per round:

| arm against rafatxf | fill 0: r1 | r2 | r3 | all rounds | 28K: r1 | r2 | r3 | all rounds |
|---|---|---|---|---|---|---|---|---|
| w13 mode 1 | -0.8% | -1.3% | -1.1% | -1.0% | +2.1% | -5.3% | -5.2% | -2.8% |
| w13 mode 2 | +11.7% | +5.2% | +7.6% | +8.2% | +9.2% | +6.5% | +1.0% | +5.5% |

- **Mode 1 was slower in all three rounds at fill 0.** By the rule, "W13 closes rafatxf's MTP lead" is not written.
  The lead that remains is 1.0% at fill 0 (66.18 against 65.50) and 2.9% at 28K (63.09 against 61.33).
- **The lead W13 shrank, in this session:** rafatxf over w11b was +21.2% at fill 0 (66.18 against 54.60) and +24.9% at
  28K. R1's +16.7% was measured with fdq4 falling back to Triton.
- **Mode 2 was faster than rafatxf in every round at both fills:** +8.2% at fill 0 and +5.5% at 28K. It is not adopted
  until its T4 is in.
- **Predictions:**
  - "Mode 1 about level with rafatxf" held, at -1.0%.
  - "Mode 2 ahead by 10% or more" failed, at +8.2%.
- **This compares best configurations,** as R1 did. rafatxf carries a 4-bit KV cache and our arms the 8-bit one.
- **Draft acceptance was the same in all nine MTP arms:** medians 0.632-0.663, pooled 0.535-0.556. The differences
  are kernel time.

**A correction to the 13:42 text.** It says the rafatxf arms "carry a Q4 KV cache against our FP16". Our arms ran the
8-bit cache (`cache_quant` "8,8" in every row), as R1 and every adopted configuration have. rafatxf ran "4,4". No rule
depended on it. The text above is kept as written.

**Speed of record F2** (pooled tok/s, 2080 Ti at 260 W, one session, `ab/final2.md`):

| mode | fill | stock | w11b | adopted (w13 build) | w13 mode 2 (T4 pending) | rafatxf |
|---|---|---|---|---|---|---|
| undrafted | 0 | 25.33 | 45.45 | 47.53 | - | - |
| undrafted | 28K | - | 41.91 | 43.88 | - | - |
| MTP | 0 | 46.40 | 54.60 | 65.50 | 71.61 | 66.18 |
| MTP | 28K | - | 50.51 | 61.33 | 66.57 | 63.09 |

- **Adopted over stock at fill 0:** undrafted +87.6% (1.88x, per round +86.1%, +88.6%, +88.3%); MTP +41.2% (1.41x,
  per round +43.5%, +44.7%, +35.5%). Mode 2 would make the MTP figure +54.3%.
- **MTP over undrafted in the adopted build:** +37.8% at fill 0 and +39.8% at 28K. At F1 it was +16.5% at fill 0
  (49.32 against 42.33).
- **Session drift,** for reading across sessions:
  - Stock undrafted at fill 0 ran 25.33 here against 23.40 at F1, and w11b ran 45.45 against 42.33.
  - w13 mode 1 with MTP at fill 0 ran 65.50 here against 67.62 in `w13ab1`.
  - Only comparisons within a session are claims.

## W14: draft settings on the W13 build, and R4: MTP censuses at 28K with fdq4 running (pre-registered 2026-10-31 15:01 EDT, before any of these steps runs)

**Why now.** W8 (30 October, stock wheel) tested the sparse draft head and the draft length in one drifting sweep:
- sdh 16,384 met its rule by the letter (+12.4%), but against the adjacent `base` it was +3%;
- "It goes to the interleaved repeats before any claim" was written then, and that repeat never ran.

W13 has since cut the verify rows' weight time from 15.06 to 10.70 ms (mode 2 census), so the draft step's share of
each MTP cycle has grown. The draft step includes the full 248,320-column head read once per drafted token.

The sparse head keeps an adaptive token-frequency profile and saves it at exit, so it carries state between runs.
That state is controlled as follows:
- **One location.** Every W14 arm runs with `XDG_CACHE_HOME=$L/cache`, which puts the profile under
  `$L/cache/exllamav3/token_stats/`. No profile exists there or in `~/.cache` now.
- **Warmed in advance.** One sdh process (`x-w14-warm`, rows discarded) warms the profile before the A/B, so all
  three rounds start warm, as a daily server would.
- **Logged.** The profile file's size and mtime are logged before and after the A/B.

**The configuration (M).** The MTP configuration of record at the time the step is queued: overlay `exl3-w13`,
`EXL3_INT8_SQ_MAX_M=2`, `EXL3_MGEMM_N_THRESHOLD=0`, `EXL3_FD75D_LIVE=0`, with `EXL3_GEMV_SK=2` if mode 2's T4
(`x-q-w13s2-mtp`, running) holds under the 13:42 rule, and `EXL3_GEMV_SK=1` otherwise. That rule is already fixed,
so this leaves no choice open.

**`x-w14-ab`:** 6 arms, 3 interleaved rounds, `-cs 32768`, all on M:
- fill 0: `base` (`-mtp -ndt 6 -dds`), `sdh16` (`-sdh 16384` added), `ndt8` (`-ndt 8`) and `ndt4` (`-ndt 4`);
- fill 28K: `base` and `sdh16`.

**Rules:**
- **S (sparse head, speed and acceptance):** at fill 0, `sdh16` is at least 3% faster than `base` in pooled tok/s in
  every round, and its pooled draft acceptance is within 0.03 of `base`'s.
- **S adoption:** S and then a T4 with `-sdh 16384` on M, GSM8K-500 within -1.0 point of `TURING-exl3-mtp-w11j2`,
  discordant pairs reported. That T4 is queued only if S holds.
  - Verification keeps the full head, so the sparse head can change which tokens are proposed but not the verified
    distribution.
  - It can still change verify batch shapes, and with them the bits, as any MTP setting does (W8), so the T4 is
    required anyway.
- **L (draft length, reported):** `ndt8` and `ndt4` against `base`, per round. A length that is at least 3% faster in
  every round gets its own T4 pre-registration before any adoption.
- **28K (reported):** `sdh16` against `base`.

**Predictions:**
- S: +4 to +10% at fill 0, with acceptance within 0.01.
- 28K: a smaller gain than at fill 0, because attention takes a larger share there.
- `ndt8`: within ±3% of `base`.
- `ndt4`: 3 to 8% slower.

**`x-census-r4`:** three MTP censuses, diagnostic, with no decision:
- M at 28K;
- rafatxf at 28K with fdq4 running (R1-redo's settings: `-cq 4,4`, `NVCC_PREPEND_FLAGS=--compiler-bindir=/usr/bin/gcc-13`);
- rafatxf at fill 0 with fdq4 running. R2's rafatxf census ran with fdq4 falling back to Triton, which R1f says
  matters little at fill 0; this measures it.

**Rules:**
- **Gate.** The rafatxf censuses count only if the step log has no "kernels unavailable" line and the census lists
  fdq4's own kernels (`fdq4_bc_split` and `fdq4_bc_combine`, the names in `attention_fn/fdq4.py`).
- **R4a (reported):** decode attention (split plus combine) and weight kernels, in ms per decoded token, for M against
  rafatxf at 28K.
- **R4b (reported):** the depth cost per decoded token (span at 28K minus span at fill 0) for each build.

**Prediction:** at 28K, attention differs by 0.5 ms per decoded token or less. Then fdq4's speed value at 28K is under
3%, and its case is capacity (half the KV bytes, so twice the context in the same memory), which would need its own
quality pre-registration.

**Order:** after `x-q-w13s2-mtp` and before `sweep`, in queue rev 41: `x-w14-warm`, `x-w14-ab`, `x-census-r4`.

## Mode 2's T4: adopted; W14 amended before it runs (recorded 2026-10-31 15:34 EDT)

**`x-q-w13s2-mtp`** ran 14:52:56-15:27:37 EDT. Files: rows in `quality/TURING-exl3-mtp-w13s2/`, the comparison in
`quality/compare_t4_w13s2.md` (and `.json`), the log in `quality/run-x-q-w13s2-mtp.log`, and the run record in
`quality/quality_runs.jsonl`.

GSM8K-500 with MTP, w13 build at `EXL3_GEMV_SK=2`:

| against | accuracy A / B | delta | discordant A-only / B-only | McNemar p | same final answer | byte-identical |
|---|---|---|---|---|---|---|
| `TURING-exl3-mtp-w11j2` (the rule's reference) | 97.6% / 97.6% | +0.00 | 0 / 0 | 1.00 | 499/500 | 0.896 |
| `TURING-exl3-mtp-w13` (mode 1) | 97.2% / 97.6% | +0.40 | 0 / 2 | 0.50 | 497/500 | 0.878 |
| `TURING-exl3-mtp` (stock) | 97.6% / 97.6% | +0.00 | 0 / 0 | 1.00 | 500/500 | 0.874 |

- **T4 holds:** +0.00 points against w11j2, inside the -1.0 bound. The prediction ("inside the bound") held.
- **Mode 1's two misses are back.** Mode 1 lost gsm8k_12 and gsm8k_505 to w11j2. Mode 2 agrees with w11j2 on every
  item's correctness, so both are correct again. The two items moved with the verify shapes, not with a loss of
  accuracy, which is what the 13:41 entry suspected when it chose to watch the direction rather than call it a
  failure.
- **Efficiency on the eval** (decode medians from the eval server, not a speed claim):
  - wall per item 4.07 s, against 4.47 s for mode 1 and 5.44 s for w11j2;
  - accuracy per minute 14.39, against 13.03 and 10.77;
  - median decode 85.9 tok/s, against 79.0 and 63.2.
  - Draft acceptance is 0.622 in all three.

**Decision: mode 2 is adopted.** X held before any A/B, A's margin over mode 1 held at 13:41 (+6.8% pooled at fill 0,
+7.5% at 28K, faster in every round) and was confirmed in F2 (+9.3% and +8.5%), and T4 now holds.

**The configurations of record:**
- **MTP:** overlay `exl3-w13`, `EXL3_GEMV_SK=2`, `EXL3_INT8_SQ_MAX_M=2`, `EXL3_MGEMM_N_THRESHOLD=0`,
  `EXL3_FD75D_LIVE=0`, `-mtp -ndt 6 -dds`, Q8 cache.
- **Undrafted:** the same build with `EXL3_GEMV_SK=1` and `EXL3_INT8_SQ_MAX_M=1`, as adopted in r2. A1' measured that
  setting. SK=2 undrafted has not been measured, so it is not claimed.

The speed of record with MTP is therefore F2's mode 2 arm:
- 71.61 tok/s at fill 0, 1.54x stock MTP (per round +61.7%, +54.3%, +47.4%);
- 66.57 tok/s at 28K.

Against rafatxf's best configuration in the same session, mode 2 is +8.2% at fill 0 and +5.5% at 28K, faster in
every round at both fills (R3).

### W14 amendment (before any W14 step runs)

1. **M is mode 2** (`EXL3_GEMV_SK=2`), by the 15:01 rule and the decision above.
2. **Order.** `sweep` started at 15:27:37 EDT, before the queue could be rotated, so rev 41 puts `x-w14-warm`,
   `x-w14-ab` and `x-census-r4` right after it instead of before it.
3. **A component that would not have run.** `tools/turing/census.py`, the measuring tool behind `ab_speed.sh`, parsed
   `-sdh` but never passed it to the `Generator`. Every `sdh16` arm would have drafted through the full head while
   labelled sparse: W10a's and R1's failure again. The fix:
   - `census.py` passes `sparse_draft_head` when it is set. Builds without the option, such as rafatxf's 1.5.3,
     keep their call.
   - Every row records `sparse_draft_head` and `sparse_head`: the sparse head's state after the decode (enabled,
     sparse mode, active columns, rebuilds, mode switches, lookups, misses, miss rate, probes), or null for the full
     head.
   - The helper was tested in the sandbox, and the desktop copy is md5 `451c9f44…`, equal to the repo's.
   - W8 measured sdh through `draft_sweep.py`, which does pass the option, so W8's sdh arms did run.
4. **A gate in `x-w14-ab`.** Before the first arm, the step checks every `w14warm` row for a non-null `sparse_head`.
   It prints "W14 GATE: sparse head missing in the warm rows, A/B skipped" and runs nothing if one is missing.
5. **Reported with S:** each `sdh16` row's `sparse_head` state. The sparse head's cost model can switch itself back
   to the full head when the sparse path does not pay ("the measured miss rate then switches the sparse path off").
   An `sdh16` arm that ran in full-head mode is reported as such, beside its speed.

## The llama.cpp `sweep`: twelve tries lost to one deterministic out-of-memory; skipped by hand (recorded 2026-10-31 20:08 EDT)

**What happened.** `sweep` (`tools/llamacpp_sweep.sh`, the llama.cpp baseline sweep, in the queue since rev 2-4) started
at 15:27:37 EDT and ran twelve times, about 21 minutes each, until 20:03:43. Every try failed the same 4 of its 21
llama-bench calls with "CUDA error: out of memory" (rc 134):
- call 10: IQ2_XXS, prompt processing, `-fa 0`, f16 cache;
- calls 19-21: IQ2_S, prompt processing, `-fa 1` with q8_0 and with f16 cache, and `-fa 0` with f16 cache;
- each with `-p 2048 -b 2048 -ub 256,512,1024,2048 -d 0,8192`.

The other 17 calls finished on every try. The script keeps the rows a call wrote before it aborted, and those rows
show that only the largest micro-batch failed:
- calls 10 and 19: `-ub 2048` at depth 8192 (depth 0 measured);
- calls 20 and 21: `-ub 2048` at both depths.

Every other (micro-batch, depth) point was measured, twelve times. The script exits 1 when any call fails, and a step
without the `x-` prefix gets 30 tries, so the queue would have spent about 10.5 hours repeating a failure that
cannot change.

**Cost.** 4 h 36 min of desktop GPU time. W14 and R4 waited behind it.

**Why nobody saw it.** The background watcher waited only for the W14 and R4 labels, so it never matched "not
finished: sweep". The retries came to light at 20:02 EDT, when Matt asked how things were going.

**Action, 20:05:19 EDT:**
1. The queue unit was stopped between tries, with no step unit running.
2. `runs/queue/sweep.gaveup` was written by hand, with the reason inside it, and a line in `logs/queue.log` says so.
3. The queue was relaunched with the command in `full_queue.sh`'s header.

`x-w14-warm` started at once. Nothing was deleted: the twelve run directories
`runs/sweep/20261031T152737-full` to `20261031T194233-full` stay on the desktop, each with its `summary.tsv`.

**W14's gate passed** (`logs/run-x-w14-ab.log`): "W14 GATE: sparse head built in all 3 warm rows".
- All three rows: enabled, sparse mode, capacity 16,384, no mode switches.
- Active columns 306, 308 and 363; draft acceptance 0.681 (code), 0.715 (math), 0.369 (prose).
- The profile started empty (no `token_stats` directory existed under `$L/cache`). The warm run wrote
  `e431518ee9ad94cd.pt` (995,013 bytes) at 20:05:54.
- `x-w14-ab` started at 20:05:59.

**Lessons:**
- **Retries are for transient failures.** A step that fails the same way on its second try is deterministic, and
  more tries only spend GPU time.
- **A capacity limit is a result, not a failure.** An out-of-memory at the largest micro-batch is a measurement of
  the card. The sweep should record it and still finish. A pre-registered fix to `llamacpp_sweep.sh` comes before
  it runs again.
- **Watch the head of the queue, not only the steps waited for.** The new watcher alerts on any step's second failed
  try, on any give-up, and on `x-census-r4` finishing.

## W14 and R4: results (recorded 2026-10-31 20:36 EDT)

`x-w14-warm` ran 20:05:19-20:05:59, `x-w14-ab` 20:05:59-20:20:39, and `x-census-r4` 20:21:40-20:25:50 EDT.

Files:
- W14: `ab/w14warm.jsonl`, `ab/w14ab.jsonl`, `ab/w14ab.md` (`ab_report.py`, run in the sandbox), `ab/w14-profile.log`,
  and the two run logs.
- R4: `census/d28-w13-mtp-sk2`, `census/raf-d28-mtp-fdq4` and `census/raf-d0-mtp-fdq4`, the comparisons
  `census/compare_r4_d28.md` and `census/compare_r4_depth.md`, and `census/run-x-census-r4.log`.

### W14 (M = mode 2, `-cs 32768`, three interleaved rounds)

Pooled tok/s per round; acceptance is accepted over (accepted + rejected) draft tokens, pooled:

| arm | r1 | r2 | r3 | all rounds | change vs base | acceptance |
|---|---|---|---|---|---|---|
| base, fill 0 (`-mtp -ndt 6 -dds`) | | | | 75.53 | | 0.560 |
| sdh16, fill 0 (+`-sdh 16384`) | -1.8% | +8.5% | +11.6% | 80.03 | +6.0% | 0.549 |
| ndt8, fill 0 | -7.1% | -1.1% | -3.9% | 72.46 | -4.1% | 0.541 |
| ndt4, fill 0 | -6.7% | +1.9% | -3.1% | 73.53 | -2.6% | 0.546 |
| base, 28K | | | | 65.49 | | 0.540 |
| sdh16, 28K | +10.1% | +15.7% | +8.4% | 72.96 | +11.4% | 0.549 |

- **Rule S does not hold.** At fill 0, sdh16 was +6.0% pooled over all rounds, but round 1 was -1.8%, so it was not
  "at least 3% faster than base in pooled tok/s in every round". Acceptance was within 0.03 (-0.011). The sparse head
  is not adopted, and no T4 is queued.
- **L:** neither draft length was at least 3% faster in every round. ndt8 was slower in all three rounds (-4.1%);
  ndt4 was -2.6%, faster in one round. `-ndt 6` stays.
- **28K (reported):** sdh16 was +11.4% pooled, faster in every round.

**Predictions:**
- S "+4 to +10%, acceptance within 0.01": the pooled gain was in range (+6.0%), but acceptance missed by 0.001
  (-0.011), and the rule failed on round 1.
- "28K a smaller gain": missed; it was larger (+11.4%).
- "ndt8 within ±3%": missed (-4.1%).
- "ndt4 3-8% slower": missed narrowly (-2.6%).

**The sparse head's state (the amendment's item 5).** Every sdh16 row drafted in sparse mode, with no mode
switches and `misses` 0. So no arm fell back to the full head, and the gate's proof held through the A/B.

The arm was not stationary, though:
- **The active set kept growing.** Its active columns grew from process to process: 502 → 740 → 970 at fill 0, and
  658 → 826 → 884 at 28K. Both fills share one profile file (`e431518ee9ad94cd.pt`), rewritten at every exit
  (mtime 20:05:54 after the warm-up, 20:20:31 after the A/B).
- **Acceptance and speed rose with it.** sdh16's acceptance at fill 0 went 0.521 → 0.552 → 0.574, against base's
  0.578, 0.552 and 0.550, and its speed rose with it: 83.3 → 86.3 → 89.7 tok/s by prompt mean.
- **Reading (not a result).** The warm-up's three prompts left the profile far from steady state, so round 1 timed
  a head that could not yet draft many of the tokens it needed. By round 3 it was ahead on acceptance as well as
  speed.
- **Train-on-test.** The profile learns from the very prompts being timed, which flatters later rounds. A clean
  test needs a profile warmed on a separate corpus until its active set stops growing, and then held fixed while it
  is timed.

**W14b**, not yet pre-registered: first check whether the sparse head can load a profile without updating it. If it
can, warm on text outside the A/B prompts, freeze, and rerun S. If it cannot, report the sparse head as a
self-warming feature, and measure it only after a long warm-up on other text.

### R4 (MTP censuses; gate passed)

**Gate.** The step log has no "kernels unavailable" line, and both rafatxf censuses list `fdq4_bc_split` and
`fdq4_bc_combine`, so fdq4 ran.

Kernel time per decoded token, summed over code, math and prose (one capture per task, a mechanism check, not a
speed claim):

| census | span ms/token | attention | weights | other | acceptance |
|---|---|---|---|---|---|
| ours, fill 0 (`d0-w13-mtp-sk2`, captured earlier on 31 October) | 13.56 | 0.45 | 10.80 | 1.20 | 0.576 |
| ours, 28K (`d28-w13-mtp-sk2`) | 14.70 | 1.57 | 10.92 | 1.22 | 0.570 |
| rafatxf, Q4 cache with fdq4, fill 0 | 14.95 | 0.34 | 12.36 | 1.20 | 0.559 |
| rafatxf, Q4 cache with fdq4, 28K | 15.67 | 1.17 | 12.24 | 1.19 | 0.557 |

How the columns are built:
- **Attention:** the decode-attention split and combine kernels, rope and the cache writer.
- **Weights:** every EXL3 GEMV and GEMM family.
- **Acceptance:** token-weighted.

- **R4a, at 28K.** Attention: ours 1.57 against 1.17 ms/token. The `fd75d_split_kernel` alone is 1.38 against
  `fdq4_bc_split`'s 1.02. Weights: ours 10.92 against 12.24.
  - rafatxf's 4-bit cache saves 0.40 ms/token in attention, and our weight path saves 1.32. The net is 14.70 against
    15.67 ms/token, consistent with R3's +5.5% at 28K.
  - **The prediction held:** "attention differs by 0.5 ms/token or less at 28K, so fdq4's case is capacity, not
    speed." fd75d reads twice fdq4's bytes per cached token for 1.35× its time.
- **R4b, depth cost (28K minus fill 0).** Ours +1.14 ms/token (attention +1.12). rafatxf +0.72 (attention +0.83).
  - The 4-bit cache grows more slowly with depth, as its bytes say it should.
  - On these two points the totals would meet near 93K tokens, a straight-line extrapolation and not a
    measurement. That is past the 32K cache measured here.
  - Our fill-0 census comes from an earlier session than the other three. Its 13.56 ms/token (73.7 tok/s) agrees
    with F2's 71.61 pooled tok/s.

**What R4 leaves:**
- fdq4's value to the lab is capacity: half the cache bytes, so a longer context fits in 11 GB.
- The speed contest is in the weights, where the lab leads.
- A 4-bit cache would need its own quality gate (T4 at long context) before any adoption. It is a candidate for the
  long-context lane, not for the speed of record.

## Correction: `layer_bits.py` folded the drafting layer into text layer 0 (recorded 2026-10-31 21:03 EDT)

Found while drafting Volume I.5, chapter 3. `tools/turing/layer_bits.py` matched `layers.N.` anywhere in a tensor
name, so the drafting layer's `mtp.layers.0.*` tensors overwrote text layer 0's entries. Text layer 0's three 1-bit
MLP tensors (gate, up and down) showed as 2-bit.

- `census/layer_bits_qwen38-exl3-2.0.txt` (kept as it was) therefore counts 11 tensors at 1 bit.
- The fixed tool counts 14 at 1 bit, and keeps the drafting layer under its own `mtp.` rows. The new output is
  `census/layer_bits_qwen38-exl3-2.0.v2.txt`. The 14 are:
  - 11 MLP matrices: gate and up in layers 0-2, 14 and 15, and down in layer 0;
  - attention q_proj in layers 7 and 11;
  - the GDN in_proj_qkv in layer 61.
- The new counts are 14 tensors at 1 bit, 342 at 2 (334 text plus 8 drafting), 39 at 3 (with the head) and 14 at 4.
  They agree with the quant's `recipe_stock20.yaml` and the research note's §5.1.

**Effect on earlier results: none found.**
- `census_bits.py` takes K from each kernel's template arguments, not from this table, so the W12 bit-width timings
  (`int8_by_bits.md`) stand.
- The dispatch table near the top of this note quotes typical K = 2 rows, which do not change.

## Corrections found while drafting Volume I.5, chapters 8 and 9 (recorded 2026-10-31 21:27 EDT; the text above is kept as written)

1. **ab1-ab3 were never re-read pooled.** The 10:56 correction (mean of per-prompt rates overstates MTP speed)
   recomputed only `rafatxf1` and `final1`. `ab_report.py` now writes the pooled figures for the three older runs
   to `ab/ab1.pooled.md`, `ab/ab2.pooled.md` and `ab/ab3.pooled.md`. The originals are kept. Pooled over all rounds:

   | pair | printed (mean of rates) | pooled |
   |---|---|---|
   | W7+G1 vs stock, undrafted | +57.0% | +57.1% |
   | W7+G1 vs stock, MTP | +4.6% | +5.6% |
   | F1 vs F0 (W9), undrafted | +10.3% | +10.4% |
   | J2 vs J0 (W10b), MTP | +1.8% | +2.8% |

   Each is faster in 3 of 3 rounds either way, so no decision changes.
2. **W10a's pre-registration says the M = 3-4 instances "compile to 64 registers".** W10b's R0 misses and the w11b
   register dump show 66-73.
3. **W7 "sized every instance at a 512-row cap".** In the source, the 80 KB caps at M = 3 and 4 are 360 and 280 rows.
   Those still ask for more than 80 KB, so the conclusion stands.
4. **W12c: "IMMA at M = 8 costs 1.68-2.13x its M = 1 time".** From `sqmb1` it is 1.68-1.98×. 2.13 is the
   dp4a-over-IMMA ratio at eight rows on MLP down. The rule fails either way.
5. **W12e's gap-closure shares do not reproduce from the medians.** The text has 46/26/22/44% (V13) and 52/26/19/48%
   (V8). The medians give 44.5/25.9/24.1/43.9% and 55.4/26.9/20.4/48.2%. The conclusion (V13 gains no more than V8)
   holds.
6. **The overlap ratio "1.25-1.44x"** starts at 1.24 (MLP gate 53.22/42.75 = 1.245).
7. **L2's reading that the remaining limit is the memory path, not SM clock, is not settled by L2.** Its own PW2 says
   the int8 path responds to clock, and the adopted build's +6.2% mean speed at 320 W is close to its roughly 7%
   clock rise. L2 cannot separate the two.
8. **W12 scouting's fixed cost.** The fit's ~12 µs per launch (about 4.8 ms per token over ~400 launches) was mostly
   extrapolation. The later bench with tiny and small shapes put the intercept at 7.42 µs, and an empty launch at
   5.6-7.1 µs. Read the 4.8 ms as an upper bound. (Volume I.5 ch02 now says so.)

## Corrections found while drafting Volume I.5, chapter 7 (recorded 2026-10-31 21:28 EDT; the text above is kept as written)

1. **R1's drafting ratios were means of per-prompt rates.** Pooled from `ab/rafatxf1.jsonl`, rafatxf's MTP is 1.98×
   its undrafted speed (63.02 against 31.85 tok/s) and ours 1.20× (54.02 against 44.99), not 2.11× and 1.28×. The
   reading (rafatxf's verify step is cheap relative to its one-row step) stands. Volume 0's two facts are updated.
2. **R2's shape description.** "68 x 512 threads" for our verify rows was the family table's most frequent shape,
   not where the time went. `d0-w11b-mtp`'s launch-shape table splits the 10.50 ms into 5.26 ms on 136 × 256 and
   5.01 ms on 68 × 512. R2's conclusion is unaffected.
3. **R4's "weights" column also counts the small GDN a/b GEMV**, because every family named `*gemv*` was summed.
   Without it the weight time is 10.70 ms (ours, fill 0) and 12.23 ms (rafatxf, fill 0). R4's comparison does not
   change.
4. **Open:** both MTP censuses from the R2 session show about 2.3-2.4 ms of host idle per token, against about
   1.0-1.1 ms in later MTP censuses. So the census span fell more from W11b to W13 (-29.5%) than F2's interleaved
   change (-23.7% in ms). The cause is not known, and cross-session census spans should not be read as speed changes.

## Corrections found while drafting Volume I.5, chapter 10 (recorded 2026-10-31 21:45 EDT; the text above is kept as written)

1. **depth2 was never re-read pooled.** The pooled report is now `ab/depth2.pooled.md`, beside the original.
   - W11 with MTP at 28K is 33.85 → 47.75 tok/s pooled, not "37.5 to 52.1" (means of per-prompt rates). The ratio
     is +41.1% either way.
   - W11 undrafted at 28K is +37.5% pooled (printed +38.1%), and the control +2.6% either way.
   - No decision changes.
2. **R4: "fd75d reads twice fdq4's bytes".** With the scales counted it is 1.89×: 2,176 against 1,152 bytes per
   token per layer. "For 1.35× its time" stands.
3. **R4's comparison tables list only the largest launch shapes.** `compare_r4_d28.md` and `compare_r4_depth.md`
   show fd75d at 1,319 µs, where the kernel's full total is 1,381, and fdq4 at 559 + 293 µs, where its total is 1,024.
   R4's text uses the full totals.
4. **W11's bench "floor" assumed 550 GB/s.** At the rated 616 GB/s, fd75d's 145 µs at 28K is about 1.47× the floor,
   not "within 1.3x".
5. **`tools/turing/fd75d_test.py`'s docstring said its reference is fp32.** `ref_attn` computes in fp64. The
   docstring is fixed, and every stored test result stands.

## Corrections found while drafting Volume I.5, chapter 11 (recorded 2026-10-31 21:45 EDT; the text above is kept as written)

1. **Mode 2's changed cases.** The W13 gate entry says mode 2 "also changes mlp_gate, 75 of 100 cases". The rows
   (`w12/w13_check_sk2.jsonl`) show it also changes attention k/v at K = 2-4: the 75 is 45 + 15 + 15. Every changed
   case stayed inside X's numerics bound.
2. **Open: no exactness check at eight rows or more.** Gate X covered M = 1-7, after the amendment narrowed it from
   3-8. W14's `ndt8` arm reaches nine-row verify passes, and it was timed but never checked for exactness or
   quality. Any adoption of a draft length above 6 needs that check first. None is planned, because ndt8 was slower.
3. **Bench noise.** "Same-code M = 1 pairs vary 0.87-1.04" holds at K = 2. Across K = 1-4 the same pairs range
   0.87-1.26.
4. **Time zones in quality reports.** The "generated" stamps in `quality/compare_t4_w13.md` (10:35:44) and
   `compare_t4_w13s2.md` (12:31:40) are laptop-local Pacific time, where the reports were written, against runs
   that started at 12:46 and 14:52 EDT. Read them as PDT.

## Correction: energy per token was a median of per-row ratios (recorded 2026-10-31 22:09 EDT; the text above is kept as written)

Found while drafting Volume I.5, chapter 15. `tools/turing/energy.py` gave each arm the median over its rows of
tokens per joule, and the median of tok/s. That is the flaw the 10:56 correction fixed in `ab_report.py`. Under MTP,
prose rows run far slower than code and math rows, so the median overstates the arm.

**Fix.** `energy.py` now also reports pooled figures (total tokens over total time, and over total joules) and
compares pairs on them. It was checked against the F1 telemetry: adopted MTP at fill 0 is 0.1945 pooled against
0.2250 by the median. `tools/turing/energy_pooled.py` recomputes pooled figures from a stored energy JSON. The
pooled reports are `ab/final1_energy.pooled.md` and `ab/power1_energy.pooled.md`, beside the originals.

**What changes:**
- **EJ3 (line 1536).** The adopted build gives 0.166 tokens/J undrafted and 0.195 with MTP at fill 0 (pooled), not
  "0.16-0.17 and about 0.23". Against the laptop's 0.559 drafted, the 2080 Ti spends about 2.9× the energy per
  token, not 2.5×. Paper 7's laptop figure comes from a different tool and is not re-read here.
- **EJ2 holds on pooled figures.** Adopted over stock, the speed and tokens-per-joule ratios agree within 1.7% in all
  four conditions: 1.808 against 1.826, 1.136 against 1.134, 2.137 against 2.160, and 1.388 against 1.411.
- **PW4 (line 1807).** At 320 W, tokens per joule falls by 10.6-18.5% in all eight configurations (ratios
  0.815-0.894), not "2-19% (ratios 0.806-0.977)". Adopted MTP at fill 0 is 0.858, not 0.977.
- **Line 1809.** The 290 W "exception" (adopted MTP at fill 0, 1.07) is 0.950 pooled: no configuration gains energy
  efficiency at 290 W either.
- **L2's decision (stay at 260 W) is unchanged, and stronger:** more power never paid in energy.

Volume 0's chapters 7 and 8 and Volume I.5's chapters 1 and 17 quote the median figures, and they are being updated
from the pooled reports.

## Corrections found while drafting Volume I.5, chapter 17 (recorded 2026-10-31 22:09 EDT; the text above is kept as written)

1. **The 10:56 correction's F1 "42.57 tok/s pooled"** was computed from `final1.jsonl`, which still holds the six rows
   of the guard-stopped first attempt. From `final1.clean.jsonl` the adopted arm is 42.21 and stock 23.35 (not
   23.65). The F1 ratios in Volume 0 were already computed from the clean file.
2. **The F2 entry mixed averaging methods.** "At F1 it was +16.5% (49.32 against 42.33)" divides a pooled MTP speed
   by a mean-of-rates undrafted one. Pooled against pooled it is +16.8%. Its session-drift line likewise compares
   F2's pooled stock 25.33 with F1's mean-of-rates 23.40; the pooled F1 stock is 23.35.
3. **F1 at 28K, pooled,** is 2.14× undrafted (39.96 against 18.70) and 1.39× with MTP (47.06 against 33.90), not
   2.13× and 1.38×. Volume 0's two facts are updated.
4. **W9's +10.4% was measured against the W9 build with its fit switched off**, which ran 2.7% behind W7 + G1 in that
   session. Against the previous adopted configuration (W7 + G1) the gain is +7.4% (`ab/ab2.pooled.md`).
5. **depth2's +41.1% with MTP at 28K** came from the live-splits arm (`w11Lm-28k`, `EXL3_FD75D_LIVE=1`), not the
   adopted setting. Live splits were a null, so the size of the gain stands. Volume I.5 ch10's fact note now says so.

## Correction: the speed tools sampled; they were never greedy (recorded 2026-10-31 22:13 EDT; the text above is kept as written)

Found while drafting Volume I.5, chapter 14. `tools/turing/census.py`, `smoke.py` and `draft_sweep.py` build their
`Job` without a sampler. exllamav3's `Job` then uses `DefaultSampler()`: min-p 0.08, temperature 0.8, unseeded
(checked in `engines/exl3-1.4.7-lab/exllamav3/generator/job.py` and `sampler/presets.py`; the agent read the
same in the 1.6.5 source). All three docstrings, and this note, said greedy. `ab_speed.sh` runs `census.py`, so every
census, every interleaved A/B (ab1 to w14ab, F1 and F2), every smoke run and every draft sweep on the 2080 Ti
decoded sampled text.

**What stands:**
- **Every speed comparison.** All arms sampled the same way, so the comparisons are fair. Sampling adds variance
  (different texts, different token counts per prompt), and the interleaved rounds and the every-round rules carry
  that variance.
- **The exactness gates.** X and X' test kernels on fixed inputs, and T3 (`numerics.py`) generates by argmax.
- **The quality gates.** T4 and the GSM8K and MMLU lanes ran through TabbyAPI at temperature 0.

**What does not stand:**
- **Any claim drawn from output hashes of the speed tools.**
  - W8's "MTP replies vary because verify shapes change": its two undrafted runs in one process also differed, on 0
    of 3 hashes, with different token counts.
  - `draft_sweep.py`'s purpose "to see whether a setting changes the greedy output".
- **Acceptance figures as greedy acceptance.** Every census and A/B acceptance figure is sampled acceptance, at
  temperature 0.8, and is lower than greedy acceptance would be. Compare it only with other sampled figures.

**Tool docstrings** are corrected, with no behaviour change, so measurements stay comparable with the record. A
greedy option for the speed tools would need its own pre-registration, because it would change acceptance and so
speed.

**Further points from the same draft:**
1. **W8 was never re-read pooled.** `-sdh 16384` was +7.1% against the mean of the base repeats (recorded +12.4%),
   and -1.1% against the adjacent base (recorded +3%).
2. **W14's `misses` 0 is not evidence.** In 1.6.5 a miss counts only if the token is still outside the subset
   after the next rebuild, and while the subset has room every new id is added. The probe rounds found the full
   head's pick outside the subset at 7 of 70 positions. The proof that no arm fell back to the full head is
   `mode_sparse` true with 0 mode switches, not `misses`.
3. **W14's +11.4% at 28K exceeds a constructed ceiling of about +8%** for removing the full head entirely
   (Volume I.5 ch14, `calc.sdh_ceiling`). The cause is open. W14's speeds 83.3 → 89.7 above are means of per-prompt
   rates (pooled 76.7 → 83.7), and its active-column counts are first-prompt values per process.
4. **Two descriptions of 1.6.5's recurrent state are 1.4.7's design.** `research/TURING_EXL3_RESEARCH_2026-10-30.md`
   §5.2 says the 1.6.5 verify pass "writes one state per token". 1.6.5 keeps two rows and replays the last pass's
   inputs. `draft_sweep.py`'s "recurrent snapshots" comment also describes 1.4.7.

**Residual (recorded 2026-10-31 22:16 EDT).** `ab/power1.md` predates the 10:56 pooled fix, so L2's speed figures there are means of
per-prompt rates. Adopted fill 0 at 320 W is +6.2% that way and +5.9% pooled. The L2 decision does not change, and
the speed facts the books take from `power1.md` are not yet re-read pooled.
