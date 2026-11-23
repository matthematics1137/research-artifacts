# Pre-registration: four-resources tests 1-3 (CPU only, data already on disk)

Written 2026-11-01 18:22 EST, before any fit, by Claude under Matt's go-ahead ("lets just do it ... doesnt need to be
paper yet sounds like this is an interesting viz/ backend thing?"). Gate: `research/gates/2026-11-01-four-resources-one-graph.md`.
Designs: `research/FOUR_RESOURCES_PROGRAM_2026-11-01.md`, section 5, tests 1-3. The predictions and kill criteria below
are copied from that report word for word and are frozen by this file's first commit. Any later change is a dated
amendment appended at the end. The first version stays.

These are lab measurements that feed the living map (the conversion table and its visual). They are not a paper.

## Common rules

- **CPU only.** No GPU job, no model download, no edit to any hashed or frozen harness (P5, P6 and P8 tools are read,
  never changed). Analyses run under `nice -n 10`.
- **Code and outputs.** Code goes in `tools/four_resources/` and outputs in `testsuite/results/four_resources/test<N>/`:
  a markdown report, a JSON file with every number in the report, and `rates.jsonl` rows in the living map's schema.
- **Order inside each test.** First write the predictions that this file says must be computed before fitting (test
  1's census predictions and test 2's training-set constants), with the clock time in the file. Then fit. Then compare.
- **Pooling.** Speeds are pooled as total tokens over total time, never as means of per-row rates (the 2026-10-31
  lesson).
- **Exclusions.** Rows are excluded only for the reasons named here. Each exclusion is counted and listed.
- **Deviations.** If a registered step cannot run as written (a missing field, a census that does not exist), the
  report says so plainly and labels any substitute analysis as exploratory. A falsified prediction is reported as
  falsified. No re-specification after seeing fits.

## Numbers already seen (disclosure)

All of these were seen before this file was written, most of them when the lab first recorded them:
- **power1:** the arm-level means and ms/token in `testsuite/results/turing/ab/power1.md`, and the pooled energy in
  `power1_energy.pooled.md`. These include each arm's speed at each cap: the caps barely move stock speed (23.24-23.64
  tok/s undrafted at fill 0). The arm-level clock and speed changes in the L2 results.
- **Census summaries:** W2 (undrafted decode is GPU-bound, and EXL3 matrix kernels take 92% of the time), the adopted
  census (the int8 GEMV takes about 85% of GPU-busy time), and R4 (fdq4 saves 0.40 ms/token in attention, our
  weights save 1.32).
- **Matrix timing at 1 and 2 bits:** `int8_by_bits.md` (53.5 µs at 1 bit, 54.8 µs at 2 bits).
- **final2 pooled speeds:** 47.53 tok/s for w13 undrafted at fill 0, 54.60 and 50.51 for adopted MTP at fill 0 and
  28K, 71.61 and 66.57 for w13 mode 2. The +4.6% undrafted and +20.0% drafted gains of w13.
- **Acceptance:** 0.66 for code, 0.68 for math and 0.40 for prose (six draft tokens).
- **P6:** pooled per-cell speeds, computed from the journals on 2026-11-01.
- **W12c:** the per-matrix row costs at M = 1-7, seen when W12c was written up.
- **P8 quality lanes**, scored 2026-11-01 18:20 EST. They do not enter these tests.

No clock elasticity, no priced-graph fit and no composed drafting cycle has been computed by anyone in the lab.

## Test 1. Clock elasticity

**Data.**
- `testsuite/results/turing/ab/power1.jsonl`: 216 rows.
- `testsuite/results/turing/ab/power1_telemetry.csv`: 1 Hz samples.
- Each row's decode window is the window `tools/turing/energy.py` assigns it. The SM clock per row is the median SM
  clock over the telemetry samples in that window.
- The response is ms/token = 1000 × `decode_wall_s` / `new_tokens`.

**Configurations.** Eight: build (stock 1.6.5; adopted w11b) × mode (undrafted; MTP) × fill (0; 28K). Each holds 27 rows:
3 caps × 3 tasks × 3 rounds.

**Model.** Fitted per configuration by OLS: ln(ms/token) = task + round − s·ln(f_SM). The cap does not enter, because it
acts through f_SM. The memory clock is 6,800 MHz in every arm.

**CI.** A percentile bootstrap over rows within the configuration: 10,000 resamples, seed 20261101. OLS HC3 intervals
are reported beside it as a check.

**Exclusion.** A row whose window holds fewer than 3 telemetry samples is excluded and counted. If more than 5 of a
configuration's 27 rows are excluded, the configuration is reported as not testable.

**Census prediction, written before the fit.** s_pred = Σ_k t_k·σ_k / T_token. The sum runs over the kernel classes k
of the configuration's census:
- t_k is the per-token kernel time.
- σ_k = 1 − min(1, (bytes_k / t_k) / 598 GB/s): the share of a kernel's time above its bandwidth floor, which scales
  with SM cycles.
- Host idle has σ = 0.
- T_token is the census's per-token wall time.

Where to get the inputs:
- Bytes per class come from the census's own byte accounting (`census_bits.py`, `weight_bytes_*.txt`,
  `layer_bits_*.v2.txt`).
- A class with no byte accounting gets σ = 1. The report flags it and also gives s_pred with σ = 0 for those classes,
  so the result is bracketed.

Census assignment:
- Stock undrafted fill 0: `w2`. Adopted undrafted: `d0-f1-mgemmn0` and `d28-f1-mgemmn0`. Adopted MTP: `d0-w11b-mtp`.
- Any configuration without a census of its own build, mode and fill gets no prediction. It is left out of the "3 or
  more of 8" count, and the count of predictable configurations is reported.

**Registered predictions and kill criteria (verbatim, report section 5):**

> Predicts: adopted, undrafted, fill 0: s = 0.55-0.9 (the census suggests about 0.8, since the int8 GEMV is 85% of
> GPU-busy time); 0.03-0.10 lower at 28K; stock undrafted s ≤ 0.3 at under 35% of bandwidth, a latency-bound third
> regime; MTP arms 0.3-0.9; neither 2-bit configuration bandwidth-bound.
>
> Falsified if the adopted undrafted s has its 95% CI entirely below 0.3 (decode is memory-bound after all), or the
> census misses the fitted s by more than 0.3 in 3 or more of the 8 configurations. CIs wider than ±0.4 mean the
> existing data cannot tell; then defer to a clock-lock sweep, which needs Matt's extended sudoers line.

**Known risk, stated now.** The caps barely moved stock speed, so the SM clock may not vary enough within a configuration
to identify s. In that case the ±0.4 rule applies, and the result is "the data cannot tell". That is a result, not a
failure.

## Test 2. Bytes-only against the priced graph

**Bytes-only predictor (as in the report).**
- On the 2080 Ti with EXL3 2.0, it prices per token:
  - 6.63 GB of weights (5.726 GiB of text layers plus a 455 MiB lm_head;
    `testsuite/results/turing/census/weight_bytes_qwen38-exl3-2.0.txt`);
  - the fp32 GDN state read and written (2 × 48 × 3 MiB = 0.30 GB);
  - the Q8 KV at the given depth.
  It divides by 598 GB/s.
- On the laptop it prices P6's device, offloaded and KV bytes by placement.

**Priced graph.** It adds up, serially, max(bytes/BW, weights × c_K / f_SM) per kernel class, plus launch costs and the
measured host idle.

**Training set, declared now.** Node constants (c_K per bit-width K, launch cost, the host-RAM edge cost) are fitted only
on:
- the 2080 Ti stock undrafted census at fill 0 (`w2`) and the adopted undrafted census at fill 0 (`d0-f1-mgemmn0`);
- P6's 27B UD-Q2_K_XL and IQ3_XXS cells on the laptop.

**Held out:**
- final2's adopted and w13 arms;
- the 28K census (`d28-f1-mgemmn0`);
- P6's 27B Q4_K_XL cells and all Qwen3-8B cells.

The report's 80 W laptop census is a GPU job and is deferred. The held-out claims are judged without it, and the report
says so.

**Registered predictions and kill criteria (verbatim, report section 5):**

> Predicts: bytes-only underpredicts 2080 Ti undrafted time by 37-45% (75-86 tok/s against 47.53 measured); predicts
> llama.cpp IQ2_S 16-18% slower than IQ2_XXS (measured 7.7%) and +1.9 ms for the KV read at 32K (measured +8.75 ms);
> gets the sign wrong for the laptop 8B's KV contrast at 16K (predicts q4_0 1.3-1.45x faster than f16; measured
> 0.88-0.98x); and fits the EXL3 28K KV penalty (+1.5-1.9 ms against +1.73). The priced graph lands within ±12%, median
> at most 8%, except cells with attention on the CPU. A host-RAM edge costs 0.021-0.026 ms per offloaded MiB, 4.5-5.5x
> a VRAM-resident MiB. For capped arms, energy equals cap × time within ±3%.
>
> Falsified if bytes-only predicts every held-out 2-bit configuration within ±15%, or if the priced graph misses 2 or
> more held-out configurations by more than 20%.

**Output.** Besides the report, one priced-graph document per operating point in the living map's graph schema.

## Test 3. The drafting chain on one graph (CPU stage only)

**Inputs, as in the report.**
- The adopted MTP censuses: six draft tokens, with per-task acceptance.
- The undrafted census's node costs, including the 990 µs lm_head that every drafted token also pays.
- W12c's per-matrix row costs at M = 1-7.

**Composition.**
- T_cycle = T_draft + Σ node costs at R = d+1 + host idle.
- ms/token = T_cycle ÷ E[tokens per cycle].
- J/token = 260 W × time.

**Targets.** The pooled adopted MTP speeds, 54.60 tok/s at fill 0 and 50.51 at 28K, predicted without using drafted
kernel times.

**Deferred.** The w13 extension needs a 10-minute split-K bench at R = 1-8, which is a GPU job. The acceptance stage
needs a private logging patch. Both wait. The w13 halves of the prediction and kill criteria are reported as not run.

**Registered predictions and kill criteria (verbatim, report section 5):**

> Predicts: ms/token within ±10% (adopted MTP 18.32 ms pooled, w13 mode 2 13.97 ms); reproduces why w13 gains about
> four times more drafted (+20%) than undrafted (+4.6%), because split-K serves the multi-row verify passes; acceptance
> within ±0.03 per task.
>
> Falsified if the error exceeds 20% on any task, or the sign and rough size (within 2x) of the drafted-versus-undrafted
> gain difference is missed.

## What each result feeds

- Every fitted or composed number becomes a row in the living map (`research/livingmap/`), with its operating point,
  evidence paths, this file as its pre-registration and a RIFE item.
- Claims that survive become RIFE facts. A falsified claim becomes a RIFE fact too, with the correction it implies for
  the frame's commentary.

## Amendment 1 (2026-11-01 18:36 EST): a mislabelled census

The test-1 run found that `d0-f1-mgemmn0` and `d28-f1-mgemmn0` are build 1.6.5+w9, not the adopted w11b that this file
calls them. At 28K, `d28-f1` also runs the slow paged-attention kernel. Nothing is re-specified:
- **Test 1** used the census assignment as written. It disclosed the mismatch and gave `d28-w11-mgemmn0` as an
  exploratory comparison. Its fits were all unidentified, so no verdict depends on this.
- **Test 2's training set** is the files as named. Read it as "the w9 undrafted census at fill 0", which makes the
  held-out arms (w11b and w13) a cross-build prediction.

Any later rerun, such as the clock-lock sweep, should assign censuses by build.
