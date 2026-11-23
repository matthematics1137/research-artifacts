# P8 extension: the per-call split on a second GPU generation (RTX 4080 Laptop, Ada)

Registered 2026-11-05 10:32 EST (clock read inside the writing command). No laptop census of this model exists before this
file is committed. Matt, 2026-11-05: "Do it so that we can add it to the paper." RIFE: research-ledger-reference.

**Research gate.** This extends a line the gate already covers (P8; research/gates/2026-11-05-p8-novelty-recheck.md).
- **Cost:** under an hour, about 15 minutes of it on the laptop GPU.
- **Prior art:** the megakernel work on later GPUs (P8 cites spector2025megakernel), which shows the per-launch cost
  in kind. Nothing found splits a 2-bit model's matrix time into a fixed per-call part and a per-weight part, by bit
  width, on two GPU generations.

## Question

Does P8's finding on the 2080 Ti (Turing) hold on Ada? The finding has two parts:
- **(a)** at about 2 bits per weight, 1-bit matrix calls run no faster than 2-bit calls of the same shape;
- **(b)** most of the fitted matrix calls' time above the time to read their weights is a fixed cost per call.

## Design

- **Card.** The laptop's RTX 4080 Laptop GPU (sm_89, 12 GiB), at its settings as found. Its power limit (80 W
  enforced), clocks and fan are recorded and not changed. Its memory clock during decode is recorded from
  nvidia-smi.
- **Engine.** Stock exllamav3 1.6.5: the same wheel as P8's stock engine,
  `exllamav3-1.6.5+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl`, copied from the desktop's wheelhouse with its
  SHA-256 recorded. It is installed with `--no-deps` into its own folder and put first on PYTHONPATH over the laptop's
  `engines/exl3-env` (torch 2.10.0+cu128, Python 3.12). The env itself is not changed.
- **Setting.** `EXL3_MGEMM_N_THRESHOLD=0`, so the projections run unfused. This is the upstream setting both 2080 Ti
  censuses used (d0-f1-mgemmn0 for w9, d0-w12p).
- **Model.** The same EXL3 weights, revision SC_2.00bpw_H3, at ~/models/qwen38-exl3-2.0 (shard SHA-256 prefixes
  cf8966ebc885 and 2a72afda6bf0, checked before the run).
- **Run.** `tools/turing/census.py`, unchanged, under Nsight Systems 2024.6.2, as census.sh's undrafted config:
  - an empty context, the three speed prompts and 256 new tokens each;
  - a 16K 8-bit cache (`-cs 16384 -cq 8,8`), `-rcs 2`, chunk 1024;
  - three captures (`--capture-range=cudaProfilerApi --capture-range-end=repeat:3`).
- **Analysis.**
  - `tools/turing/census_bits.py`, unchanged, on the captures' SQLite exports, writes `int8_by_bits.md`.
  - The split is computed exactly as `call_split()` in `paper8/scripts/build_claims.py`.
  - The read rate is the card's rated bandwidth at the recorded memory clock: 432 GB/s × clock / 9001 MHz.
  - The primary analysis takes 89.3% of that rate, the same fraction as the 2080 Ti's assumed 550 of 616 GB/s. The
    sensitivity analysis takes 100%.
- **Fallback.** If the stock engine on Ada does not run the `exl3_gemv_int8_sq_kernel` family, the same per-bit fit is
  made on the matrix kernel family it does run. That change is reported as a deviation.

## Predictions

- **P1.** For every matrix role measured at both 1 and 2 bits, the 1-bit call's median time is at least 90% of the
  2-bit call's. On the 2080 Ti it was 96–100%.
- **P2.** The fitted 2-bit intercept is positive, and at the primary rate the fixed part is more than half of the
  fitted calls' excess over their reads.
- **P3.** P2's majority still holds at 100% of the rated bandwidth.

## What each outcome does to the paper

- **P1 and P2 hold:** P8 says the split holds on two GPU generations (Turing and Ada) and gives Ada's numbers.
  Section 4.3 and the abstract's limitation change. The title, scoped to the 2080 Ti, stays.
- **P1 holds and P2 fails:** P8 says 1-bit calls are no faster on both generations, and that the per-call share
  depends on the card.
- **P1 fails:** P8 says 1-bit calls are faster on Ada, and narrows "fewer bits would not make it faster" explicitly to
  Turing.

Anything else from the run (absolute times, kernel shares) is descriptive.

## Stops

- No run starts while another compute process holds the laptop GPU.
- The run stops if the GPU passes 87 °C.
- Laptop power, fan, clock and swap settings are untouched.
- Nothing is downloaded.

## Amendment 1 (registered 2026-11-05 10:49 EST, before the rerun)

Matt, 2026-11-05: "ok raise the laptop power limit and rerun it".

**Settings changed.**
- The platform profile was switched from balanced to performance with `powerprofilesctl set performance`, which
  needs no sudo.
- Under load, the GPU's memory clock now reaches 9,001 MHz, against 7,001 MHz in the first run (a 6-second probe ending at
  10:49 EST).
- The GPU power limit stays at 80 W. Raising it needs Dynamic Boost (`nvidia-powerd`, now disabled) or a sysfs
  write, and both need Matt's sudo. A third run follows if he enables it.

**Rerun.** The same census, tagged `d0-stock-mgemmn0-perf`, with the same analysis. The rated bandwidth comes from
the recorded memory clock, which should now give 432 GB/s.

**Predictions:**
- **A1.** The median memory clock during decode is 9,001 MHz.
- **A2.** With 29% more bandwidth, the 1-bit advantage shrinks: every role's 1-bit/2-bit time ratio is higher than in
  the first run (0.835–0.900).
- **A3.** P2 holds: the fixed per-call part is most of the excess at the primary rate.

**In the paper.** Both laptop runs are reported, with their memory clocks, as one test of whether fewer bits help
when memory is less of a bottleneck.

## Outcome of Amendment 1 (written 2026-11-05 10:52 EST, after the rerun)

The rerun is `testsuite/results/ada/census/d0-stock-mgemmn0-perf/` (split.md).

- **A1 failed.** During decode the memory clock again sat at 7,001 MHz: 95 of 110 busy samples, against 96 of 134 in
  the first run. Under the 80 W limit, the driver holds the memory clock below its maximum for this load, whatever the
  platform profile. The 9,001 MHz probe was a different load: one matrix-vector product at a lower SM clock.
- **A2 failed.** Unchanged within ±0.01: the ratios are 0.844–0.894, against 0.835–0.900. Three roles moved down and
  one moved up.
- **A3 held.** The fixed part is 98% of the excess at the primary rate and 64% at the rated rate.

So the rerun is a replicate of the first laptop run at the same memory clock. It reproduces it: a fixed 10.0 µs per
call against 9.7, and 1-bit calls 11–16% faster against 10–16%.

Testing faster memory needs the GPU's own limit raised (nvidia-powerd, and ASUS's dynamic boost from 5 to 25 W). That
needs Matt's sudo. The paper reports the replicate in one clause and keeps the statement that the laptop's memory
ran below its maximum.

## Amendment 2 (registered 2026-11-05 10:56 EST, before the raised-limit run)

Matt, 2026-11-05: "But we should actually test that before we publish the paper, right?"

**Change.** Matt runs one sudo command:
- `systemctl enable --now nvidia-powerd` turns on Dynamic Boost, which lets the GPU go above 80 W up to its 175 W
  maximum;
- writing 25 to `/sys/devices/platform/asus-nb-wmi/nv_dynamic_boost` raises ASUS's boost allowance from 5 W.

The platform profile stays performance. Before starting, the run records the enforced limit, the dynamic boost value
and powerd's state.

**Run.** The same census, tagged `d0-stock-mgemmn0-boost`, with the same analysis. The rated bandwidth comes from the
recorded memory clock.

**Predictions:**
- **B1.** The enforced power limit during decode is above 80 W, and the median memory clock during decode is
  8,001 MHz or more.
- **B2.** With more bandwidth, the 1-bit advantage shrinks: every role's 1-bit/2-bit ratio is higher than in the first
  run (0.835–0.900).
- **B3.** P2 holds: the fixed per-call part is most of the excess at the primary rate.

**If B1 fails** (the limit or the clock does not rise), the run is another replicate. The paper keeps its statement
that the laptop's memory ran below its maximum.

**If B1 holds,** the paper reports the run whichever way B2 goes:
- if B2 holds, as support for the bandwidth explanation;
- if it fails, as a finding that 1-bit calls stay faster on Ada at full memory speed, with the explanation narrowed.

## Outcome of Amendment 2 (written 2026-11-05 11:10 EST, after the raised-limit run)

The run is `testsuite/results/ada/census/d0-stock-mgemmn0-boost/` (power_state.txt, split.md). nvidia-powerd was
active, ASUS dynamic boost was 25 W and the profile was performance.

- **B1 held.** The enforced limit during decode was 175 W, and the memory clock sat at 9,001 MHz in all 78 busy
  samples. The SM clock's median was 2.3 GHz, against 1.5 GHz at 80 W. The GPU peaked at 76 C.
- **B2 failed, in the other direction.** The 1-bit advantage grew: the ratios are 0.732–0.809 (19–27% faster),
  against 0.835–0.900. Arithmetic rose 52% (the SM clock) and bandwidth 29%, so unpacking got cheaper relative to the
  read.
- **B3 held.** The fixed part is 5.5 µs per call at 2 bits: 86% of the excess at the primary rate and 52% at the rated
  rate.
- **Speed.** The profiled token took 21.9 ms, against 31.3 ms at 80 W: 43% more tokens per second.

**In the paper.** Section 4.3 reports all three laptop runs. It replaces the first explanation (the capped laptop
starved of bandwidth) with the one the runs support: whether fewer bits help tracks the card's arithmetic per byte of
bandwidth. The cost per call held in every run.
