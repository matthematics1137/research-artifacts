# Pre-registration: P8 long context on the 2080 Ti, stage 1: each KV cache format's measured reach

Registered 2026-11-01 21:50 EST by the ambassador. The first commit of this file freezes it. Later changes are dated amendments
appended at the end. It is built from the gate fork's draft (`research/prereg/2026-11-01-p8-64k-q4-cache-DRAFT.md`):
stage 1 is registered alone here, and stages 2-3 stay a draft until their item set is fixed (see the end of this
file).

- Gate: `research/gates/2026-11-01-p8-64k-q4-cache.md` (decision: proceed, reframed). RIFE: research-ledger-reference.
- Fit projection: `research/raw/2026-11-01-p8-64k-q4-cache/fit_projection.py` and `fit_projection.out.txt`.
- Matt's standing direction for P8 (2026-10-31): "we can talk about publishing p8 more later after incorporating all
  our lessons to see how far we can get it to go". This is the P6 lesson, taken capability first.

## Question

How long a filled context can the 2080 Ti serve with the adopted w13 build for each KV cache format, with at least
512 MiB of headroom? The lab has claimed since 2026-10-30 that 64K does not fit with the 8-bit cache. That claim is a
budget derived before any 2080 Ti run. Projected from the lab's measured memory readings, the 8-bit cache reaches
about 72K and the 4-bit cache about 132K with MTP on. This stage measures the reach directly.

## Numbers already seen (disclosure)

- **Memory readings** (nvidia-smi `memory.used` after decode, 32K cache, chunk 1024, 260 W):
  - final2, w13: MTP 8,991 MiB at fill 0 and 9,147 at 28K; undrafted 8,745 and 8,865;
  - rafatxf1, w12p: 4-bit cache 8,459 and 8,615 with MTP, 8,236 and 8,353 undrafted.
  - The TabbyAPI lanes' served peaks were also seen.
  - The fit projection is built from these readings.
- **Prefill timing:** stock 1.6.5 in the 2026-10-30 smoke ran 624 tokens/s at a 30K prompt (chunk 4096).
- **Never measured on the 2080 Ti:** a filled prefill beyond 30K.

## Method

- **Machine.** The desktop RTX 2080 Ti (11,264 MiB total), through the desktop queue only, at the 260 W default, with
  the desktop guard on.
- **Engine.** The adopted w13 build (exllamav3 1.6.5 plus the lab's patches; mode 2 for MTP), unchanged.
- **Server.** TabbyAPI through the lab's lane tooling, one fresh server process per step:
  - `cache_size` = `max_seq_len` = L;
  - `cache_mode` per format;
  - `max_batch_size` 1, chunk size 1024;
  - MTP on, with 6 draft tokens, drafted dynamically;
  - sparse draft head off; vision off.
  - The allocator configuration is identical to the anchors' (no expandable segments).
- **Formats:** exllamav3's native cache as `fp16`, `8,8`, `8,4` and `4,4` (key bits, value bits). Decode kernels are
  whatever the build selects: fd75d for `8,8`, Triton for the others.
- **Each step:**
  1. Start the server at length L.
  2. Send one filled prompt of L - 512 tokens with thinking off, at temperature 0. The prompt comes from P6's frozen
     generator (`testsuite/paper6/make_longctx.py`, imported unchanged), using one single-needle item scaled to the
     length, or LongMemEval-S filler built the same way. Content does not affect memory.
  3. Decode up to 64 tokens.
  4. Sample nvidia-smi `memory.used` at 1 Hz from server start to exit, and record the maximum. PyTorch's caching
     allocator keeps what it reserved, so a peak normally stays visible. If the engine released cached memory
     between samples, a peak could be missed; completing without an out-of-memory error still bounds it by the total.
  5. Record load time, prefill time, decode time, and the exit status or error text.
- **Ladder** (MTP on). Each format stops at its first failure:

| Format | Steps (L) |
|---|---|
| `8,8` | 64K, 72K, 80K |
| `8,4` | 64K, 88K, 96K |
| `4,4` | 64K, 96K, 128K, 136K |
| `fp16` | 32K, 36K, 40K |

- **Pass.** A step passes if the filled prefill and the decode both complete and the peak `memory.used` is at most
  11,264 - 512 = 10,752 MiB. A format's **reach** is its longest passing step.
- **Order.** `8,8`, then `4,4`, then `8,4`, then `fp16`. If the budget runs out, the formats that matter most to the
  question are already done.

## Predictions (derived; peak in MiB, low to high, from `fit_projection.py`)

| Step | Projected peak | Predicted |
|---|---|---|
| `8,8` 64K | 10,303-10,450 | pass |
| `8,8` 72K | 10,592-10,739 | pass (13 MiB margin at the high end) |
| `8,8` 80K | 10,881-11,028 | fail |
| `8,4` 64K | 9,759-9,906 | pass |
| `8,4` 88K | 10,422-10,569 | pass |
| `8,4` 96K | 10,643-10,790 | fail |
| `4,4` 64K | 9,237-9,384 | pass |
| `4,4` 96K | 9,849-9,996 | pass |
| `4,4` 128K | 10,461-10,608 | pass |
| `4,4` 136K | 10,614-10,761 | fail |
| `fp16` 32K | 10,167-10,186 | pass |
| `fp16` 36K | 10,439-10,474 | pass |
| `fp16` 40K | 10,711-10,762 | fail |

- **P1.** The measured peak of every completed step lies within -150 to +250 MiB of its projected range. Outside that
  band, the projection is refuted for that format.
- **P2.** The 8-bit cache passes at 64K, which refutes the 2026-10-30 claim.
- **P3.** The 4-bit cache reaches at least 128K.

## Rules after stage 1

- **K1.** If the `4,4` reach is below 96K, the line stops: the 4-bit cache does not buy the claimed capability on this
  card.
- **B1.** If `8,8` fails at 64K, the 2026-10-30 claim stands. Stage 2's paired comparison at 64K then moves to 48K.
- **Corrections if P2 holds.** Dated corrections, with the originals kept, go in:
  - `research/TURING_EXL3_RESEARCH_2026-10-30.md` §5.3;
  - `research/TURING_RELATED_WORK_2026-10-31.md` (borrow item 1);
  - the goal file's lessons map;
  - a thread note on RIFE research-ledger-reference.
- Measured prefill times set stage 2's budget and item counts.

## Safeguards (memory, other chats, heat)

- **One job at a time.** One GPU job runs at a time, through the desktop queue. The desktop guard stays on:
  - it pauses when a foreign process opens the GPU;
  - it keeps the 10 GiB MemAvailable floor for system RAM;
  - it keeps its thermal limits;
  - power stays at 260 W.
- **System RAM.** Each job uses under 4 GB of system RAM (`-rcs 2`), and no layer is offloaded to host RAM.
- **Failures.**
  - A CUDA out-of-memory error ends that step and counts as a failure. The queue does not retry it.
  - A step that hangs is stopped after 25 minutes, and its server process (ours) is ended.
  - Any rerun needs a dated amendment.
- **Budget:** at most 1.5 GPU hours. If the next step would exceed it, the stage stops and reports what it reached.

## Not covered here

- Accuracy at any length (stages 2 and 3).
- Decode kernels for the 4-bit or split cache.
- Prefill speed work.
- Host-RAM offload of the cache.

## Stages 2-3: draft, not registered

Before stages 2-3 are registered, the ambassador's review requires these changes to the draft:
1. **P6's ceiling and floor.** In P6 at 16K, the 2-bit 27B scored 50/50 on the single, multikey and multivalue
   families in every cell, and 4-12/50 on vartrack (`testsuite/results/paper6/analysis-2026-11-01/analysis.json`,
   `by_family_longctx`).
   - On those items a paired test would show near-zero discordance, so an "equivalent within ±5" verdict would be
     nearly guaranteed and would say little.
   - Stage 2 needs items that discriminate. The plan: keep P6's families as a cliff set, and add a graded set whose
     difficulty is calibrated on the reference arm only, with separate calibration items, so its accuracy sits in the
     middle. Claims are made per set.
2. **Stage 3's N rule.** The draft's minimum of 48 items contradicts its stop-at-budget rule. The budget governs, since
   stage 3 is descriptive.
3. **Noise discordance.** exllamav3's greedy decoding is not bit-reproducible on GDN hybrids (research-ledger-reference),
   so identical arms disagree on some items. Measure that rate, or disclose it and size N with it.
4. **Power.** Stage 1's prefill times fix N. At N below about 80 and 10% discordance, the ±5 test cannot conclude.
   Either raise the 64K budget or narrow the arms.

## Amendment 1 (2026-11-01 22:04 EST): implementation, fixed before any step ran

Written by the stage 1 fork before any GPU work. Nothing in the questions, the ladder, the predictions or the rules
changes. Code: `testsuite/p8_longctx/`; CPU tests (13) pass in zarael-sandbox.

1. **Allocator.** The registration says the allocator configuration is identical to the anchors' "(no expandable
   segments)". The parenthetical is wrong.
   - The anchors ran with exllamav3's default, which turns expandable segments on whenever PYTORCH_CUDA_ALLOC_CONF
     is unset: `_default_allocator_settings` in `exllamav3/__init__.py` of the stock 1.6.5 wheel and the w12p and
     w13 overlays.
   - No lab tool sets PYTORCH_CUDA_ALLOC_CONF, and the anchor rows record no EXL3_EXPANDABLE_SEGMENTS.
   - "Identical to the anchors'" governs. Stage 1 runs with PYTORCH_CUDA_ALLOC_CONF, PYTORCH_ALLOC_CONF and
     EXL3_EXPANDABLE_SEGMENTS unset and TabbyAPI's cuda_malloc_async off, so expandable segments are on, as in the
     anchors.
2. **Launcher.** The lab's lane tooling (`tools/turing/lanes.sh`) fixes cache_mode in its lane configs, which may not
   be edited, and TabbyAPI's config file beats its environment.
   - So `testsuite/p8_longctx/stage1_run.py` starts TabbyAPI with the command `lanes.sh` builds for TabbyAPI lanes:
     checkout b1c2253, the lab venv, the same boot line and environment.
   - It adds the w13 environment of queue step x-q-w13s2-mtp: PYTHONPATH=overlay/exl3-w13, EXL3_MGEMM_N_THRESHOLD=0,
     EXL3_FD75D_LIVE=0, EXL3_INT8_SQ_MAX_M=2, EXL3_GEMV_SK=2.
   - Each step gets its own config, written to the run folder.
   - The server runs as a child process inside the step's capped queue unit, so the guard counts it as the lab's.
3. **Config.** Copied from lane exl3-mtp except as registered (`stage1_config.py`):
   - cache_size = max_seq_len = L, and cache_mode per format: "8,8", "8,4", "4,4" or "FP16".
   - draft_cache_mode is the same format as cache_mode. The anchors' loader gives the MTP layer's cache the main
     cache's bits, and the projection assumes it.
   - sysmem_recurrent_cache is 2048 MB, TabbyAPI's spelling of census.py's -rcs 2.
   - Unchanged from the lane: chunk_size 1024; max_batch_size 1; recurrent_checkpoint_interval_pp 1024; MTP with
     draft_num_tokens 6, dynamic_draft true and sparse_draft_head 0; vision off.
4. **Prompts** (`build_stage1_prompts.py`, manifest `testsuite/p8_longctx/stage1_prompts_manifest.json`).
   - One "single" item per length from P6's generator, imported unchanged. The wrapper sets the module's SEED to
     "p8-longctx-stage1", TARGET_TOKENS so that the no-think prompt is at most L - 512 tokens, and the tolerance
     bounds as P6 derives them.
   - The prompts were built on the laptop with the model's tokenizer.json, whose SHA-256 the generator checks.
   - The filler stops up to 48 tokens short of its budget, so the built prompts are 15-50 tokens under L - 512. One
     prompt per length is shared across formats.
   - The 1K harness-check prompt uses a 6% tolerance instead of P6's 1%, which is narrower than the filler's slack
     at that size.
   - The server's reported prompt_tokens are recorded as the check on the token counts.
5. **Harness check.** Before the ladder, one check starts the 8,8 server at L = 16K and sends the 1K prompt with a
   16-token decode, to catch configuration errors. It is not a ladder step: its memory is recorded and not judged, and
   its time counts against the budget.
6. **Failure classes.** A step fails when:
   - (a) the load, prefill or decode reports a CUDA out-of-memory error;
   - (b) it completes with its peak above 10,752 MiB;
   - (c) it hangs past 25 minutes from server start.

   Any other error is a harness error: a rejected config, an HTTP error with no out-of-memory cause, or a server exit
   with none. The stage then stops with no verdict for that step, and the error is reported. A thermal stop (item 10)
   is a safety stop, not a verdict.
7. **Budget rule.** The 1.5 GPU hours run from the harness check's server start: wall time from each server start to
   its memory settling, summed over queue tries. Before each step, its time is estimated as the last load time, plus
   the last measured prefill scaled by (L / its prompt length)^1.5, plus 2 minutes. The stage stops if that estimate
   would pass 90 minutes.
8. **Between steps.** Our server is stopped with SIGTERM, then SIGKILL after 60 s. The next step starts only after
   memory.used is back within 200 MiB of that step's baseline, or after 90 s (recorded). Each step records its baseline
   memory.used and the compute processes on the GPU. sunshine, the desktop's game-streaming server, holds about 160
   MiB, and was resident during the anchors too.
9. **Interruptions.** If the guard stops the queue step mid-step, the queue retries it. Steps with a recorded verdict
   are not rerun. A step interrupted before its verdict is run again and listed in the report as interrupted.
10. **Also recorded:** the server's peak resident memory, GPU temperature, power and SM clock, all at 1 Hz. The runner
    stops our server and the stage if the GPU reads 86 °C or more on three samples running (the guard's own limit is
    87 °C).
11. **Power.** The step does not use sudo. It checks that the power limit reads 260 W and stops if it does not.
