# Pre-registration DRAFT: P8, long context on the 2080 Ti: each KV cache's measured reach and its accuracy at depth

**Status: DRAFT, not registered.** Written 2026-11-01 21:44 EST by the gate fork for review. Nothing here has run. To register it,
the reviewer renames it without `-DRAFT` and commits; that first commit freezes it. Later changes are dated amendments
appended at the end.

- Gate: `research/gates/2026-11-01-p8-64k-q4-cache.md`. RIFE: research-ledger-reference.
- Prior art and fit: `research/prior-art-p8-64k-q4-cache-2026-11-01.md`;
  `research/raw/2026-11-01-p8-64k-q4-cache/fit_projection.py` and `.out.txt`.
- Matt's standing direction for P8 (2026-10-31): "we can talk about publishing p8 more later after incorporating all
  our lessons to see how far we can get it to go". This is the P6 lesson, taken capability first.

## Why the frame changed from "64K"

The gate opened on the premise that 64K does not fit with the 8-bit cache. That premise is a 2026-10-30 budget
derived before the port ran. Projected from the lab's measured 2080 Ti memory anchors, the 8-bit cache reaches about
72K and the 4-bit cache about 132K (MTP on, at least 512 MiB headroom). So this registration:
- measures each cache's reach first (stage 1);
- compares 4-bit against 8-bit at 64K, where both should fit, and 8-bit against fp16 at 32K, where fp16 still fits
  (stage 2);
- maps the 4-bit cache's accuracy out to about 128K (stage 3).

## Common rules

- **Pre-registration discipline.** Predictions are frozen by the first commit. Timestamps come from the clock inside
  the writing command. A falsified prediction is reported as falsified. Deviations are disclosed and any substitute
  analysis is labelled exploratory. Stages run in order; each starts only after the previous stage's report is
  committed.
- **Machine.** Desktop RTX 2080 Ti (11,264 MiB) through the desktop queue only, at the 260 W default, with the desktop
  guard on.
- **Engine.** The adopted w13 build (exllamav3 1.6.5 + lab patches; mode 2 for MTP), unchanged.
- **Runtime settings:** chunk_size 1024; `-rcs 2` (system RAM); MTP on with 6 draft tokens, dynamic (`-ndt 6 -dds`);
  sparse draft head off; vision off.
- **Cache formats:** exllamav3's native cache as `fp16`, `8,8`, `4,4` and `8,4` (keys, values).
  - Decode kernels are what the build selects: fd75d for `8,8`, Triton for the others. fdq4 is not part of this
    registration.
- **Harness.** TabbyAPI through the lab's lane tooling, with one server config per arm:
  - `max_seq_len` = `cache_size` = the stage length + 512, rounded up to 256; `cache_mode` per arm; `max_batch_size` 1.
  - Requests use P6's long-context settings: thinking off (`chat_template_kwargs: {"enable_thinking": false}`, checked
    on 3 items before each arm), temperature 0, max 128 tokens.
  - Every item is a fresh prompt; there is no prefix reuse between items.
- **Never edited:** the P6 harness (`testsuite/paper6/*`, `research/design-kv-allocation.md`) and the lab's tabby lane
  configs. New code goes in `testsuite/p8_longctx/`; results go in `testsuite/results/turing/longctx/`.

## Numbers already seen (disclosure)

- **Memory anchors:** final2 and rafatxf1 rows (w13 and w12p, 8-bit and 4-bit, fills 0 and 28K, 32K cache), the TabbyAPI
  lanes' served peaks, and the fit projection built from them.
- **Speed and timing:**
  - stock 1.6.5 prefill in the 2026-10-30 smoke: 624 t/s at a 30K prompt, 707 at 15K, chunk 4096;
  - R4 attention times;
  - rafatxf1: the adopted build on a 4-bit cache ran 25.9% slower at 28K than rafatxf's build with fdq4.
- **Quality:**
  - rafatxf's needle results with a Q4 cache at 4.0 bpw (25/25 at 32K-250K);
  - P6's results: q4_0 cost at most 1.5 points in any 16K cell; I_hybrid +3.68 points.
- **Not yet measured by the lab:** no accuracy of exllamav3's quantized cache at 32K or beyond on this model, and no
  filled prefill beyond 30K on the 2080 Ti.

## Stage 1: measured reach (budget at most 1.5 GPU hours)

**Steps.** Each runs in a fresh server process:
1. Start the arm's server at length L.
2. Send one filled prompt of about L - 512 tokens: a stage-1 haystack built like the stage 2 items, with one question.
3. Decode up to 64 tokens.
4. Throughout, sample nvidia-smi `memory.used` at 1 Hz and record the maximum.
5. Record prefill and decode times, and the exit status.

**Ladder** (MTP on). Each format's ladder stops at its first failure:

| Format | Steps (L) |
|---|---|
| `8,8` | 64K, 72K, 80K |
| `8,4` | 64K, 88K, 96K |
| `4,4` | 64K, 96K, 128K, 136K |
| `fp16` | 32K, 36K, 40K |

**Pass.** A step passes if the filled prefill and the decode complete and peak `memory.used` is at most 11,264 - 512 =
10,752 MiB. **Reach** is the longest passing step.

**Predictions** (derived; peak in MiB, low to high, from `fit_projection.py`):

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

**Prediction P1.** The measured peak of every completed step lies within -150 to +250 MiB of its projected range.
Outside that band, the projection is refuted for that format.

**Kill and branch rules:**
- **K1.** If the `4,4` reach is below 96K, stop the line and report that the 4-bit cache does not buy the claimed
  capability on this card.
- **B1.** If `8,8` fails at 64K, the original premise stands. Stage 2's 64K comparison then moves to 48K (8-bit
  against 4-bit, paired), and 4-bit at 64K joins stage 3's ladder as descriptive.
- Measured prefill times set stage 2's item counts.

## Stage 2: paired accuracy at depth (at most 12 GPU hours; conditional rescue at most 4)

**Items.**
- A new wrapper imports P6's frozen generator `testsuite/paper6/make_longctx.py` unchanged. It sets the module's
  `TARGET_TOKENS` (and the derived tolerance bounds) to 32,768 and 65,536, with seed SHA-256("p8-longctx-2026-11").
- The four P6 families are balanced: single, multikey, multivalue, vartrack.
- Haystacks are LongMemEval-S sessions (MIT, SHA-256 verified).
- Items are built once before any measurement. The manifest is hashed and committed, and the generator's self-test
  must pass at both lengths.

**N rule.**
- Per length, N = min(100, floor(B × 60 / (A × t))), rounded down to a multiple of 4.
- B is the stage budget in hours: 4 at 32K, 8 at 64K. A is the number of arms at that length.
- t is stage 1's measured prefill time at that length × 1.1, plus 10 s, in minutes.
- N is never below 48. If it would be, run 48 and report the overrun.
- **Power (for the reviewer before registering).** With N near 80 and paired discordance at or below 5%, the 90%
  interval is about ±4 points, so equivalence at ±5 is reachable. At 10% discordance it is about ±6 points, and the
  test would be inconclusive at any true difference. If stage 1's prefill times put N at 64K below 80, consider
  raising B there.

**Arms:**

| Length | Arms (paired on the same items) |
|---|---|
| 32K | `fp16`, `8,8`, `4,4` |
| 64K | `8,8`, `4,4` |
| 64K, conditional | `8,4` (only if H2 fails) |

**Scoring.** P6's `score_longctx` (think-stripped, exact on the required parts), 0 or 1 per item.

**Hypotheses.** Each is a paired difference in accuracy points. Equivalence means the 90% percentile-bootstrap interval
(10,000 resamples over items, seed 20261102) lies inside ±5 points, as TOST at α 0.05 per side:

| ID | Role | Comparison | Length |
|---|---|---|---|
| H2 | primary | `4,4` minus `8,8` | 64K |
| H1 | secondary | `8,8` minus `fp16` (the deployed default) | 32K |
| H3 | secondary | `4,4` minus `fp16` | 32K |

Holm applies across H1 and H3. Exact McNemar p-values and discordant counts are reported for each.

**Predictions:**
- H2 holds (from rafatxf's needles at 4.0 bpw, Minima-KV at 64K and exllamav3's rotation).
- H1 holds.
- H3 holds.
- Any failure is a cliff, not a uniform loss: it concentrates in multikey and vartrack (where keys must
  discriminate).

**Rules after stage 2:**
- **What "fails" means.** H2 fails when its 90% interval's lower bound is below -5, so the 4-bit cache could cost more
  than 5 points. An interval above +5 is reported, not treated as failure.
- **Rescue.** If H2 fails, run `8,4` at 64K on the same items. H2s is `8,4` minus `8,8` within ±5.
- **K2.** If H2 and H2s both fail, no 4-bit or split mode is adopted. The line stops after stage 2.
- **F1.** If H1 fails, the deployed 8-bit default gets its own follow-up. It is reported, not fixed here.

## Stage 3: the 4-bit cliff (at most 8 GPU hours; only if H2 holds)

- **Items.** New items at 96K and 128K from the same wrapper, capped at stage 1's `4,4` reach minus 1K. Lengths beyond
  the reach are dropped. N3 = min(48, the N rule with B = 4 per length).
- **Arm.** `4,4` only; there is no on-card reference at these lengths.
- **Report (descriptive).** Accuracy by length (32K, 64K, 96K, 128K) with Wilson 95% intervals, and the usable
  context. Usable context is the longest length whose accuracy lies within 5 points of the format's 32K accuracy, with
  the gap's 95% interval given.
- **Prediction P3.** 4-bit accuracy at 128K is within 10 points of its 32K accuracy.
- No equivalence claim is made beyond 64K.

## Secondary, descriptive only

At each format and length:
- decode tok/s, pooled as total tokens over total time;
- prefill tok/s;
- peak memory;
- per-family accuracy.

The 4-bit decode runs on Triton here. Its speed against fd75d is not a claim of this registration.

## Safeguards (memory, other chats, heat)

- **Fit first.** Stage 2 and 3 arms run only at format-length steps that passed stage 1 with at least 512 MiB headroom,
  in the same harness and settings.
- **One job at a time.** One GPU job runs at a time, through the desktop queue. The desktop guard stays on:
  - it pauses when a foreign process opens the GPU;
  - it keeps the 10 GiB MemAvailable floor for system RAM;
  - it keeps the thermal limits;
  - the power stays at 260 W.
- **Memory settings.** Each job uses under 4 GB of system RAM (`-rcs 2`), and no arm offloads layers to host RAM. The
  allocator configuration is identical to the anchors' (no expandable segments).
- **Out-of-memory handling.** A CUDA out-of-memory error ends that step. The queue does not retry it; it is recorded as
  a stage 1 failure, or it stops the stage 2 or 3 arm with its journal kept. Any rerun needs a dated amendment.
- **Journaling.** Every item's result is journaled as it completes, so a stopped arm resumes exactly.

## GPU-hour budget

| Stage | Budget |
|---|---|
| Stage 1 | at most 1.5 h |
| Stage 2 | at most 12 h, plus at most 4 h for the conditional rescue |
| Stage 3 | at most 8 h |
| Total | at most 25.5 h at 260 W (at most about 6.6 kWh) |

An arm that would exceed its stage budget stops at the budget and reports the N it reached.

## Exclusions

An item is excluded only for:
- a server error unrelated to memory, such as a template rejection;
- a single request over 20 minutes.

Exclusions are counted and listed per arm.

## Not covered

- Decode kernels: borrowing fdq4 (MIT, rafatxf), a 4-bit read path in fd75d, or a split-cache kernel.
- Prefill speed: rafatxf's fa75.
- A short-context T4 (GSM8K-500) for making the 4-bit cache the default.
- Host-RAM offload of the cache, which is the P5 lesson.

Each needs its own gate or registration.

**Superseded (2026-11-01 22:49 EST):** stage 1 of this draft was registered as `research/prereg/2026-11-01-p8-longctx-stage1.md`
(c75ee2e), and stages 2-3 as `research/prereg/2026-11-01-p8-longctx-stages2-3.md`, with a graded item set calibrated on
the reference arm. This draft stays as the gate fork wrote it.
