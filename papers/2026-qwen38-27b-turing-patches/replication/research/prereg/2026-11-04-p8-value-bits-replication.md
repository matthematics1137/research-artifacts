# Pre-registration: P8, replicating the 64K value-bits result

Registered 2026-11-04 04:12 EST by the P8 fork, at the ambassador's request. Matt approved the replication before P8's release.
Gate record: `research/gates/2026-11-04-p8-value-bits-replication.md`. The first commit of this file freezes it; later
changes are dated amendments appended at the end. No arm below has run, and no model has seen the fresh items.

## Question

With the decode kernel held fixed, the split cache (`8,4`: 8-bit keys, 4-bit values) scored 11.5 points above the
8-bit cache at 64K. That is K2 in `research/prereg/2026-11-03-p8-fd75d-accuracy-arm.md`:
- Triton read both caches, on the 96 graded 64K items;
- interval +5.2 to +18.8;
- 14 items right only with the split cache, 3 only with the 8-bit cache;
- exact McNemar p 0.0127.

The lead has no precedent in prior work, and it rests on one item set. This registration asks:
1. **Does it repeat on fresh items?** (R1, primary)
2. **What does a harmless numerical change do?** How many answers does it flip on the same items, the null that stage
   2's bit-identical noise check could not give? (N)
3. **Does it hold against fp16?** Does the split cache also lead fp16 at 32K? (R2)
4. **What does a 6-bit value cache do?** (R3)

## What is known before the run

- **Stage 2's noise check was bit-identical.** The 8-bit arm, rerun on the first 16 graded items in a fresh server,
  gave identical texts (0 of 16 discordant). It measured determinism, not spread.
- **The split cache on those 16 items.** It disagreed with the 8-bit cache on 6 (4 for it, 2 against; 37.5%), outside
  that check's Wilson interval of 0-19.4%. On all 96 items it disagreed on 17 (17.7%), inside it. (Number check of
  2026-11-04, F14, from the stage 2 journal.)
- **The kernel swap.** Changing the decode kernel on the 8-bit cache (K1) changed 3 of 96 texts and 2 outcomes, one
  in each direction.
- **The two lengths.** At 32K the 8-bit cache (fd75d) and fp16 gave identical scores on all 96 items, and identical
  texts on 94 (H1).

## Arms, in priority order

Every arm uses stage 2's server settings:
- the w13 build and environment;
- TabbyAPI through `stage2_run.py`, one fresh server per arm;
- `cache_size` = `max_seq_len` = the arm's length; chunk 1024; `max_batch_size` 1;
- MTP with 6 dynamic drafts, with the draft cache in the arm's format; `sysmem_recurrent_cache` 2048;
- thinking off, temperature 0 (greedy), at most 768 output tokens.

| Priority | Arm | Format, length | Items | Kernel and environment |
|---|---|---|---|---|
| R1 | `rep-64k-88-triton` | `8,8`, 64K | fresh `rep-64k` (96) | Triton (`EXL3_FD75D=0`) |
| R1 | `rep-64k-84` | `8,4`, 64K | fresh `rep-64k` (96) | Triton (the build's only reader of `8,4`) |
| N | `noise-64k-88-triton-qcs128` | `8,8`, 64K | original `graded-64k` (96) | Triton, plus `EXL3_QC_STAGING_PAGES=128` |
| R2 | `rep-32k-84` | `8,4`, 32K | original `graded-32k` (96) | Triton |
| R2 | `rep-32k-fp16-check` | `fp16`, 32K | `graded-32k`, the first 16 | as stage 2 |
| R2 | `rep-32k-fp16` | `fp16`, 32K | `graded-32k` (96) | only if the check does not reproduce |
| R3 | `rep-64k-86` | `8,6`, 64K | original `graded-64k` (96) | Triton (the build's only reader of `8,6`) |

### The fresh items (R1)

- `rep-64k.jsonl` keeps the graded design and changes only the seed text, now "p8-value-bits-replication-2026-11":
  - 96 multi-value items at 12 values each, at 64K;
  - indices 0-95 on the 96-item depth grid;
  - P6's generator unchanged, with the wrapper's checks.
- Ids carry a `rep-` prefix.
- They were built by `testsuite/p8_longctx/build_replication_items.py` with the environment that rebuilds the original
  items byte for byte. Manifest: `testsuite/p8_longctx/manifests/rep-64k.json`, SHA-256 26b5d9b4c689eaae....
- Prompts run 64,895-64,944 tokens. The set shares no needle value with the original items.

### The noise arm (N)

- **Why this knob.** The decode split count has no knob in exllamav3 1.6.5 (it follows the SM count), and the lab's
  overlay adds none for Triton. The other `EXL3_*` knobs switch kernels instead of reordering a sum: `EXL3_GEMV_SK`,
  `EXL3_BC_ATTN`, `EXL3_FA75` and `EXL3_SDPA_PREFILL`.
- **What it does.** `EXL3_QC_STAGING_PAGES` is exllamav3 1.6.5's own setting for the quantized cache's prefill
  staging slice. The default, 256 pages (64K tokens), stages each 64K prompt's window in one slice. At 128, every
  prefill window longer than 32K tokens is staged in two slices: the same fa75 kernel attends each slice, and the two
  are merged through fp32 partial sums. The formats, kernels and precision stay the same; the order of the prefill
  attention sums changes for the second half of each prompt, and so does everything computed from it. Decode is not
  touched. Stage 3's 96K arm and stage 1's longer steps already ran this sliced path.
- **The limit.** N bounds the spread of a rounding-level change. That change is far smaller than quantizing values to
  4 bits.

### Reusing stage 2's fp16 arm (R2)

- `rep-32k-fp16-check` reruns fp16 at 32K on the first 16 items in a fresh server.
- **If all 16 answer texts equal stage 2's `s2-32k-fp16` texts exactly,** stage 2's arm is R2's fp16 reference, and
  the full rerun is skipped. The build, settings and items are the same, and greedy decoding reproduced across server
  sessions in every earlier check:
  - stage 2's noise check, 16 of 16 texts;
  - the Triton arm against the fd75d rerun, 16 of 16.
- **If any text differs,** `rep-32k-fp16` reruns all 96 items, and R2 uses the rerun.

### The 6-bit value arm (R3)

- **Reach.** Not measured. An `8,6` cache takes 1,920 bytes per token per layer (31.9 MiB per 1K tokens with the MTP
  layer), less than `8,8`'s 2,176 (36.1) at every length. So the 8-bit cache's 64K measurement bounds it: a peak of
  10,231 MiB when Triton read it.
- **The harness.** It treats `8,6` as inside the reach that `8,8` measured.

## Budget and stop rule

- **Total.** 15 GPU hours, as group `replication`.
- **Registered per-item estimates:**

  | Arm | Estimate | From the earlier runs |
  |---|---|---|
  | 64K | 117 s | Triton `8,8` 116.7 s, split cache 116.0 s |
  | `8,4` at 32K | 55 s | `4,4` read by Triton at 32K, 54.5 s |
  | fp16 at 32K | 62 s | 61.9 s |

- **The whole-arm rule.** An arm starts only if its whole estimate (items × estimate + 120 s) fits the time left. If it
  does not, that arm and every later arm are skipped, so the cut falls from the end of the priority order.
- **The hard cap.** Stage 2's per-item rule stays: before each item, the arm's mean so far must fit, or the arm and the
  plan stop.
- **Expected:**
  - with reuse: R1 6.3 h, N 3.2 h, R2 1.8 h and R3 3.2 h, about 14.5 h with every arm;
  - with the fp16 rerun: R2 takes 3.5 h, and R3 is cut.

## Comparisons and decision rules

The statistics are those of stages 2-3:
- 90% percentile bootstrap intervals over items: 10,000 resamples, `random.Random(20261102)`, linear interpolation;
- exact McNemar counts and p-values; Wilson 95% intervals for single proportions;
- the scorer is P6's `score_longctx` through `stage2_stats.score`, unchanged;
- "equivalent" means the whole interval lies within ±5 points.

### R1 (primary)

D1 = accuracy(`rep-64k-84`) − accuracy(`rep-64k-88-triton`), paired on the 96 fresh items.

| Outcome | Condition | Meaning |
|---|---|---|
| **Replicated** | lower bound > 0 | the split cache scores higher on fresh items; the report says whether the interval also lies above +5 |
| **Equivalent** | whole interval within ±5, lower bound ≤ 0 | no value-bits effect of the claimed size: K2 is not replicated |
| **Not replicated** | lower bound ≤ 0 and the interval reaches beyond ±5 | unresolved; if the upper bound is < 0, reported as reversed |

### N

- **The paired difference.** D_N = accuracy(N) − accuracy(the accuracy arm's `s2-64k-88-triton`), paired on the 96
  original items. Reported with its interval, discordant counts and text identity.
- **Registered use.** K2's 17 discordant items are **beyond the harmless spread** if the Wilson 95% upper bound of N's
  discordance (out of 96) is below 17/96. Otherwise they are **within it**. The same comparison is reported on the
  first 16 items: N's discordance there against the split cache's 6 of 16.

### R2

- D2 = accuracy(`rep-32k-84`) − accuracy(fp16 at 32K), paired on the 96 original 32K items. The fp16 reference is
  stage 2's arm or the rerun, as above.
- Outcomes as for R1.
- Descriptive: `rep-32k-84` against stage 2's fd75d-read 8-bit arm at 32K.

### R3 (descriptive)

- D3a = accuracy(`rep-64k-86`) − accuracy(`s2-64k-88-triton`): 6-bit against 8-bit values.
- D3b = accuracy(`s2-64k-84`) − accuracy(`rep-64k-86`): 4-bit against 6-bit values.
- Both on the 96 original items, with Triton reading every arm.
- Reported with intervals. Monotone if both point estimates are ≥ 0; no claim beyond the description.

### Pairing across server sessions

N, R2 with reuse, and R3 compare a new arm with one run on 2-4 November in an earlier session, with the same build,
settings and items. Greedy reruns reproduced exactly in every earlier check, which is what makes the pairing valid; R2's
check tests it again.

## Predictions (before any data)

- **R1:** regression toward 0. D1's point estimate lies between 0 and +6 points, below K2's +11.5. The outcome is
  "equivalent" or "not replicated" rather than "replicated".
- **N:** equivalent (D_N within ±5), with at most 6 of 96 items flipped in both directions, about as few as the kernel
  swap's 2.
- **R2:** D2 equivalent.
- **R3:** D3a equivalent.

## What the outcome changes in P8

- **Replicated:** P8 may report the value-bits lead as replicated on fresh items, at the size R1 measures.
- **Equivalent or not replicated:** P8 reports K2 as a single-set observation that did not replicate. Its title,
  abstract and conclusion do not rest on it.
- **Either way,** N's spread enters P8 as the null for K2, and R2 and R3 as descriptive context.

## Safeguards

As in stage 2:
- one GPU job at a time through the desktop queue, with the guard on, at 260 W and without sudo;
- the 20 GiB unit cap and the 10 GiB system-RAM floor;
- no retry after an out-of-memory error;
- a 25-minute stop for a hung request, with the item excluded and listed.

## Implementation (CPU-tested before the run: `test_stage2.py`, TestReplication)

- `build_replication_items.py` builds `rep-64k`. `items.py` and `build_stage2_items.py` are unchanged, so the earlier
  manifests' code hashes still match.
- `stage2_run.py` gains:
  - the group budget;
  - the registered per-item estimates (`if_budget.per_item_s`);
  - `cut_from_end`;
  - `skip_if_reproduced`;
  - `8,6` in the format map for stages 2-3 only, with stage 1's module unchanged, and reach 72K by the bound above.
- `plans/replication.json` holds the arms in the order above.
- The analysis follows this text with `stage2_analyze`'s arms and comparisons. Its script is written before any arm's
  answers are read.
